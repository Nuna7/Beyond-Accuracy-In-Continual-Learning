import copy, os, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base_trainer import BaseTrainer
from sae.offline_trainer import OfflineSAETrainer, SAEBank

HOOK_LAYERS = ['layer1', 'layer2', 'layer3', 'layer4']
HOOK_DIMS   = {'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512}

class SAEFSGPTrainer(BaseTrainer):

    def __init__(self, model, cfg):
        super().__init__(model, cfg)
        self._sae_bank        = SAEBank()
        self._old_models      = []
        self._last_imgs       = None
        self._last_ce         = 0.0
        self._last_sel        = 0.0
        self._ncm_prototypes  = {}
        self._eval_mode       = 'fc'

    def _sae_cfg(self, d_in: int) -> dict:
        c = self.cfg.sae
        k = max(8, d_in // 16) if c.topk_by_layer else c.topk
        return dict(d_in=d_in, expansion=c.expansion_factor, k=k,
                    auxk_alpha=c.aux_loss_coeff, lr=c.lr,
                    epochs=c.epochs, batch_size=c.batch_size,
                    device=str(self.device), verbose=False)

    @torch.no_grad()
    def _build_task_mask(self, sparse_codes: torch.Tensor) -> torch.Tensor:
        mode          = getattr(self.cfg.method, 'fsgp_mask_mode', 'activity')
        d_sae         = sparse_codes.size(1)
        mask          = torch.zeros(d_sae, dtype=torch.bool)
        k_sel         = max(1, int(self.cfg.sae.topk))

        if mode == 'activity':
            for code in sparse_codes:
                mask[code.abs().topk(k_sel).indices] = True

        elif mode == 'discriminative':
            C         = sparse_codes.size(0)
            top_k_pc  = getattr(self.cfg.method, 'fsgp_topk_per_class', 16)
            sigma_all = sparse_codes.std(0).clamp_min(1e-6)
            for c in range(C):
                mu_c = sparse_codes[c]
                if C > 1:
                    others   = torch.cat([sparse_codes[:c], sparse_codes[c+1:]])
                    mu_other = others.mean(0)
                else:
                    mu_other = torch.zeros_like(mu_c)
                score   = (mu_c - mu_other).clamp(min=0) / sigma_all
                k_eff   = min(top_k_pc, d_sae)
                mask[score.topk(k_eff).indices] = True
        else:
            raise ValueError(f"Unknown fsgp_mask_mode: {mode}")

        return mask

    @torch.no_grad()
    def _run_old_model_l4(self, loader: DataLoader, old_model) -> torch.Tensor:
        old_model.to(self.device).eval()
        buf = []
        for imgs, _ in loader:
            old_model(imgs.to(self.device))
            buf.append(old_model.hooks['layer4'].cpu())
        old_model.to('cpu')
        return torch.cat(buf).float()

    def after_task(self, task_id: int, train_loader: DataLoader):
        print(f"\n  [SAE-FSGP] Post-task {task_id}: collecting activations …")
        act_dict, labels_cpu = self.collect_all_activations(train_loader)
        labels_dev = labels_cpu.to(self.device)
        classes_per_task = self._last_classes_this_task
        task_start = self.seen_classes - classes_per_task

        if task_id > 0 and self._old_models and self._ncm_prototypes:
            new_mean = act_dict['layer4'].float().mean(0).cpu()
            old_acts = self._run_old_model_l4(train_loader, self._old_models[-1])
            old_mean = old_acts.mean(0)
            delta = new_mean - old_mean
            print(f"[SDC] delta_norm={delta.norm():.4f} new_mean_norm={new_mean.norm():.4f} old_mean_norm={old_mean.norm():.4f}")
            for c in self._ncm_prototypes:
                self._ncm_prototypes[c] = self._ncm_prototypes[c] + delta
            print(f"  [NCM drift] delta_norm={delta.norm():.4f} "
                  f"| corrected {len(self._ncm_prototypes)} prototypes")

        print(f"  [SAE-FSGP] Training SAEs …")
        for layer in HOOK_LAYERS:
            d_in = HOOK_DIMS[layer]
            acts = act_dict[layer].to(self.device)

            sae = OfflineSAETrainer(**self._sae_cfg(d_in)).fit(acts)
            self._sae_bank.add_sae(task_id, layer, sae)

            dense_protos, sparse_codes = [], []
            for c_local in range(classes_per_task):
                c_global = task_start + c_local
                mask_c = (labels_dev == c_global)
                if mask_c.sum() == 0:
                    dense_protos.append(torch.zeros(d_in))
                    sparse_codes.append(torch.zeros(sae.dict_size))
                    continue
                mu_c = acts[mask_c].float().mean(0)
                z_c = sae.encode(mu_c.unsqueeze(0)).squeeze(0).cpu()
                dense_protos.append(mu_c.cpu())
                sparse_codes.append(z_c)
                if layer == 'layer4':
                    self._ncm_prototypes[c_global] = mu_c.float().cpu()

            dense_stack = torch.stack(dense_protos)
            sparse_stack = torch.stack(sparse_codes)
            task_mask = self._build_task_mask(sparse_stack)
            self._sae_bank.add_prototypes(
                task_id, layer, dense_stack, sparse_stack, task_mask)

            k_val = sae.k.item() if hasattr(sae.k, 'item') else sae.k
            print(f"    layer={layer:8s} | d={d_in}→{sae.dict_size} "
                  f"k={k_val} | mask: {int(task_mask.sum())}/{sae.dict_size}")
            sae.to('cpu')

        snap = copy.deepcopy(self.model)
        snap.eval()
        for p in snap.parameters():
            p.requires_grad_(False)
        snap.to('cpu')
        self._old_models.append(snap)
        print(f"  [SAE-FSGP] Snapshot saved (n_old={len(self._old_models)}) "
              f"| NCM classes: {len(self._ncm_prototypes)}")
        self._eval_mode = 'ncm'

    @torch.no_grad()
    def _eval_accuracy(self, loader: DataLoader) -> float:
        print(f"[EVAL] eval_mode={self._eval_mode} n_protos={len(self._ncm_prototypes)}")
        if self._eval_mode == 'fc' or len(self._ncm_prototypes) < 2:
            return super()._eval_accuracy(loader)
        self.model.eval()
        classes      = sorted(self._ncm_prototypes.keys())
        proto_matrix = F.normalize(
            torch.stack([self._ncm_prototypes[c].float() for c in classes]), dim=1
        ).to(self.device)

        correct = total = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            self.model(imgs)
            feats = F.normalize(self.model.hooks['layer4'].float(), dim=1)
            preds = (feats @ proto_matrix.T).argmax(1)
            pred_labels = torch.tensor(
                [classes[p] for p in preds.cpu().tolist()], device=self.device)
            correct += (pred_labels == labels).sum().item()
            total   += labels.size(0)
        return 100.0 * correct / max(total, 1)

    def compute_loss(self, logits, targets, task_id):
        ce = F.cross_entropy(logits, targets)
        if task_id == 0:
            self._last_ce  = ce.item()
            self._last_sel = 0.0
            return ce

        sel_loss    = torch.tensor(0.0, device=self.device)
        dense_loss  = torch.tensor(0.0, device=self.device)
        λ_layers    = self.cfg.method.fsgp_lambda
        use_sae     = self.cfg.method.fsgp_use_sae
        norm_proj   = getattr(self.cfg.method, 'fsgp_norm_proj', True)
        λ_dense     = getattr(self.cfg.method, 'fsgp_dense_lambda', 0.0)

        t_old        = len(self._old_models) - 1
        old_model    = self._old_models[-1].to(self.device)
        with torch.no_grad():
            old_model(self._last_imgs)

        for l_idx, layer in enumerate(HOOK_LAYERS):
            λ     = λ_layers[l_idx] if l_idx < len(λ_layers) else 1.0
            h_new = self.model.hooks[layer]
            h_old = old_model.hooks[layer].detach()

            if use_sae:
                sae  = self._sae_bank.get_sae(t_old, layer)
                info = self._sae_bank.get_prototypes(t_old, layer)
                if sae is not None and info is not None:
                    V        = sae.to(self.device).decoder.weight[
                                   :, info['mask'].to(self.device)].detach()
                    p_new = h_new @ V
                    p_old = h_old @ V
                    if norm_proj:

                        cos_sim_batch = F.cosine_similarity(p_new, p_old, dim=1)
                        sel_loss = sel_loss + λ * (1 - cos_sim_batch).mean()
                    else:

                        sel_loss = sel_loss + λ * F.mse_loss(p_new, p_old)
                    sae.to('cpu')
                else:
                    print(f"[SEL MISS] layer={layer} sae_is_none={sae is None} info_is_none={info is None}")
            else:
                if norm_proj:
                    cos_sim_batch = F.cosine_similarity(h_new, h_old, dim=1)
                    sel_loss = sel_loss + λ * (1 - cos_sim_batch).mean()
                else:
                    sel_loss = sel_loss + λ * F.mse_loss(h_new, h_old)

            if layer == 'layer4' and λ_dense > 0:
                dense_loss = dense_loss + λ_dense * F.mse_loss(
                    F.normalize(h_new.float(), dim=1),
                    F.normalize(h_old.float(), dim=1))

        old_model.to('cpu')
        self._last_ce  = ce.item()
        self._last_sel = (sel_loss + dense_loss).item()
        return ce + sel_loss + dense_loss

    def train_task(self, task_id, train_loader, test_loader, classes_this_task):
        self._eval_mode          = 'fc'
        self.model.detach_hooks  = False

        self.task_id      = task_id
        self.seen_classes += classes_this_task
        self.model.expand_classifier(self.seen_classes)
        self.model.to(self.device)

        self.before_task(task_id, train_loader)
        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer, self.cfg.train.epochs_per_task)

        import torch.nn as nn
        t0   = time.time()
        pbar = tqdm(range(self.cfg.train.epochs_per_task),
                    desc=f"Task {task_id} [SAE-FSGP]", leave=True)

        for epoch in pbar:
            self.model.train()
            epoch_loss = 0.0
            for imgs, labels in train_loader:
                imgs, labels  = imgs.to(self.device), labels.to(self.device)
                self._last_imgs = imgs
                logits = self.model(imgs)
                loss   = self.compute_loss(logits, labels, task_id)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()
                epoch_loss += loss.item()
            if scheduler:
                scheduler.step()
            if (epoch + 1) % self.cfg.log_every == 0:
                self.model.detach_hooks = True
                acc = self._eval_accuracy(test_loader)
                self.model.detach_hooks = False
                pbar.set_postfix(
                    ce=f"{self._last_ce:.3f}", sel=f"{self._last_sel:.3f}",
                    loss=f"{epoch_loss/len(train_loader):.3f}", acc=f"{acc:.2f}")

        self.model.detach_hooks         = True
        self._last_classes_this_task    = classes_this_task
        self.after_task(task_id, train_loader)

        if getattr(self.cfg, 'save_snapshots', False):

            method_tag = getattr(self.cfg.method, 'original_method_name',
                      getattr(self.cfg.method, 'name', 'unknown'))
            run_tag = (f"{method_tag}_{self.cfg.data.dataset}"
                       f"_t{self.cfg.data.num_tasks}_s{self.cfg.seed}")
            snap_dir = os.path.join(self.cfg.save_dir, 'snapshots', run_tag)
            os.makedirs(snap_dir, exist_ok=True)
            snap_path = os.path.join(snap_dir, f'task{task_id}.pt')

            torch.save({
                'task_id':      task_id,
                'state_dict':   self.model.state_dict(),
                'seen_classes': self.seen_classes,
                'class_order':  getattr(self, '_class_order', None),
            }, snap_path)
            self._snapshot_paths.append(snap_path)
            print(f"  [BaseTrainer] Snapshot saved → {snap_path}")

        acc = self._eval_accuracy(test_loader)
        metrics = {'task_id': task_id, 'test_acc': acc, 'time_s': time.time() - t0}
        self._results.append(metrics)
        print(f"  → Task {task_id} | acc (NCM): {acc:.2f}%")
        return metrics
