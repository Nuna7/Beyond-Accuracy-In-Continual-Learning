import os
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

def cifar100_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(
                policy=transforms.AutoAugmentPolicy.CIFAR10
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                  (0.2675, 0.2565, 0.2761)),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                              (0.2675, 0.2565, 0.2761)),
    ])

def cifar100_clean_transforms():
    return cifar100_transforms(train=False)

def tinyimagenet_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975),
                                  (0.2770, 0.2691, 0.2821)),
        ])
    return transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975),
                              (0.2770, 0.2691, 0.2821)),
    ])

def tinyimagenet_clean_transforms():
    return tinyimagenet_transforms(train=False)

def download_tinyimagenet(root: str):
    import urllib.request, zipfile, shutil
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(root, "tiny-imagenet-200.zip")
    extract_path = os.path.join(root, "tiny-imagenet-200")
    val_dir = os.path.join(extract_path, "val")

    if not os.path.exists(extract_path):
        print("Downloading Tiny-ImageNet …")
        os.makedirs(root, exist_ok=True)
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(root)

    images_dir = os.path.join(val_dir, "images")
    ann_file = os.path.join(val_dir, "val_annotations.txt")

    if os.path.exists(images_dir) and os.path.exists(ann_file):
        print("Restructuring Tiny-ImageNet val set...")
        with open(ann_file) as f:
            lines = f.readlines()

        img2cls = {}
        for line in lines:
            parts = line.strip().split('\t')
            img2cls[parts[0]] = parts[1]

        for img, cls in img2cls.items():
            cls_dir = os.path.join(val_dir, cls)
            os.makedirs(cls_dir, exist_ok=True)
            src = os.path.join(images_dir, img)
            dst = os.path.join(cls_dir, img)
            if os.path.exists(src):
                shutil.move(src, dst)

        shutil.rmtree(images_dir)
        os.remove(ann_file)
        print("Restructure complete: val/ now has 200 class folders")

    return extract_path

class ClassIncrementalScenario:
    def __init__(
        self,
        dataset: str,
        num_tasks: int,
        data_root: str,
        seed: int = 42,
        batch_size: int = 128,
        num_workers: int = 4,
    ):
        self.dataset = dataset
        self.num_tasks = num_tasks
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers

        if dataset == "cifar100":
            self.num_classes = 100
            train_tf = cifar100_transforms(train=True)
            test_tf  = cifar100_transforms(train=False)
            self._full_train = datasets.CIFAR100(
                data_root, train=True, download=True, transform=train_tf)
            self._full_train_clean = datasets.CIFAR100(
                data_root, train=True, download=False,
                transform=cifar100_clean_transforms())
            self._full_test = datasets.CIFAR100(
                data_root, train=False, download=True, transform=test_tf)
            raw_labels_train = np.array(self._full_train.targets)
            raw_labels_test  = np.array(self._full_test.targets)

        elif dataset == "tinyimagenet":
            self.num_classes = 200
            tiny_root = download_tinyimagenet(data_root)
            train_tf = tinyimagenet_transforms(train=True)
            test_tf  = tinyimagenet_transforms(train=False)

            self._full_train = ImageFolder(
                os.path.join(tiny_root, "train"), transform=train_tf)
            self._full_train_clean = ImageFolder(
                os.path.join(tiny_root, "train"),
                transform=tinyimagenet_clean_transforms())
            self._full_test = ImageFolder(
                os.path.join(tiny_root, "val"), transform=test_tf)

            raw_labels_train = np.array(
                [s[1] for s in self._full_train.samples])
            raw_labels_test  = np.array(
                [s[1] for s in self._full_test.samples])
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        assert self.num_classes % num_tasks == 0,            f"{dataset} {self.num_classes} classes not divisible by {num_tasks} tasks"
        self.classes_per_task = self.num_classes // num_tasks

        rng = np.random.default_rng(seed)
        self.class_order = rng.permutation(self.num_classes).tolist()

        self.task_classes: List[List[int]] = [
            self.class_order[t * self.classes_per_task:
                             (t + 1) * self.classes_per_task]
            for t in range(num_tasks)
        ]

        self.class_to_idx = {c: i for i, c in enumerate(self.class_order)}

        self._train_indices: List[np.ndarray] = []
        self._test_indices:  List[np.ndarray] = []
        for t in range(num_tasks):
            task_cls = self.task_classes[t]
            self._train_indices.append(
                np.where(np.isin(raw_labels_train, task_cls))[0])
            self._test_indices.append(
                np.where(np.isin(raw_labels_test, task_cls))[0])

        print(f"[Scenario] {dataset} | {num_tasks} tasks × "
              f"{self.classes_per_task} classes | "
              f"class order seed={seed}")

    def _remap(self, dataset, indices: np.ndarray):
        subset = Subset(dataset, indices)
        return _RemappedSubset(subset, dataset, self.class_to_idx)

    def get_task_loaders(
        self, task_id: int
    ) -> Tuple[DataLoader, DataLoader]:
        train_ds = self._remap(self._full_train, self._train_indices[task_id])
        test_ds  = self._remap(self._full_test,  self._test_indices[task_id])
        return (
            DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,
                       num_workers=self.num_workers, pin_memory=True),
            DataLoader(test_ds,  batch_size=self.batch_size, shuffle=False,
                       num_workers=self.num_workers, pin_memory=True),
        )

    def get_cumulative_test_loader(self, up_to_task: int) -> DataLoader:
        all_idx = np.concatenate(self._test_indices[:up_to_task + 1])
        ds = self._remap(self._full_test, all_idx)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def get_activation_loader(
        self, task_id: int, max_samples: Optional[int] = 5000
    ) -> DataLoader:
        idx = self._train_indices[task_id]
        if max_samples and len(idx) > max_samples:
            rng = np.random.default_rng(task_id)
            idx = rng.choice(idx, max_samples, replace=False)
        ds = self._remap(self._full_train, idx)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def get_clean_activation_loader(
        self, task_id: int, max_samples: Optional[int] = 5000
    ) -> DataLoader:
        idx = self._train_indices[task_id]
        if max_samples and len(idx) > max_samples:
            rng = np.random.default_rng(task_id)
            idx = rng.choice(idx, max_samples, replace=False)
        ds = self._remap(self._full_train_clean, idx)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def get_split_activation_loaders(
        self,
        task_id: int,
        fit_frac: float = 0.8,
        max_samples: Optional[int] = None,
        split_seed: int = 0,
    ) -> Tuple[DataLoader, DataLoader]:
        idx = self._train_indices[task_id].copy()
        rng = np.random.default_rng(task_id * 1_000_003 + split_seed)
        rng.shuffle(idx)
        if max_samples and len(idx) > max_samples:
            idx = idx[:max_samples]
        n_fit = int(round(len(idx) * fit_frac))
        n_fit = max(1, min(n_fit, len(idx) - 1))
        fit_idx, eval_idx = idx[:n_fit], idx[n_fit:]

        fit_ds  = self._remap(self._full_train_clean, fit_idx)
        eval_ds = self._remap(self._full_train_clean, eval_idx)
        return (
            DataLoader(fit_ds, batch_size=self.batch_size, shuffle=False,
                       num_workers=self.num_workers, pin_memory=True),
            DataLoader(eval_ds, batch_size=self.batch_size, shuffle=False,
                       num_workers=self.num_workers, pin_memory=True),
        )

    def task_class_range(self, task_id: int) -> Tuple[int, int]:
        """Global label range [start, end) for a task."""
        start = task_id * self.classes_per_task
        return start, start + self.classes_per_task

class _RemappedSubset(torch.utils.data.Dataset):
    def __init__(self, subset, base_dataset, class_to_idx):
        self.subset = subset
        self.base = base_dataset
        self.c2i = class_to_idx

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return img, self.c2i[label]
