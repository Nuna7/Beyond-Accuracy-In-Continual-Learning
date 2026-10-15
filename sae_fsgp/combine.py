from pathlib import Path

def combine_directory(dir_name, output_file, order):
    dir_path = Path(dir_name)
    files = [dir_path / name for name in order if (dir_path / name).exists()]

    with open(output_file, 'w', encoding='utf-8') as out:
        out.write(f"# COMBINED FILE FOR: {dir_name}/\n")
        out.write(f"# This file contains {len(files)} scripts merged into one\n\n")

        for file_path in files:
            file_name = f"{dir_name}/{file_path.name}"
            
            # START LABEL
            out.write(f"\n{'#'*100}\n")
            out.write(f"# START OF CODE SCRIPT: {file_name}\n")
            out.write(f"{'#'*100}\n\n")
            
            content = file_path.read_text(encoding='utf-8')
            out.write(content.strip())
            
            # END LABEL
            out.write(f"\n\n{'#'*100}\n")
            out.write(f"# END OF CODE SCRIPT: {file_name}\n")
            out.write(f"{'#'*100}\n")

    print(f"Created {output_file} from {len(files)} files")

# 1. For analysis directory
combine_directory(
    "analysis",
    "analysis_all.py",
    order=[
        "__init__.py",
        "feature_fate.py",
        "feature_fate_runner.py",
        "forgetting_trajectory.py",
        "interclass_separability.py",
        "temporal_fate_plot.py",
        "visual_confirmation.py",
        "posthoc_sae.py",
        "method_compare.py",
        "run_ablation.py",
        "run_analysis.py",
    ]
)

# 2. For methods directory
combine_directory(
    "methods",
    "methods_all.py",
    order=[
        "__init__.py",
        "base_trainer.py",
        "ewc.py",
        "lwf.py",
        "fecam.py",
        "sae_fsgp.py",
    ]
)