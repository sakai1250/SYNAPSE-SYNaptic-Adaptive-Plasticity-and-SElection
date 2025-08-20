import os
import shutil
from pathlib import Path

projects = ["NICE", "SYNAPSE"]

for project in projects:
    base = f"/media/blackie/8000GB_blackie/SYNAPSE-SYNaptic-Adaptive-Plasticity-and-SElection/{project}/Logs/"
    roots = ["CIFAR100", "TinyImageNet", "Other"]

    for root in roots:
        # フォルダのルート
        root = Path(base) / root
        trash = root / "trash"
        trash.mkdir(exist_ok=True)

        # 直下の実験フォルダだけ対象にする
        for exp_dir in root.iterdir():
            if exp_dir.is_dir() and exp_dir.name.startswith("2025"):
                end_file = exp_dir / "End_of_Sequence.csv"
                if not end_file.exists():
                    print(f"移動対象: {exp_dir}")
                    shutil.move(str(exp_dir), trash / exp_dir.name)
                else:
                    # print(f"保持: {exp_dir}")
                    continue
