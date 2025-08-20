import os
import shutil
from pathlib import Path

projects = ["NICE", "SYNAPSE"]

for project in projects:
    roots = ["CIFAR100", "TinyImageNet", "Other"]
    for root in roots:
        root = Path(f"{project}/Logs/{root}")

        # 新しいカテゴリ分けフォルダを作成
        def get_number_of_tasks(exp_dir: Path) -> int:
            args_file = exp_dir / "args.txt"
            if args_file.exists():
                with open(args_file) as f:
                    for line in f:
                        if "number_of_tasks" in line:
                            # "number_of_tasks: 5" のような行を想定
                            return int(line.split(":")[1].strip())
            return None

        for exp_dir in root.iterdir():
            if exp_dir.is_dir() and exp_dir.name.startswith("2025"):
                number_of_tasks = get_number_of_tasks(exp_dir)
                if number_of_tasks is not None:
                    target_dir = root / f"number_of_tasks_{number_of_tasks}"
                    target_dir.mkdir(exist_ok=True)
                    print(f"移動: {exp_dir} -> {target_dir}")
                    shutil.move(str(exp_dir), str(target_dir / exp_dir.name))
                else:
                    print(f"args.txt に number_of_tasks が見つからない: {exp_dir}")
