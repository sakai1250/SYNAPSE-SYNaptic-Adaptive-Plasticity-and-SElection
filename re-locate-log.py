# -*- coding: utf-8 -*-
import os
import re
import shutil
from pathlib import Path

# ===== 設定 =====
BASE_DIR = Path("/media/blackie/8000GB_blackie/SYNAPSE-SYNaptic-Adaptive-Plasticity-and-SElection/NICE/Logs")
KEEP_LATEST = 1   # 最新ログフォルダをいくつ残すか
DRY_RUN = False   # True で移動せず確認だけ

BIN_TINY  = "TinyImageNet"
BIN_CIFAR = "CIFAR100"
BIN_OTHER = "Other"

# ===== 実装 =====
TS_RE = re.compile(r"^\d{8}_\d{6}$")  # 例: 20250806_175044

def ensure_dir(p: Path):
    if not p.exists():
        print(f"[make] {p}")
        if not DRY_RUN:
            p.mkdir(parents=True, exist_ok=True)

def unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    i = 1
    while True:
        cand = dest.with_name(dest.name + f"_{i}")
        if not cand.exists():
            return cand
        i += 1

def dataset_bin(name_lower: str) -> str:
    if "tiny" in name_lower:
        return BIN_TINY
    if "cifar" in name_lower:
        return BIN_CIFAR
    return BIN_OTHER

def move(src: Path, dst: Path):
    dst = unique_dest(dst)
    print(f"[move] {src} -> {dst}")
    if not DRY_RUN:
        shutil.move(str(src), str(dst))

def main():
    entries = [d for d in BASE_DIR.iterdir() if d.is_dir()]
    log_dirs = [d for d in entries if TS_RE.match(d.name)]
    if not log_dirs:
        print("ログフォルダが見つかりません")
        print("iterdir:", entries)
        return

    # 新しい順に並べて KEEP_LATEST を残す
    log_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    keep = set(log_dirs[:KEEP_LATEST])
    print(f"[keep] 残すログ: {[d.name for d in keep]}")

    for bin_name in (BIN_TINY, BIN_CIFAR, BIN_OTHER):
        ensure_dir(BASE_DIR / bin_name)

    for ld in log_dirs[KEEP_LATEST:]:
        print(f"\n[scan] {ld.name}")

        # 実験フォルダを移動
        for sub in [p for p in ld.iterdir() if p.is_dir()]:
            bin_name = dataset_bin(sub.name.lower())
            # 日時 + 実験名 で保存
            new_name = f"{ld.name}_{sub.name}"
            dst = BASE_DIR / bin_name / new_name
            move(sub, dst)

        # 残ったファイル（args.txt など）
        leftovers = [p for p in ld.iterdir() if p.is_file()]
        if leftovers:
            misc_root = BASE_DIR / BIN_OTHER / f"{ld.name}_misc"
            ensure_dir(misc_root)
            for f in leftovers:
                move(f, misc_root / f.name)

        # 空になったログフォルダを削除
        try:
            if not any(ld.iterdir()):
                print(f"[rmdir] {ld}")
                if not DRY_RUN:
                    ld.rmdir()
        except FileNotFoundError:
            pass

if __name__ == "__main__":
    main()
