"""下载本项目运行所需的外部资源（tokenizer 与语料），全部落在被 gitignore 的目录里。

用法：
    uv run python scripts/download_assets.py                # 只要 tokenizer
    uv run python scripts/download_assets.py --with-data    # 可选的大型 SFT 语料

资源来源与授权：
    tokenizer 来自 jingyaogong/minimind；数据来自其数据集镜像，使用前请核对来源许可。
    本仓库不复制、不重新分发这些文件，只在使用者本机按需下载。
"""

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path.cwd()
ASSETS = ROOT / "assets"
DATA = ROOT / "data"

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
)
TOKENIZER_BASE = "https://raw.githubusercontent.com/jingyaogong/minimind/master/model"

DATA_URL = "https://www.modelscope.cn/datasets/lulich/minimind_dataset/resolve/master/sft_data_single.zip"


def fetch(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached  {dest}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  fetch   {url}\n     -> {dest}")
    try:
        with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED  {exc}", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False
    tmp.rename(dest)
    print(f"  done    {dest.stat().st_size / 1e6:.1f} MB")
    return True


def extract_csv(archive: Path, destination: Path):
    """只读取目标 CSV，不使用归档里的路径创建文件。"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    try:
        with zipfile.ZipFile(archive) as z:
            matches = [info for info in z.infolist() if Path(info.filename).name == destination.name]
            if len(matches) != 1:
                raise ValueError(f"归档必须包含唯一的 {destination.name}")
            with z.open(matches[0]) as source, open(tmp, "wb") as target:
                shutil.copyfileobj(source, target)
        tmp.replace(destination)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--with-data", action="store_true", help="下载并解压大型 MiniMind SFT 语料")
    ap.add_argument("--assets-dir", type=Path, default=ASSETS)
    ap.add_argument("--data-dir", type=Path, default=DATA)
    args = ap.parse_args()

    ok = all([fetch(f"{TOKENIZER_BASE}/{f}", args.assets_dir / f) for f in TOKENIZER_FILES])
    if not ok:
        print("\ntokenizer 下载失败，无法继续。", file=sys.stderr)
        return 1

    if args.with_data:
        zip_path = args.data_dir / "sft_data_single.zip"
        csv_path = args.data_dir / "sft_data_single.csv"
        if not csv_path.exists():
            if not fetch(DATA_URL, zip_path):
                return 1
            print("  unzip   sft_data_single.csv")
            try:
                extract_csv(zip_path, csv_path)
            except (OSError, ValueError, zipfile.BadZipFile) as exc:
                print(f"解压失败: {exc}", file=sys.stderr)
                return 1

    print("\nassets ->", args.assets_dir, "\ndata   ->", args.data_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
