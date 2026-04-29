#!/usr/bin/env python3

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
CLEAN_PATTERNS = ("*.xlsx", "*.csv")


def clear_xlsx_files(output_dir: Path = OUTPUT_DIR) -> int:
    if not output_dir.exists():
        print(f"输出目录不存在: {output_dir}")
        return 0

    deleted_count = 0
    for pattern in CLEAN_PATTERNS:
        for file_path in output_dir.glob(pattern):
            if file_path.is_file():
                file_path.unlink()
                deleted_count += 1
                print(f"已删除: {file_path}")

    print(f"完成，共删除 {deleted_count} 个 xlsx/csv 文件。")
    return deleted_count


if __name__ == "__main__":
    clear_xlsx_files()
