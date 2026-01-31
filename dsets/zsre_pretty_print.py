import json
from pathlib import Path

# ===== 配置路径 =====
src_path = Path("/rds/general/user/ff422/home/FYP/AlphaEdit/data/zsre_mend_eval.json")
dst_path = Path("/rds/general/user/ff422/home/FYP/AlphaEdit/data/zsre_mend_eval_pretty.json")

# ===== 确保目标目录存在 =====
dst_path.parent.mkdir(parents=True, exist_ok=True)

# ===== 读取并格式化写出 =====
with src_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

with dst_path.open("w", encoding="utf-8") as f:
    json.dump(
        data,
        f,
        indent=2,
        ensure_ascii=False,
        sort_keys=False
    )

print(f"Pretty-printed JSON written to: {dst_path}")
