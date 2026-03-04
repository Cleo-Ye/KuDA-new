#!/usr/bin/env python3
"""
从 logs/full_multi/ 下各配置的 train_*.log 中解析 Best-MAE 的 Test 结果，
生成六配置的 Best-MAE 表格（MAE, Corr, Acc-2, Acc-3, Acc-5, F1）。
"""
import os
import re

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs", "full_multi")
OUT_FILE = os.path.join(LOG_DIR, "best_mae_table.txt")

# 日志文件名 -> 表格中的配置名
CONFIG_NAMES = {
    "train_baseline.log": "Baseline",
    "train_+IEC_r05.log": "+IEC (r=0.5)",
    "train_+ICR_only.log": "+ICR only",
    "train_IEC+ICR_full.log": "IEC+ICR full",
    "train_IEC+ICR_KL.log": "IEC+ICR KL",
    "train_IEC+ICR_r03.log": "IEC+ICR r=0.3",
}


def parse_best_mae_block(content: str):
    """在 content 中找 'Best-MAE Test Results' 后、'Best-Corr' 前的 MAE/Corr/Acc-2/Acc-3/Acc-5/F1 行。"""
    pattern = r"Best-MAE Test Results.*?(?=Best-Corr Test Results|$)"
    match = re.search(pattern, content, re.DOTALL)
    if not match:
        return None
    block = match.group(0)
    out = {}
    for key in ["MAE", "Corr", "Acc-2", "Acc-3", "Acc-5", "F1"]:
        m = re.search(rf"{re.escape(key)}:\s*([\d.]+)", block)
        out[key] = m.group(1) if m else "N/A"
    return out


def main():
    rows = []
    for log_name, config_name in CONFIG_NAMES.items():
        path = os.path.join(LOG_DIR, log_name)
        if not os.path.isfile(path):
            rows.append((config_name, None))
            continue
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        metrics = parse_best_mae_block(content)
        rows.append((config_name, metrics))

    # 打印并写入文件
    header = ("Config", "MAE", "Corr", "Acc-2", "Acc-3", "Acc-5", "F1")
    col_widths = (22, 8, 8, 8, 8, 8, 8)
    sep = "-" * sum(col_widths)

    lines = [
        "Best-MAE Test Results (CH-SIMS, full_multi logs)",
        sep,
        "".join(h.ljust(w) for h, w in zip(header, col_widths)),
        sep,
    ]
    for config_name, metrics in rows:
        if metrics is None:
            line = config_name.ljust(col_widths[0]) + " (no log or parse failed)"
        else:
            cells = [config_name] + [metrics[k] for k in ["MAE", "Corr", "Acc-2", "Acc-3", "Acc-5", "F1"]]
            line = "".join(str(c).ljust(w) for c, w in zip(cells, col_widths))
        lines.append(line)
    lines.append(sep)

    text = "\n".join(lines)
    print(text)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\nTable saved to: {OUT_FILE}")


if __name__ == "__main__":
    main()
