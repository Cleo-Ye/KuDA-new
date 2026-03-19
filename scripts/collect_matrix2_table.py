#!/usr/bin/env python3
"""
从 sweep_root 下的各 run 目录读取 summary.json，输出类似：
Config | MAE | Corr | Acc-2 | Acc-3 | Acc-5 | F1 | F1_cls

并写入 table.csv（按 MAE 升序）。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional


def _fmt(v: Any) -> str:
    try:
        if v is None:
            return "N/A"
        fv = float(v)
        if fv != fv:  # NaN
            return "N/A"
        return f"{fv:.4f}"
    except Exception:
        return "N/A"


def _get(metrics: Dict[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in metrics:
            return metrics[k]
    return None


def load_row(run_dir: str, run_name: str) -> Optional[Dict[str, Any]]:
    p = os.path.join(run_dir, "summary.json")
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 以 test_at_best_mae 为准（与 best.pth 对齐）
    m = data.get("test_at_best_mae") or {}
    row = {
        "Config": run_name,
        "MAE": _get(m, "MAE"),
        "Corr": _get(m, "Corr"),
        "Acc-2": _get(m, "Mult_acc_2", "Has0_acc_2"),
        "Acc-3": _get(m, "Mult_acc_3"),
        "Acc-5": _get(m, "Mult_acc_5"),
        "F1": _get(m, "F1_score", "Has0_F1_score"),
        # 分类头指标在 KuDA/core/metric.py 里叫 F1_score_cls
        "F1_cls": _get(m, "F1_score_cls"),
    }
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep_root", type=str, required=True)
    args = ap.parse_args()

    sweep_root = os.path.abspath(args.sweep_root)
    if not os.path.isdir(sweep_root):
        raise SystemExit(f"Not a directory: {sweep_root}")

    run_names = sorted(
        [d for d in os.listdir(sweep_root) if os.path.isdir(os.path.join(sweep_root, d))]
    )

    rows: List[Dict[str, Any]] = []
    for name in run_names:
        r = load_row(os.path.join(sweep_root, name), name)
        if r is not None:
            rows.append(r)

    def _mae_key(x: Dict[str, Any]) -> float:
        try:
            v = float(x.get("MAE"))
            return v if v == v else 1e9
        except Exception:
            return 1e9

    rows.sort(key=_mae_key)

    headers = ["Config", "MAE", "Corr", "Acc-2", "Acc-3", "Acc-5", "F1", "F1_cls"]

    # 打印对齐表格（类似你截图风格）
    w = max(14, max((len(r["Config"]) for r in rows), default=14))
    print(f"{'Config':<{w}} {'MAE':>8} {'Corr':>8} {'Acc-2':>8} {'Acc-3':>8} {'Acc-5':>8} {'F1':>8} {'F1_cls':>8}")
    for r in rows:
        print(
            f"{r['Config']:<{w}} "
            f"{_fmt(r['MAE']):>8} {_fmt(r['Corr']):>8} {_fmt(r['Acc-2']):>8} "
            f"{_fmt(r['Acc-3']):>8} {_fmt(r['Acc-5']):>8} {_fmt(r['F1']):>8} {_fmt(r['F1_cls']):>8}"
        )

    # 写 CSV
    csv_path = os.path.join(sweep_root, "table.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=headers)
        wcsv.writeheader()
        for r in rows:
            out = {k: r.get(k) for k in headers}
            wcsv.writerow(out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

