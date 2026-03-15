#!/usr/bin/env python3
"""
根据 valid_curve.csv 判断过拟合是否明显。

过拟合信号：
1. Train MAE 明显低于 Valid MAE（或 Train Corr 明显高于 Valid Corr）→ 训练集拟合过度
2. Valid MAE 先降后升，最后几 epoch 变差 → 后期过拟合
3. Best 出现在很早期，后面 valid 一路变差 → 早停或加强正则

用法:
  python scripts/check_overfitting.py --checkpoint_dir ./checkpoints/pid_prior_full2
  python scripts/check_overfitting.py --curve ./checkpoints/pid_prior_full2/valid_curve.csv
"""
import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    parser = argparse.ArgumentParser(description='Check overfitting from valid_curve.csv')
    parser.add_argument('--checkpoint_dir', type=str, default='', help='checkpoint 目录，其下 valid_curve.csv')
    parser.add_argument('--curve', type=str, default='', help='直接指定 valid_curve.csv 路径')
    parser.add_argument('--last_n', type=int, default=10, help='看最后 N 个 epoch valid 是否变差')
    args = parser.parse_args()

    if args.curve:
        curve_path = os.path.abspath(args.curve)
    elif args.checkpoint_dir:
        curve_path = os.path.join(os.path.abspath(args.checkpoint_dir), 'valid_curve.csv')
    else:
        print('请指定 --checkpoint_dir 或 --curve')
        return 1

    if not os.path.isfile(curve_path):
        print(f'文件不存在: {curve_path}')
        return 1

    with open(curve_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    if len(lines) < 2:
        print('CSV 行数不足，无法分析')
        return 0

    header = [h.strip().lower() for h in lines[0].split(',')]
    rows = []
    for line in lines[1:]:
        parts = line.split(',')
        if len(parts) < 3:
            continue
        row = {}
        for i, h in enumerate(header):
            if i < len(parts):
                try:
                    row[h] = float(parts[i])
                except ValueError:
                    row[h] = None
        rows.append(row)

    if not rows:
        print('无有效数据行')
        return 0

    has_train = 'train_mae' in header and 'train_corr' in header
    epochs = [r.get('epoch') for r in rows if r.get('epoch') is not None]
    valid_maes = [r.get('valid_mae') for r in rows if r.get('valid_mae') is not None]
    valid_corrs = [r.get('valid_corr') for r in rows if r.get('valid_corr') is not None]
    best_maes = [r.get('best_mae') for r in rows if r.get('best_mae') is not None]

    n = len(valid_maes)
    if n == 0:
        print('无 valid 数据')
        return 0

    last_mae = valid_maes[-1]
    last_corr = valid_corrs[-1]
    best_mae = min(valid_maes)
    best_corr = max(valid_corrs)
    idx_best_mae = valid_maes.index(best_mae)
    idx_best_corr = valid_corrs.index(best_corr)
    best_epoch_mae = int(rows[idx_best_mae].get('epoch', idx_best_mae + 1)) if rows else None
    best_epoch_corr = int(rows[idx_best_corr].get('epoch', idx_best_corr + 1)) if rows else None

    print(f'\n--- 过拟合检查: {curve_path} ---\n')
    print(f'总 epoch 数: {n}')
    print(f'当前 best (CSV): Valid MAE={best_mae:.4f}, Valid Corr={best_corr:.4f}')
    print(f'最后一 epoch:   Valid MAE={last_mae:.4f}, Valid Corr={last_corr:.4f}')

    # 1) Train vs Valid 差距（若有 train 列）
    if has_train:
        train_maes = [r.get('train_mae') for r in rows if r.get('train_mae') is not None]
        train_corrs = [r.get('train_corr') for r in rows if r.get('train_corr') is not None]
        if len(train_maes) == n and len(train_corrs) == n:
            gap_mae = last_mae - train_maes[-1]
            gap_corr = train_corrs[-1] - last_corr
            print(f'\n[1] Train vs Valid（最后一 epoch）:')
            print(f'    Train MAE={train_maes[-1]:.4f}, Valid MAE={last_mae:.4f}  → 差距 {gap_mae:.4f}')
            print(f'    Train Corr={train_corrs[-1]:.4f}, Valid Corr={last_corr:.4f}  → 差距 {gap_corr:.4f}')
            if gap_mae > 0.08:
                print('    → MAE 差距偏大，可能存在过拟合（可考虑加大 dropout/weight_decay）')
            elif gap_mae > 0.04:
                print('    → MAE 有一定差距，可适当加强正则')
            else:
                print('    → Train/Valid 差距不大，过拟合不明显')
    else:
        print('\n[1] CSV 无 train_mae/train_corr 列（需新训练一轮才会写入），跳过 Train vs Valid 对比')

    # 2) Valid 后期是否变差
    last_n = min(args.last_n, n - 1)
    if last_n >= 1:
        last_n_maes = valid_maes[-last_n:]
        last_n_corrs = valid_corrs[-last_n:]
        mae_worse = last_n_maes[-1] > last_n_maes[0]
        corr_worse = last_n_corrs[-1] < last_n_corrs[0]
        print(f'\n[2] 最后 {last_n} 个 epoch:')
        print(f'    Valid MAE:  {last_n_maes[0]:.4f} → {last_n_maes[-1]:.4f}  {"↑ 变差" if mae_worse else "↓ 持平或更好"}')
        print(f'    Valid Corr: {last_n_corrs[0]:.4f} → {last_n_corrs[-1]:.4f}  {"↓ 变差" if corr_worse else "↑ 持平或更好"}')
        if mae_worse or corr_worse:
            print('    → 后期 valid 变差，建议早停或减少 epoch / 加强正则')

    # 3) Best 出现位置
    print(f'\n[3] Best 出现位置:')
    print(f'    Best MAE  at epoch ~{best_epoch_mae},  Best Corr at epoch ~{best_epoch_corr}')
    if best_epoch_mae is not None and n > 10 and best_epoch_mae < n * 0.4:
        print(f'    → Best 出现在前 40% epoch，后面长期无提升，可考虑减少 n_epochs 或加强正则')

    print('\n' + '=' * 50 + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
