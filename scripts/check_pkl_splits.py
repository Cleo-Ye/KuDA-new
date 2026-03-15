#!/usr/bin/env python3
"""
检查各数据集 pkl 的 train/valid/test 样本数是否与论文 Table 1 一致。

论文 Table 1 参考：
  CH-SIMS:    Train 1368,  Valid 456,  Test 457,  Total 2281
  CH-SIMSv2:  Train 2722,  Valid 647,  Test 1034, Total 4403
  MOSI:       Train 1284,  Valid 229,  Test 686,  Total 2199
  MOSEI:      Train 16326, Valid 1871, Test 4659, Total 22856

用法（项目根目录）:
  python scripts/check_pkl_splits.py
  python scripts/check_pkl_splits.py --dataset sims
"""
import os
import sys
import pickle
import argparse

# 与 experiment_configs 中一致的数据路径（仅用标准库，不 import 项目模块）
DEFAULT_PATHS = {
    'sims':    '/18T/yechenlu/MSA_datasets/SIMS/Processed/unaligned_39.pkl',
    'simsv2':  '/18T/yechenlu/MSA_datasets/SIMS-v2/ch-sims2s/unaligned.pkl',
    'mosi':    '/18T/yechenlu/MSA_datasets/MOSI/Processed/unaligned_50.pkl',
    'mosei':   '/18T/yechenlu/MSA_datasets/MOSEI/Processed/unaligned_50.pkl',
}

# 论文 Table 1 的期望样本数 (train, valid, test)
PAPER_TABLE1 = {
    'sims':    (1368,  456,  457),
    'simsv2':  (2722,  647,  1034),
    'mosi':    (1284,  229,  686),
    'mosei':   (16326, 1871, 4659),
}


def get_count(data, split, fallback_keys=('vision', 'text_bert', 'audio', 'text')):
    """从 pkl 的 data[split] 中取样本数，优先用 vision 的 shape[0]。"""
    if split not in data:
        return None
    d = data[split]
    for key in fallback_keys:
        if key in d:
            arr = d[key]
            if hasattr(arr, 'shape'):
                return int(arr.shape[0])
            if hasattr(arr, '__len__'):
                return len(arr)
    return None


def main():
    parser = argparse.ArgumentParser(description='Check pkl train/valid/test counts')
    parser.add_argument('--dataset', type=str, default='', help='只检查指定数据集，如 sims；空则检查全部')
    parser.add_argument('--path', type=str, default='', help='直接指定 pkl 路径，覆盖 --dataset')
    args = parser.parse_args()

    if args.path:
        paths_to_check = [('custom', args.path)]
    elif args.dataset:
        path = DEFAULT_PATHS.get(args.dataset.lower())
        if not path:
            print(f'未知数据集: {args.dataset}，可选: {list(DEFAULT_PATHS.keys())}')
            return 1
        paths_to_check = [(args.dataset, path)]
    else:
        paths_to_check = list(DEFAULT_PATHS.items())

    for name, path in paths_to_check:
        if not os.path.isfile(path):
            print(f'[{name}] 文件不存在: {path}')
            continue

        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            print(f'[{name}] 读取失败: {e}')
            continue

        n_train = get_count(data, 'train')
        n_valid = get_count(data, 'valid')
        n_test = get_count(data, 'test')
        total = (n_train or 0) + (n_valid or 0) + (n_test or 0)

        expected = PAPER_TABLE1.get(name.lower()) if name != 'custom' else None
        match = ''
        if expected and n_train is not None and n_valid is not None and n_test is not None:
            exp_train, exp_valid, exp_test = expected
            if (n_train, n_valid, n_test) == (exp_train, exp_valid, exp_test):
                match = '  ✓ 与论文 Table 1 一致'
            else:
                match = f'  ✗ 与论文 Table 1 不一致（论文: Train {exp_train}, Valid {exp_valid}, Test {exp_test}）'

        print(f'\n[{name}] {path}')
        print(f'  Train: {n_train},  Valid: {n_valid},  Test: {n_test},  Total: {total}{match}')

    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
