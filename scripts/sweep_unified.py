#!/usr/bin/env python3
"""
统一超参扫描：一、四、七 三个部分的参数（学习率 5e-5→1e-6，batch/seed/path_layers，损失权重，loss_fn/grad_clip）
支持多 GPU 并行、JSON 配置、随机采样控制规模。

默认不保存模型（--no_save_model），仅保留：
  - sweep_dir/run_name/summary.json      # best-MAE、best-Corr 及 test 指标
  - sweep_dir/run_name/valid_curve.csv   # 每 epoch 的 train/valid/test 曲线
  - sweep_dir/run_name/train_progress.log
  - sweep_dir/run_name/SIMS/*.log        # 主训练日志
加 --save_model 则额外保存 best.pth、best_corr.pth。

用法:
  python scripts/sweep_unified.py
  python scripts/sweep_unified.py --config scripts/sweep_unified_config.json
  python scripts/sweep_unified.py --config scripts/sweep_unified_small.json
  python scripts/sweep_unified.py --config ... --max_configs 64
  python scripts/sweep_unified.py --dry_run
  python scripts/sweep_unified.py --summary
"""
import argparse
import itertools
import json
import os
import random
import subprocess
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

# 默认扫参空间（一、四、七）
DEFAULT_PARAMS = {
    # 一：lr, bert_lr, batch_size, seed, path_layers
    'lr': [5e-5, 2e-5, 5e-6, 1e-6],
    'bert_lr': [1e-5, 2e-5],
    'batch_size': [16, 32, 64],
    'seed': [1111, 2024, 42],
    'path_layers': [2, 3],
    # 四：损失权重
    'lambda_aux': [0.05, 0.1],
    'lambda_ortho': [0.0025, 0.005],
    'lambda_rank': [0.05, 0.1],
    'lambda_pid': [0.2, 0.3],
    'lambda_classification': [0.3, 0.35],
    'lambda_task_stage1': [0.05, 0.08],
    'lambda_task_stage2': [0.5, 0.7],
    # 七：损失函数与优化器
    'loss_fn': ['smoothl1', 'l1'],
    'grad_clip': [0.5, 1.0],
}


def parse_args():
    p = argparse.ArgumentParser(description='Sweep params from sections 1, 4, 7')
    p.add_argument('--gpu_ids', type=str, default='0,1,2,3',
                   help='GPU 编号，如 0,1,2,3')
    p.add_argument('--config', type=str, default='',
                   help='JSON 配置：指定要 sweep 的 param -> [values]；未列出的用默认单值')
    p.add_argument('--sweep_dir', type=str, default='./checkpoints/sweep_unified',
                   help='结果根目录，每 run 子目录含 summary.json、valid_curve.csv、train_progress.log、SIMS/*.log')
    p.add_argument('--save_model', action='store_true',
                   help='默认不保存模型；加此参数则保存 best.pth/best_corr.pth')
    p.add_argument('--n_epochs', type=int, default=100)
    p.add_argument('--max_configs', type=int, default=0,
                   help='>0 时对网格随机采样，控制总规模；0=全网格')
    p.add_argument('--seed', type=int, default=42, help='随机采样种子')
    p.add_argument('--dry_run', action='store_true', help='只打印命令不执行')
    p.add_argument('--summary', action='store_true', help='只打印已有实验汇总，不训练')
    return p.parse_args()


def load_config(path):
    if not path or not os.path.isfile(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_grid(config_overrides):
    """根据 config 构建网格。config 中 key 为 param 名，value 为 [values] 列表。
    未在 config 中列出的 param 使用默认单值（不参与 sweep）。"""
    config_overrides = {k: v for k, v in config_overrides.items() if not k.startswith('_')}
    params_to_sweep = {}
    for param, default_vals in DEFAULT_PARAMS.items():
        if param in config_overrides and config_overrides[param]:
            params_to_sweep[param] = config_overrides[param]
        else:
            params_to_sweep[param] = [default_vals[0]]

    keys = list(params_to_sweep.keys())
    values = [params_to_sweep[k] for k in keys]
    grid = []
    for combo in itertools.product(*values):
        cfg = dict(zip(keys, combo))
        cfg['run_name'] = _make_run_name(cfg)
        grid.append(cfg)
    return grid


def _make_run_name(cfg):
    """生成简短 run_name，便于目录命名。"""
    parts = []
    abbrev = {
        'lr': 'lr', 'bert_lr': 'bl', 'batch_size': 'bs', 'seed': 'sd', 'path_layers': 'pl',
        'lambda_aux': 'la', 'lambda_ortho': 'lo', 'lambda_rank': 'lrk', 'lambda_pid': 'lpid',
        'lambda_classification': 'lcls', 'lambda_task_stage1': 'lt1', 'lambda_task_stage2': 'lt2',
        'loss_fn': 'loss', 'grad_clip': 'gc',
    }
    for k in ['lr', 'bert_lr', 'batch_size', 'seed', 'path_layers',
              'lambda_aux', 'lambda_ortho', 'lambda_rank', 'lambda_pid',
              'lambda_classification', 'lambda_task_stage1', 'lambda_task_stage2',
              'loss_fn', 'grad_clip']:
        if k in cfg:
            v = cfg[k]
            if isinstance(v, float):
                s = f"{v:.0e}".replace('-0', '-').replace('.', '')
            else:
                s = str(v)
            parts.append(f"{abbrev.get(k, k[:2])}{s}")
    return "_".join(parts)[:90]


def run_one(cfg, sweep_dir, n_epochs, gpu_id, dry_run, no_save_model=True):
    ckpt_dir = os.path.join(sweep_dir, cfg['run_name'])
    cmd = [
        sys.executable, '-u', 'train.py',
        '--datasetName', 'sims',
        '--model_type', 'pid_dualpath',
        '--use_batch_pid_prior', 'True',
        '--n_epochs', str(n_epochs),
        '--checkpoint_dir', ckpt_dir,
        '--log_path', ckpt_dir + '/',
        '--gpu', str(gpu_id),
    ]
    if no_save_model:
        cmd.extend(['--no_save_model', 'True'])
    for k, v in cfg.items():
        if k == 'run_name':
            continue
        if isinstance(v, bool):
            cmd.extend([f'--{k}', 'True' if v else 'False'])
        else:
            cmd.extend([f'--{k}', str(v)])

    print(f"  GPU {gpu_id}: {cfg['run_name']}")
    if dry_run:
        print(f"    {' '.join(cmd)}")
        return (ckpt_dir, None)
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return (ckpt_dir, proc)


def load_summary(ckpt_dir):
    p = os.path.join(ckpt_dir, 'summary.json')
    if not os.path.isfile(p):
        return None
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_summary_table(sweep_dir, grid):
    print("\n" + "=" * 100)
    print("Sweep Results Summary")
    print("=" * 100)
    headers = ['run_name', 'MAE', 'Corr', 'ep_mae', 'ep_corr']
    print(f"{'run_name':<50} {'MAE':>8} {'Corr':>8} {'ep_mae':>8} {'ep_corr':>8}")
    print("-" * 100)

    for cfg in grid:
        ckpt_dir = os.path.join(sweep_dir, cfg['run_name'])
        s = load_summary(ckpt_dir)
        if s:
            mae = s.get('best_test_mae', s.get('best_valid_mae'))
            corr = s.get('best_test_corr', s.get('best_valid_corr'))
            ep_mae = s.get('best_epoch_mae', '-')
            ep_corr = s.get('best_epoch_corr', '-')
            mae_s = f"{mae:.4f}" if mae is not None else "-"
            corr_s = f"{corr:.4f}" if corr is not None else "-"
        else:
            mae_s = corr_s = "-"
            ep_mae = ep_corr = "-"
        name = cfg['run_name'][:48]
        print(f"{name:<50} {mae_s:>8} {corr_s:>8} {str(ep_mae):>8} {str(ep_corr):>8}")
    print("=" * 100)


def main():
    args = parse_args()
    if args.summary:
        # 只打印汇总：需要从 sweep_dir 下的子目录推断 grid，或读取 grid.json
        sweep_dir = args.sweep_dir
        grid_path = os.path.join(sweep_dir, 'grid.json')
        if os.path.isfile(grid_path):
            with open(grid_path, 'r') as f:
                grid = json.load(f)
        else:
            # 从目录名推断
            grid = []
            if os.path.isdir(sweep_dir):
                for d in sorted(os.listdir(sweep_dir)):
                    p = os.path.join(sweep_dir, d)
                    if os.path.isdir(p) and os.path.isfile(os.path.join(p, 'summary.json')):
                        grid.append({'run_name': d})
        print_summary_table(sweep_dir, grid)
        return 0

    config = load_config(args.config)
    if not config:
        # 无 config 时默认只 sweep lr（5e-5→1e-6）
        config = {'lr': [5e-5, 2e-5, 5e-6, 1e-6]}
    grid = build_grid(config)

    if args.max_configs > 0 and len(grid) > args.max_configs:
        random.seed(args.seed)
        grid = random.sample(grid, args.max_configs)
        print(f"Random sample {args.max_configs} from {len(build_grid(config))} configs (seed={args.seed})")

    os.makedirs(args.sweep_dir, exist_ok=True)
    with open(os.path.join(args.sweep_dir, 'grid.json'), 'w') as f:
        json.dump(grid, f, indent=2)

    gpu_ids = [x.strip() for x in args.gpu_ids.split(',') if x.strip()]
    n_gpus = len(gpu_ids)
    if n_gpus == 0:
        print("Error: need at least one GPU")
        return 1

    print(f"Sweep: {len(grid)} configs, GPUs {gpu_ids}, n_epochs={args.n_epochs}")
    print(f"Results: {args.sweep_dir}/<run_name>/  (summary.json, valid_curve.csv, logs; no model unless --save_model)")
    print()

    results = []
    for i in range(0, len(grid), n_gpus):
        batch = grid[i:i + n_gpus]
        procs = []
        for j, cfg in enumerate(batch):
            gpu_id = gpu_ids[j]
            ckpt_dir, proc = run_one(cfg, args.sweep_dir, args.n_epochs, gpu_id, args.dry_run, no_save_model=not args.save_model)
            results.append({**cfg, 'checkpoint_dir': ckpt_dir})
            if proc:
                procs.append(proc)
        if not args.dry_run and procs:
            for p in procs:
                p.wait()

    if not args.dry_run:
        print_summary_table(args.sweep_dir, grid)

    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
