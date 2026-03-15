#!/usr/bin/env python3
"""
超参扫描脚本：按不同 batch_size 及可选配置多次运行训练，结束后输出实验对比表。
支持多 GPU 并行；支持里程碑模式：到 60/80/100 各输出一份表并继续跑到下一里程碑。
用法:
  python scripts/sweep_hyperparams.py
  python scripts/sweep_hyperparams.py --n_epochs 80 --batch_sizes 16 32 48 64 --lrs 3e-5 5e-5 1e-4
  python scripts/sweep_hyperparams.py --milestones 60 80 100 --batch_sizes 16 32 48 64 --lrs 3e-5 5e-5 1e-4
  python scripts/sweep_hyperparams.py --config sweep_config.json
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

# 项目根目录
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)


def parse_args():
    p = argparse.ArgumentParser(description='Sweep hyperparameters and output comparison table')
    p.add_argument('--gpu_ids', type=str, default='0,1,2,3,4,5,6,7',
                   help='逗号分隔的 GPU 编号，用于并行多组实验，如 0,1,2,3,4,5,6,7')
    p.add_argument('--datasetName', type=str, default='sims', help='sims, simsv2, mosi, mosei')
    p.add_argument('--model_type', type=str, default='pid_dualpath', help='pid_dualpath or kmsa')
    p.add_argument('--n_epochs', type=int, default=80, help='Epochs per run (sweep 可设小一些加快)')
    p.add_argument('--batch_sizes', type=int, nargs='+', default=[16, 32, 48, 64],
                   help='List of batch sizes to try')
    p.add_argument('--lrs', type=float, nargs='+', default=None,
                   help='Optional list of learning rates, e.g. 3e-5 5e-5 1e-4')
    p.add_argument('--milestones', type=int, nargs='+', default=None,
                   help='里程碑 epoch，如 60 80 100：每到一个里程碑输出对比表并继续跑到下一里程碑；与 --n_epochs 二选一')
    p.add_argument('--sweep_dir', type=str, default='./checkpoints/sweep',
                   help='Parent dir for each run: sweep_dir/run_xxx/')
    p.add_argument('--config', type=str, default='',
                   help='Optional JSON file: {"batch_sizes": [16,32], "lrs": [5e-5], "n_epochs": 20, ...}')
    p.add_argument('--dry_run', action='store_true', help='Only print commands, do not run')
    return p.parse_args()


def build_grid(args, n_epochs_override=None):
    """生成 (batch_size, lr, run_name) 等配置列表。n_epochs_override 用于里程碑某阶段的 epoch。"""
    batch_sizes = args.batch_sizes
    lrs = args.lrs if args.lrs is not None else [5e-5]
    n_epochs = n_epochs_override if n_epochs_override is not None else args.n_epochs
    grid = []
    for bs in batch_sizes:
        for lr in lrs:
            name = f"bs{bs}_lr{lr:.0e}".replace('-0', '-')
            grid.append({
                'batch_size': bs,
                'lr': lr,
                'run_name': name,
                'n_epochs': n_epochs,
            })
    return grid


def run_one(cfg, args, run_id, gpu_id, resume_path=None):
    """单次运行，指定 GPU。resume_path 不为 None 时从该 checkpoint 续训到 cfg['n_epochs']。"""
    checkpoint_dir = os.path.join(args.sweep_dir, cfg['run_name'])
    cmd = [
        sys.executable, 'train.py',
        '--datasetName', args.datasetName,
        '--model_type', args.model_type,
        '--batch_size', str(cfg['batch_size']),
        '--lr', str(cfg['lr']),
        '--n_epochs', str(cfg['n_epochs']),
        '--checkpoint_dir', checkpoint_dir,
        '--gpu', str(gpu_id),
    ]
    if resume_path and os.path.isfile(resume_path):
        cmd.extend(['--resume', resume_path])
    print(f"[{run_id}] GPU {gpu_id}: {cfg['run_name']}  n_epochs={cfg['n_epochs']}  {'resume ' + resume_path if resume_path else ''}")
    if args.dry_run:
        return (checkpoint_dir, None)
    # 不用 stderr=PIPE：子进程大量写 stderr（tqdm/日志）会填满管道，父进程不读会导致死锁
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return (checkpoint_dir, proc)


def load_summary(checkpoint_dir):
    path = os.path.join(checkpoint_dir, 'summary.json')
    if not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    args = parse_args()
    if args.config and os.path.isfile(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            c = json.load(f)
        args.batch_sizes = c.get('batch_sizes', args.batch_sizes)
        args.lrs = c.get('lrs', args.lrs)
        args.n_epochs = c.get('n_epochs', args.n_epochs)
        if 'milestones' in c:
            args.milestones = c['milestones']
        if 'model_type' in c:
            args.model_type = c['model_type']
        if 'datasetName' in c:
            args.datasetName = c['datasetName']

    milestones = args.milestones
    if milestones is not None:
        milestones = sorted(milestones)
        print("里程碑模式: 到 {} 各输出对比表并续训到下一阶段".format(milestones))
    else:
        milestones = []

    grid = build_grid(args, n_epochs_override=milestones[0] if milestones else None)
    os.makedirs(args.sweep_dir, exist_ok=True)
    gpu_ids = [x.strip() for x in args.gpu_ids.split(',') if x.strip()]
    n_gpus = len(gpu_ids)
    if n_gpus == 0:
        print("错误: --gpu_ids 至少需要一个 GPU 编号")
        return 1
    print("使用 GPU: {}".format(gpu_ids))
    print("共 {} 组配置，每批最多 {} 组并行\n".format(len(grid), n_gpus))

    all_tables = []  # 每个里程碑一份 results -> (epoch, results)

    for phase, target_epoch in enumerate(milestones):
        # 本阶段：跑到 target_epoch；若 phase>0 则从上一阶段 best.pth resume
        grid_phase = build_grid(args, n_epochs_override=target_epoch)
        print("\n" + "#" * 80)
        print("# 阶段 {}: 训练至 {} epochs{}".format(
            phase + 1, target_epoch,
            " (从 checkpoint 续训)" if phase > 0 else " (从零开始)"))
        print("#" * 80)

        results = []
        if args.dry_run:
            for i, cfg in enumerate(grid_phase):
                gpu_id = gpu_ids[i % n_gpus]
                resume = os.path.join(args.sweep_dir, cfg['run_name'], 'best.pth') if phase > 0 else None
                ckpt_dir, _ = run_one(cfg, args, i + 1, gpu_id, resume_path=resume)
                results.append({**cfg, 'checkpoint_dir': ckpt_dir, 'gpu': gpu_id, 'status': 'dry_run'})
        else:
            for batch_start in range(0, len(grid_phase), n_gpus):
                batch = grid_phase[batch_start:batch_start + n_gpus]
                procs = []
                for j, cfg in enumerate(batch):
                    gpu_id = gpu_ids[j]
                    run_id = batch_start + j + 1
                    resume_path = None
                    if phase > 0:
                        resume_path = os.path.join(args.sweep_dir, cfg['run_name'], 'best.pth')
                    ckpt_dir, proc = run_one(cfg, args, run_id, gpu_id, resume_path=resume_path)
                    procs.append((cfg, ckpt_dir, proc))
                for cfg, ckpt_dir, proc in procs:
                    if proc is not None:
                        ret = proc.wait()
                        if ret != 0:
                            print("  [失败] {} returncode={}".format(cfg['run_name'], ret))
                    else:
                        ret = 0
                    summary = load_summary(ckpt_dir) if ret == 0 else None
                    results.append(_row_from_summary(cfg, ckpt_dir, summary))

        all_tables.append((target_epoch, results))
        _print_and_save_table(results, args.sweep_dir, target_epoch)

    if not milestones:
        # 非里程碑模式：单次跑到 n_epochs
        grid = build_grid(args)
        results = []
        if args.dry_run:
            for i, cfg in enumerate(grid):
                gpu_id = gpu_ids[i % n_gpus]
                ckpt_dir, _ = run_one(cfg, args, i + 1, gpu_id)
                results.append({**cfg, 'checkpoint_dir': ckpt_dir, 'gpu': gpu_id, 'status': 'dry_run'})
        else:
            for batch_start in range(0, len(grid), n_gpus):
                batch = grid[batch_start:batch_start + n_gpus]
                procs = []
                for j, cfg in enumerate(batch):
                    gpu_id = gpu_ids[j]
                    run_id = batch_start + j + 1
                    ckpt_dir, proc = run_one(cfg, args, run_id, gpu_id)
                    procs.append((cfg, ckpt_dir, proc))
                for cfg, ckpt_dir, proc in procs:
                    if proc is not None:
                        ret = proc.wait()
                        if ret != 0:
                            print("  [失败] {} returncode={}".format(cfg['run_name'], ret))
                    else:
                        ret = 0
                    summary = load_summary(ckpt_dir) if ret == 0 else None
                    row = _row_from_summary(cfg, ckpt_dir, summary)
                    results.append(row)
        _print_and_save_table(results, args.sweep_dir, None)
    else:
        # 里程碑模式：最后再打一次总结（以最后一阶段为准）
        print("\n" + "=" * 100)
        print("里程碑全部完成，最终阶段 ({} epochs) 结果见上表及 CSV".format(milestones[-1]))
        print("=" * 100)
    return 0


def _row_from_summary(cfg, ckpt_dir, summary):
    row = {
        'run_name': cfg['run_name'],
        'batch_size': cfg['batch_size'],
        'lr': cfg['lr'],
        'n_epochs': cfg['n_epochs'],
        'checkpoint_dir': ckpt_dir,
        'ok': summary is not None,
    }
    if summary:
        row['best_valid_mae'] = summary.get('best_valid_mae')
        row['best_valid_corr'] = summary.get('best_valid_corr')
        row['best_epoch_mae'] = summary.get('best_epoch_mae')
        row['best_epoch_corr'] = summary.get('best_epoch_corr')
        t_mae = summary.get('test_at_best_mae') or {}
        t_corr = summary.get('test_at_best_corr') or {}
        row['test_MAE'] = t_mae.get('MAE')
        row['test_Corr'] = t_mae.get('Corr')
        row['test_MAE_corr_ckpt'] = t_corr.get('MAE')
        row['test_Corr_corr_ckpt'] = t_corr.get('Corr')
    else:
        row['best_valid_mae'] = row['best_valid_corr'] = None
        row['test_MAE'] = row['test_Corr'] = None
    return row


def _print_and_save_table(results, sweep_dir, milestone_epoch):
    headers = ['run_name', 'batch_size', 'lr', 'n_epochs', 'best_valid_mae', 'best_valid_corr',
               'test_MAE', 'test_Corr', 'best_epoch_mae', 'best_epoch_corr', 'checkpoint_dir', 'ok']
    if milestone_epoch is not None:
        table_path = os.path.join(sweep_dir, 'sweep_results_epoch{}.csv'.format(milestone_epoch))
        title = "实验对比表 (epoch {} 里程碑)".format(milestone_epoch)
    else:
        table_path = os.path.join(sweep_dir, 'sweep_results.csv')
        title = "实验对比表"
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write(','.join(headers) + '\n')
        for r in results:
            def cell(h):
                v = r.get(h)
                if v is None:
                    return ''
                s = str(v)
                return '"' + s.replace('"', '""') + '"' if (h == 'checkpoint_dir' and s) else s
            f.write(','.join(cell(h) for h in headers) + '\n')

    print("\n" + "=" * 100)
    print("{} (CSV: {})".format(title, table_path))
    print("=" * 100)
    fmt = "{:14} | {:>6} | {:>8} | {:>6} | {:>12} | {:>12} | {:>10} | {:>10} | {:>6}"
    print(fmt.format(
        "run_name", "batch", "lr", "epochs",
        "best_val_mae", "best_val_corr", "test_MAE", "test_Corr", "ok"
    ))
    print("-" * 100)
    for r in results:
        bv_mae = r.get('best_valid_mae')
        bv_corr = r.get('best_valid_corr')
        t_mae = r.get('test_MAE')
        t_corr = r.get('test_Corr')
        print(fmt.format(
            r.get('run_name', '')[:14],
            r.get('batch_size', ''),
            "{:.0e}".format(r.get('lr', 0)) if r.get('lr') else '',
            r.get('n_epochs', ''),
            "{:.4f}".format(bv_mae) if bv_mae is not None else "-",
            "{:.4f}".format(bv_corr) if bv_corr is not None else "-",
            "{:.4f}".format(t_mae) if t_mae is not None else "-",
            "{:.4f}".format(t_corr) if t_corr is not None else "-",
            "Y" if r.get('ok') else "N",
        ))
    print("=" * 100)
    valid = [r for r in results if r.get('ok') and r.get('test_MAE') is not None]
    if valid:
        best_mae = min(valid, key=lambda x: (x['test_MAE'], -x['test_Corr']))
        best_corr = max(valid, key=lambda x: (x['test_Corr'], -x['test_MAE']))
        print("\n推荐配置:")
        print("  按 Test MAE 最佳: {} (test_MAE={:.4f}, test_Corr={:.4f})".format(
            best_mae['run_name'], best_mae['test_MAE'], best_mae['test_Corr']))
        print("  按 Test Corr 最佳: {} (test_MAE={:.4f}, test_Corr={:.4f})".format(
            best_corr['run_name'], best_corr['test_MAE'], best_corr['test_Corr']))


if __name__ == '__main__':
    sys.exit(main())
