"""
sweep_multiconfig.py
====================
并行跑多套命名配置，充分利用多 GPU 或单 GPU 串行，
每个配置完成后读取 summary.json 打印排行榜。

用法示例：
    # 利用 GPU 0 串行跑全部 9 个配置（每次一个）
    python scripts/sweep_multiconfig.py --gpus 0 --run_dir ./checkpoints/sweep_sota

    # 双 GPU 并行（GPU 0 和 GPU 1）
    python scripts/sweep_multiconfig.py --gpus 0,1 --run_dir ./checkpoints/sweep_sota

    # 只跑指定名字（逗号分隔）
    python scripts/sweep_multiconfig.py --gpus 0 --run_dir ./ckpts/sweep --only c2,c4

    # dry-run 预览指令但不实际跑
    python scripts/sweep_multiconfig.py --gpus 0 --run_dir ./ckpts --dry_run
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])  # /home/yechenlu/KuDA

# ─────────────────────────────────────────────
#  配置列表：用以逼近/超过 SOTA（MAE 0.408, Corr 0.613）
#  策略矩阵：
#    loss_fn × dropout × lr × path_layers × seed
# ─────────────────────────────────────────────
COMMON = dict(
    datasetName='sims',
    model_type='pid_dualpath',
    use_batch_pid_prior=True,
    pid_warmup_epochs=5,
    router_tau=0.3,
    lambda_alpha_var=0.12,
    freeze_bert=True,
    weight_decay='1e-4',
    n_epochs=60,
    batch_size=32,
)

CONFIGS = [
    # ── baseline（SmoothL1, 原版）──────────────────────────────
    dict(name='c1_base_sl1',
         lr='3e-5', dropout=0.4, loss_fn='smoothl1', path_layers=2, seed=1111),

    # ── L1 Loss：直接最小化 MAE ──────────────────────────────
    dict(name='c2_l1',
         lr='3e-5', dropout=0.4, loss_fn='l1', path_layers=2, seed=1111),

    # ── L1 + 更强正则 ──────────────────────────────
    dict(name='c3_l1_wd5e4',
         lr='3e-5', dropout=0.45, loss_fn='l1', path_layers=2,
         weight_decay='5e-4', seed=1111),

    # ── L1 + 更深 Transformer (path_layers=3) ──────────────────
    dict(name='c4_l1_layers3',
         lr='3e-5', dropout=0.4, loss_fn='l1', path_layers=3, seed=1111),

    # ── L1 + layers=3 + 更强正则 ──────────────────────────────
    dict(name='c5_l1_layers3_wd5e4',
         lr='3e-5', dropout=0.5, loss_fn='l1', path_layers=3,
         weight_decay='5e-4', seed=1111),

    # ── SmoothL1 + layers=3 ──────────────────────────────
    dict(name='c6_sl1_layers3',
         lr='3e-5', dropout=0.4, loss_fn='smoothl1', path_layers=3, seed=1111),

    # ── L1 + 较低 lr ──────────────────────────────
    dict(name='c7_l1_lr2e5',
         lr='2e-5', dropout=0.4, loss_fn='l1', path_layers=2, seed=1111),

    # ── L1 + 多 seed（重复最优 c2 配置以估计方差）
    dict(name='c8_l1_seed42',
         lr='3e-5', dropout=0.4, loss_fn='l1', path_layers=2, seed=42),

    dict(name='c9_l1_seed2024',
         lr='3e-5', dropout=0.4, loss_fn='l1', path_layers=2, seed=2024),
]


def build_cmd(cfg, run_dir, gpu_id):
    c = {**COMMON, **cfg}
    ckpt = os.path.join(run_dir, c['name'])
    cmd = [
        sys.executable, 'train.py',
        '--datasetName', c['datasetName'],
        '--model_type', c['model_type'],
        '--use_batch_pid_prior', str(c['use_batch_pid_prior']),
        '--pid_warmup_epochs', str(c['pid_warmup_epochs']),
        '--router_tau', str(c['router_tau']),
        '--lambda_alpha_var', str(c['lambda_alpha_var']),
        '--freeze_bert', str(c['freeze_bert']),
        '--weight_decay', str(c.get('weight_decay', COMMON['weight_decay'])),
        '--n_epochs', str(c['n_epochs']),
        '--batch_size', str(c['batch_size']),
        '--lr', str(c['lr']),
        '--dropout', str(c['dropout']),
        '--loss_fn', c['loss_fn'],
        '--path_layers', str(c['path_layers']),
        '--seed', str(c.get('seed', 1111)),
        '--checkpoint_dir', ckpt,
        '--gpu', str(gpu_id),
    ]
    return cmd, ckpt


def load_summary(ckpt_dir):
    p = os.path.join(ckpt_dir, 'summary.json')
    if not os.path.isfile(p):
        return None
    with open(p, 'r') as f:
        return json.load(f)


def print_leaderboard(results):
    print('\n' + '=' * 90)
    print(f"{'Name':<28} {'MAE':>7} {'Corr':>7} {'Acc2':>7} {'F1':>7}  status")
    print('-' * 90)
    sorted_r = sorted(results, key=lambda x: x.get('mae', 9) if x.get('done') else 9)
    for r in sorted_r:
        if r.get('done'):
            print(f"{r['name']:<28} {r['mae']:>7.4f} {r['corr']:>7.4f} {r['acc2']:>7.2f} {r['f1']:>7.4f}")
        else:
            print(f"{r['name']:<28}  (running/pending)")
    print('=' * 90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpus', default='0',
                    help='可用 GPU 编号，逗号分隔，如 "0,1"。配置按顺序排队，队列深度 = GPU 数量')
    ap.add_argument('--run_dir', default='./checkpoints/sweep_sota',
                    help='sweep 输出根目录')
    ap.add_argument('--only', default='',
                    help='只跑指定配置名（逗号分隔），如 "c2,c4"，空表示全跑')
    ap.add_argument('--dry_run', action='store_true')
    ap.add_argument('--max_parallel', type=int, default=0,
                    help='最大并行数，0 = GPU 数量')
    args = ap.parse_args()

    gpus = [g.strip() for g in args.gpus.split(',')]
    max_par = args.max_parallel or len(gpus)
    run_dir = os.path.abspath(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)

    only_set = set(x.strip() for x in args.only.split(',') if x.strip())
    configs = [c for c in CONFIGS if not only_set or c['name'] in only_set]
    n = len(configs)
    print(f'[sweep_multiconfig] {n} 个配置，GPU={gpus}，max_parallel={max_par}')
    print(f'输出目录: {run_dir}\n')

    # 构造 (cmd, ckpt, name) 队列
    tasks = []
    for i, cfg in enumerate(configs):
        gpu = gpus[i % len(gpus)]
        cmd, ckpt = build_cmd(cfg, run_dir, gpu)
        tasks.append({'name': cfg['name'], 'cmd': cmd, 'ckpt': ckpt,
                      'gpu': gpu, 'proc': None, 'done': False, 'mae': 9,
                      'corr': 0, 'acc2': 0, 'f1': 0})
        if args.dry_run:
            print(f"[dry-run] {cfg['name']}")
            print('  ' + ' '.join(cmd))

    if args.dry_run:
        return

    running = []   # 当前在跑的 task dict
    pending = list(tasks)
    completed = []

    def try_collect_finished():
        still_running = []
        for t in running:
            rc = t['proc'].poll()
            if rc is not None:
                s = load_summary(t['ckpt'])
                if s:
                    t['done'] = True
                    # summary.json 结构：test_at_best_mae 包含 Best-MAE 对应测试结果
                    best_mae_res = s.get('test_at_best_mae', {})
                    t['mae'] = best_mae_res.get('MAE', 9)
                    t['corr'] = best_mae_res.get('Corr', 0)
                    t['acc2'] = best_mae_res.get('Mult_acc_2', 0)
                    t['f1'] = best_mae_res.get('F1_score', 0)
                    print(f'\n[完成] {t["name"]}  MAE={t["mae"]:.4f}  Corr={t["corr"]:.4f}  Acc2={t["acc2"]:.2f}  F1={t["f1"]:.4f}  (rc={rc})')
                else:
                    t['done'] = True
                    print(f'\n[完成-无summary] {t["name"]}  (rc={rc})')
                completed.append(t)
            else:
                still_running.append(t)
        running.clear()
        running.extend(still_running)

    print('[开始训练]\n')
    all_tasks = list(tasks)

    while pending or running:
        try_collect_finished()
        # 补充任务直到 max_par
        while pending and len(running) < max_par:
            t = pending.pop(0)
            print(f'[启动] {t["name"]}  GPU={t["gpu"]}')
            proc = subprocess.Popen(t['cmd'], cwd=ROOT,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
            t['proc'] = proc
            running.append(t)
        time.sleep(5)

    print_leaderboard(all_tasks)

    # 保存排行榜到 run_dir/leaderboard.csv
    lb_path = os.path.join(run_dir, 'leaderboard.csv')
    with open(lb_path, 'w') as f:
        f.write('name,mae,corr,acc2,f1\n')
        for t in sorted(all_tasks, key=lambda x: x.get('mae', 9) if x.get('done') else 9):
            f.write(f"{t['name']},{t.get('mae', '')},{t.get('corr', '')},"
                    f"{t.get('acc2', '')},{t.get('f1', '')}\n")
    print(f'\n排行榜已保存至 {lb_path}')


if __name__ == '__main__':
    main()
