"""
sweep_multiseed.py
==================
固定「最佳配置」、仅变换随机种子，多 GPU 并行跑多 seed，
结束后汇总 mean±std，用于报告稳定结果或选最佳单次 run。

最佳配置（对应 sweep 中 c1_base_sl1）：SmoothL1, lr=3e-5, path_layers=2,
freeze_bert=True, dropout=0.4, weight_decay=1e-4, n_epochs=60, batch_size=32.

用法示例：
    # 5 个 seed，8 张 GPU 并行（前 5 个同时跑）
    python scripts/sweep_multiseed.py --seeds 1111,42,2024,123,456 --gpus 0,1,2,3,4,5,6,7 --run_dir ./checkpoints/multiseed_best

    # 默认 5 seed，用 4 张 GPU
    python scripts/sweep_multiseed.py --gpus 0,1,2,3 --run_dir ./checkpoints/multiseed_best

    # dry-run 只打印命令
    python scripts/sweep_multiseed.py --seeds 1111,42 --gpus 0,1 --run_dir ./ckpts --dry_run
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])

# 固定最佳配置（与 c1_base_sl1 一致）
BEST_CONFIG = dict(
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
    lr='3e-5',
    dropout=0.4,
    loss_fn='smoothl1',
    path_layers=2,
)


def build_cmd(seed, run_dir, gpu_id):
    ckpt = os.path.join(run_dir, f'seed_{seed}')
    cmd = [
        sys.executable, 'train.py',
        '--datasetName', BEST_CONFIG['datasetName'],
        '--model_type', BEST_CONFIG['model_type'],
        '--use_batch_pid_prior', str(BEST_CONFIG['use_batch_pid_prior']),
        '--pid_warmup_epochs', str(BEST_CONFIG['pid_warmup_epochs']),
        '--router_tau', str(BEST_CONFIG['router_tau']),
        '--lambda_alpha_var', str(BEST_CONFIG['lambda_alpha_var']),
        '--freeze_bert', str(BEST_CONFIG['freeze_bert']),
        '--weight_decay', BEST_CONFIG['weight_decay'],
        '--n_epochs', str(BEST_CONFIG['n_epochs']),
        '--batch_size', str(BEST_CONFIG['batch_size']),
        '--lr', BEST_CONFIG['lr'],
        '--dropout', str(BEST_CONFIG['dropout']),
        '--loss_fn', BEST_CONFIG['loss_fn'],
        '--path_layers', str(BEST_CONFIG['path_layers']),
        '--seed', str(seed),
        '--checkpoint_dir', ckpt,
        '--gpu', str(gpu_id),
    ]
    return cmd, ckpt


def load_summary(ckpt_dir):
    p = os.path.join(ckpt_dir, 'summary.json')
    if not os.path.isfile(p):
        return None
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='1111,42,2024,123,456',
                    help='逗号分隔的随机种子，如 1111,42,2024')
    ap.add_argument('--gpus', default='0,1,2,3,4,5,6,7',
                    help='可用 GPU 编号，逗号分隔')
    ap.add_argument('--run_dir', default='./checkpoints/multiseed_best',
                    help='输出根目录，每个 seed 一个子目录 seed_<n>')
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    seeds = [s.strip() for s in args.seeds.split(',') if s.strip()]
    gpus = [g.strip() for g in args.gpus.split(',')]
    run_dir = os.path.abspath(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)

    print('[sweep_multiseed] 固定最佳配置，多 seed 并行')
    print(f'  seeds={seeds}, gpus={gpus}, run_dir={run_dir}\n')

    tasks = []
    for i, seed in enumerate(seeds):
        gpu = gpus[i % len(gpus)]
        cmd, ckpt = build_cmd(seed, run_dir, gpu)
        tasks.append({
            'name': f'seed_{seed}',
            'seed': seed,
            'cmd': cmd,
            'ckpt': ckpt,
            'gpu': gpu,
            'proc': None,
            'done': False,
            'mae': 9.0, 'corr': 0.0, 'acc2': 0.0, 'f1': 0.0,
        })
        if args.dry_run:
            print(f"[dry-run] seed_{seed}  GPU={gpu}")
            print('  ' + ' '.join(cmd))

    if args.dry_run:
        return

    running = []
    pending = list(tasks)
    max_par = len(gpus)

    def collect_finished():
        still = []
        for t in running:
            rc = t['proc'].poll()
            if rc is not None:
                s = load_summary(t['ckpt'])
                if s:
                    t['done'] = True
                    res = s.get('test_at_best_mae', {})
                    t['mae'] = res.get('MAE', 9)
                    t['corr'] = res.get('Corr', 0)
                    t['acc2'] = res.get('Mult_acc_2', 0)
                    t['f1'] = res.get('F1_score', 0)
                    print(f'\n[完成] {t["name"]}  MAE={t["mae"]:.4f}  Corr={t["corr"]:.4f}  Acc2={t["acc2"]:.2f}  F1={t["f1"]:.4f}  (rc={rc})')
                else:
                    t['done'] = True
                    print(f'\n[完成-无summary] {t["name"]}  (rc={rc})')
            else:
                still.append(t)
        running.clear()
        running.extend(still)

    print('[开始训练]\n')
    while pending or running:
        collect_finished()
        while pending and len(running) < max_par:
            t = pending.pop(0)
            print(f'[启动] {t["name"]}  GPU={t["gpu"]}')
            t['proc'] = subprocess.Popen(
                t['cmd'], cwd=ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            running.append(t)
        time.sleep(5)

    # 汇总 mean±std
    done = [t for t in tasks if t.get('done') and isinstance(t.get('mae'), (int, float))]
    if not done:
        print('\n无有效完成 run，无法计算 mean±std')
        return

    import statistics
    mae_list = [t['mae'] for t in done]
    corr_list = [t['corr'] for t in done]
    acc2_list = [t['acc2'] for t in done]
    f1_list = [t['f1'] for t in done]

    n = len(done)
    mae_mean = statistics.mean(mae_list)
    mae_std = statistics.stdev(mae_list) if n > 1 else 0.0
    corr_mean = statistics.mean(corr_list)
    corr_std = statistics.stdev(corr_list) if n > 1 else 0.0
    acc2_mean = statistics.mean(acc2_list)
    acc2_std = statistics.stdev(acc2_list) if n > 1 else 0.0
    f1_mean = statistics.mean(f1_list)
    f1_std = statistics.stdev(f1_list) if n > 1 else 0.0

    print('\n' + '=' * 70)
    print('多 seed 汇总 (固定最佳配置, test_at_best_mae)')
    print('=' * 70)
    for t in sorted(tasks, key=lambda x: x.get('mae', 9)):
        if t.get('done'):
            print(f"  {t['name']:<16}  MAE={t['mae']:.4f}  Corr={t['corr']:.4f}  Acc2={t['acc2']:.2f}  F1={t['f1']:.4f}")
    print('-' * 70)
    print(f"  mean ± std (n={n})  MAE={mae_mean:.4f}±{mae_std:.4f}  Corr={corr_mean:.4f}±{corr_std:.4f}  Acc2={acc2_mean:.2f}±{acc2_std:.2f}  F1={f1_mean:.4f}±{f1_std:.4f}")
    print('=' * 70)

    best_single = min(done, key=lambda x: x['mae'])
    summary = {
        'config': BEST_CONFIG,
        'seeds': seeds,
        'n_runs': n,
        'MAE_mean': mae_mean, 'MAE_std': mae_std, 'MAE_list': mae_list,
        'Corr_mean': corr_mean, 'Corr_std': corr_std, 'Corr_list': corr_list,
        'Acc2_mean': acc2_mean, 'Acc2_std': acc2_std, 'Acc2_list': acc2_list,
        'F1_mean': f1_mean, 'F1_std': f1_std, 'F1_list': f1_list,
        'best_single_seed': best_single['seed'],
        'best_single_mae': best_single['mae'],
        'per_seed': [{'seed': t['seed'], 'mae': t['mae'], 'corr': t['corr'], 'acc2': t['acc2'], 'f1': t['f1']} for t in done],
    }
    out_json = os.path.join(run_dir, 'multiseed_summary.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\n汇总已保存: {out_json}')

    csv_path = os.path.join(run_dir, 'multiseed_leaderboard.csv')
    with open(csv_path, 'w') as f:
        f.write('name,seed,mae,corr,acc2,f1\n')
        for t in sorted(tasks, key=lambda x: x.get('mae', 9) if x.get('done') else 9):
            if t.get('done'):
                f.write(f"{t['name']},{t['seed']},{t['mae']:.4f},{t['corr']:.4f},{t['acc2']:.2f},{t['f1']:.4f}\n")
    print(f'排行榜 CSV: {csv_path}')


if __name__ == '__main__':
    main()
