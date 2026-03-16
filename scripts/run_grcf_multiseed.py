"""
run_grcf_multiseed.py
=====================
GRCF 两阶段训练 + baseline 对照，多 seed 验证稳定性。

配置：
  baseline: 无 ranking、无 GRCF，作为对照
  grcf:     两阶段 GRCF（40ep ranking + 20ep MAE 校准），动态 margin

用法：
    # 5 seed 验证稳定性（推荐）
    python scripts/run_grcf_multiseed.py --seeds 1111,42,2024,123,456 --gpus 0,1,2,3,4,5,6,7 --run_dir ./checkpoints/grcf_multiseed

    # 调整 Stage1 比例（默认 40/60）
    python scripts/run_grcf_multiseed.py --seeds 1111,42,2024 --grcf_stage1_epochs 45 --run_dir ./checkpoints/grcf_stage45

    # dry-run 查看命令
    python scripts/run_grcf_multiseed.py --seeds 1111,42 --dry_run
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])

BASE = dict(
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

CONFIGS = [
    dict(name='baseline', lambda_aux=0.1, lambda_ortho=0.005, lambda_rank=0.0),
    dict(name='grcf', lambda_aux=0.05, lambda_ortho=0.0025, lambda_rank=0.1,
         use_grcf=True, grcf_stage1_regression_weight=0.1,
         use_dynamic_margin_rank=True, rank_margin_gamma=0.6, ranking_threshold=0.1),
]


def build_cmd(cfg, seed, run_dir, gpu_id, overrides):
    c = {**BASE, **cfg, **overrides}
    ckpt = os.path.join(run_dir, c['name'], f"seed_{seed}")
    cmd = [
        sys.executable, 'train.py',
        '--datasetName', c['datasetName'],
        '--model_type', c['model_type'],
        '--use_batch_pid_prior', str(c['use_batch_pid_prior']),
        '--pid_warmup_epochs', str(c.get('pid_warmup_epochs', 5)),
        '--router_tau', str(c['router_tau']),
        '--lambda_alpha_var', str(c['lambda_alpha_var']),
        '--lambda_aux', str(c['lambda_aux']),
        '--lambda_ortho', str(c['lambda_ortho']),
        '--lambda_rank', str(c.get('lambda_rank', 0.0)),
        '--use_dynamic_margin_rank', str(c.get('use_dynamic_margin_rank', False)),
        '--rank_margin_gamma', str(c.get('rank_margin_gamma', 0.6)),
        '--ranking_threshold', str(c.get('ranking_threshold', 0.1)),
        '--use_grcf', str(c.get('use_grcf', False)),
        '--grcf_stage1_epochs', str(c.get('grcf_stage1_epochs', 40)),
        '--grcf_stage1_regression_weight', str(c.get('grcf_stage1_regression_weight', 0.1)),
        '--freeze_bert', str(c['freeze_bert']),
        '--weight_decay', c['weight_decay'],
        '--n_epochs', str(c['n_epochs']),
        '--batch_size', str(c['batch_size']),
        '--lr', str(c.get('lr', '3e-5')),
        '--dropout', str(c['dropout']),
        '--loss_fn', c['loss_fn'],
        '--path_layers', str(c['path_layers']),
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
                    help='逗号分隔的 seed，推荐 5 个验证稳定性')
    ap.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    ap.add_argument('--run_dir', default='./checkpoints/grcf_multiseed')
    ap.add_argument('--grcf_stage1_epochs', type=int, default=40,
                    help='GRCF Stage1 轮数，Stage2 = n_epochs - 此值')
    ap.add_argument('--grcf_regression_weight', type=float, default=0.1,
                    help='Stage1 主回归损失权重，越小越偏 ranking')
    ap.add_argument('--lr', default=None, help='覆盖学习率，如 2e-5')
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    seeds = [s.strip() for s in args.seeds.split(',') if s.strip()]
    gpus = [s.strip() for s in args.gpus.split(',')]
    run_dir = os.path.abspath(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)

    overrides = {
        'grcf_stage1_epochs': args.grcf_stage1_epochs,
        'grcf_stage1_regression_weight': args.grcf_regression_weight,
    }
    if args.lr:
        overrides['lr'] = args.lr

    tasks = []
    for cfg in CONFIGS:
        for seed in seeds:
            gpu = gpus[len(tasks) % len(gpus)]
            cmd, ckpt = build_cmd(cfg, seed, run_dir, gpu, overrides)
            tasks.append({
                'name': f"{cfg['name']}_seed{seed}",
                'cfg': cfg['name'],
                'seed': seed,
                'cmd': cmd,
                'ckpt': ckpt,
                'gpu': gpu,
                'proc': None,
                'done': False,
                'mae': 9.0, 'corr': 0.0,
            })
            if args.dry_run:
                print(f"[dry-run] {cfg['name']} seed={seed} GPU={gpu}")
                print('  ' + ' '.join(cmd))

    if args.dry_run:
        return

    print(f'[GRCF multiseed] {len(tasks)} 任务, seeds={seeds}, gpus={gpus}')
    print(f'  baseline: 对照（无 ranking/GRCF）')
    print(f'  grcf:     两阶段 ({args.grcf_stage1_epochs}ep ranking + {60-args.grcf_stage1_epochs}ep MAE)\n')

    running = []
    pending = list(tasks)
    max_par = len(gpus)

    def collect():
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
                    print(f'\n[完成] {t["name"]}  MAE={t["mae"]:.4f}  Corr={t["corr"]:.4f}')
                else:
                    t['done'] = True
                    print(f'\n[完成-无summary] {t["name"]}')
            else:
                still.append(t)
        running.clear()
        running.extend(still)

    while pending or running:
        collect()
        while pending and len(running) < max_par:
            t = pending.pop(0)
            print(f'[启动] {t["name"]}  GPU={t["gpu"]}')
            t['proc'] = subprocess.Popen(t['cmd'], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            running.append(t)
        time.sleep(5)

    # 汇总
    print('\n' + '=' * 70)
    print('GRCF vs Baseline 汇总 (test_at_best_mae)')
    print('=' * 70)
    for cfg_name in ['baseline', 'grcf']:
        subset = [t for t in tasks if t['cfg'] == cfg_name and t.get('done')]
        if not subset:
            print(f'  {cfg_name}: (无完成)')
            continue
        mae_list = [t['mae'] for t in subset]
        corr_list = [t['corr'] for t in subset]
        mae_mean = sum(mae_list) / len(mae_list)
        corr_mean = sum(corr_list) / len(corr_list)
        mae_std = (sum((x - mae_mean) ** 2 for x in mae_list) / len(mae_list)) ** 0.5 if len(mae_list) > 1 else 0
        best_mae = min(mae_list)
        print(f'  {cfg_name}:  MAE={mae_mean:.4f}±{mae_std:.4f}  Corr={corr_mean:.4f}  best={best_mae:.4f}  (n={len(subset)})')
        for t in subset:
            print(f'    seed_{t["seed"]}: MAE={t["mae"]:.4f}  Corr={t["corr"]:.4f}')
    print('=' * 70)
    print(f'\n诊断命令：')
    print(f'  python scripts/analyze_residual_by_interval.py --ckpt_dir {run_dir}/grcf --tune_weights --calibrate linear --gpu 0')
    print(f'  python scripts/plot_prediction_distributions.py --ckpt_dir {run_dir}/grcf --tune_weights -o dist_grcf.png --gpu 0')


if __name__ == '__main__':
    main()
