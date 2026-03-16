"""
run_ablation_aux_ortho.py
=========================
消融：lambda_aux、lambda_ortho 各减半；极端样本加权(lambda_extreme)。

配置：
  baseline:        lambda_aux=0.1,  lambda_ortho=0.005
  ablation_half:   lambda_aux=0.05, lambda_ortho=0.0025
  extreme_weight:  ablation_half + lambda_extreme=0.4（对|y|大样本加权，缓解向0收缩）
  ablation_half_ranking: ablation_half + lambda_rank=0.08（Pairwise MPR，缓解预测收缩）

用法：
    # 各 2 seed，4 张 GPU 并行
    python scripts/run_ablation_aux_ortho.py --seeds 1111,42 --gpus 0,1,2,3 --run_dir ./checkpoints/ablation_aux_ortho

    # 各 3 seed 更稳
    python scripts/run_ablation_aux_ortho.py --seeds 1111,42,2024 --gpus 0,1,2,3,4,5 --run_dir ./checkpoints/ablation_aux_ortho
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
    dict(name='baseline', lambda_aux=0.1, lambda_ortho=0.005, lambda_extreme=0.0, lambda_rank=0.0),
    dict(name='ablation_half', lambda_aux=0.05, lambda_ortho=0.0025, lambda_extreme=0.0, lambda_rank=0.0),
    dict(name='extreme_weight', lambda_aux=0.05, lambda_ortho=0.0025, lambda_extreme=0.4, lambda_rank=0.0),
    dict(name='ablation_half_ranking', lambda_aux=0.05, lambda_ortho=0.0025, lambda_extreme=0.0, lambda_rank=0.08, rank_margin=0.2, ranking_threshold=0.1, use_dynamic_margin_rank=False),  # 静态 margin
    dict(name='ablation_half_ranking_dynamic', lambda_aux=0.05, lambda_ortho=0.0025, lambda_extreme=0.0, lambda_rank=0.08, use_dynamic_margin_rank=True, rank_margin_gamma=0.6, ranking_threshold=0.1),  # 动态 margin
    dict(name='ablation_half_grcf', lambda_aux=0.05, lambda_ortho=0.0025, lambda_extreme=0.0, lambda_rank=0.1, use_grcf=True, grcf_stage1_epochs=40, grcf_stage1_regression_weight=0.1, use_dynamic_margin_rank=True, rank_margin_gamma=0.6),
]


def build_cmd(cfg, seed, run_dir, gpu_id):
    c = {**BASE, **cfg}
    ckpt = os.path.join(run_dir, c['name'], f"seed_{seed}")
    cmd = [
        sys.executable, 'train.py',
        '--datasetName', c['datasetName'],
        '--model_type', c['model_type'],
        '--use_batch_pid_prior', str(c['use_batch_pid_prior']),
        '--pid_warmup_epochs', str(c['pid_warmup_epochs']),
        '--router_tau', str(c['router_tau']),
        '--lambda_alpha_var', str(c['lambda_alpha_var']),
        '--lambda_aux', str(c['lambda_aux']),
        '--lambda_ortho', str(c['lambda_ortho']),
        '--lambda_extreme', str(c.get('lambda_extreme', 0.0)),
        '--lambda_rank', str(c.get('lambda_rank', 0.0)),
        '--rank_margin', str(c.get('rank_margin', 0.2)),
        '--ranking_threshold', str(c.get('ranking_threshold', 0.1)),
        '--use_dynamic_margin_rank', str(c.get('use_dynamic_margin_rank', False)),
        '--rank_margin_gamma', str(c.get('rank_margin_gamma', 0.6)),
        '--use_grcf', str(c.get('use_grcf', False)),
        '--grcf_stage1_epochs', str(c.get('grcf_stage1_epochs', 40)),
        '--grcf_stage1_regression_weight', str(c.get('grcf_stage1_regression_weight', 0.1)),
        '--freeze_bert', str(c['freeze_bert']),
        '--weight_decay', c['weight_decay'],
        '--n_epochs', str(c['n_epochs']),
        '--batch_size', str(c['batch_size']),
        '--lr', c['lr'],
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
    ap.add_argument('--seeds', default='1111,42,2024,123,456', help='逗号分隔的 seed，推荐 5 个验证稳定性')
    ap.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    ap.add_argument('--run_dir', default='./checkpoints/ablation_aux_ortho')
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    seeds = [s.strip() for s in args.seeds.split(',') if s.strip()]
    gpus = [g.strip() for g in args.gpus.split(',')]
    run_dir = os.path.abspath(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)

    tasks = []
    for cfg in CONFIGS:
        for seed in seeds:
            gpu = gpus[len(tasks) % len(gpus)]
            cmd, ckpt = build_cmd(cfg, seed, run_dir, gpu)
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

    print(f'[ablation] {len(tasks)} 任务, GPU={gpus}')
    print(f'  baseline:        lambda_aux=0.1,  lambda_ortho=0.005')
    print(f'  ablation_half:   lambda_aux=0.05, lambda_ortho=0.0025')
    print(f'  extreme_weight:  ablation_half + lambda_extreme=0.4')
    print(f'  ablation_half_ranking: ablation_half + lambda_rank=0.08 (Pairwise MPR)')
    print(f'  ablation_half_ranking_dynamic: + use_dynamic_margin_rank, gamma=0.6')
    print(f'  ablation_half_grcf: two-stage GRCF (40ep ranking + 20ep MAE calibration)\n')

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
    print('消融汇总 (test_at_best_mae)')
    print('=' * 70)
    cfg_names = ['baseline', 'ablation_half', 'extreme_weight', 'ablation_half_ranking', 'ablation_half_ranking_dynamic', 'ablation_half_grcf']
    for cfg_name in cfg_names:
        subset = [t for t in tasks if t['cfg'] == cfg_name and t.get('done')]
        if not subset:
            print(f'  {cfg_name}: (无完成)')
            continue
        mae_list = [t['mae'] for t in subset]
        corr_list = [t['corr'] for t in subset]
        mae_mean = sum(mae_list) / len(mae_list)
        corr_mean = sum(corr_list) / len(corr_list)
        mae_std = (sum((x - mae_mean) ** 2 for x in mae_list) / len(mae_list)) ** 0.5 if len(mae_list) > 1 else 0
        print(f'  {cfg_name}:  MAE={mae_mean:.4f}±{mae_std:.4f}  Corr={corr_mean:.4f}  (n={len(subset)})')
        for t in subset:
            print(f'    seed_{t["seed"]}: MAE={t["mae"]:.4f}  Corr={t["corr"]:.4f}')
    print('=' * 70)
    print(f'\n完成后可运行诊断：')
    for cname in cfg_names:
        print(f'  python scripts/analyze_residual_by_interval.py --ckpt_dir {run_dir}/{cname} --tune_weights --calibrate linear --gpu 0')
        print(f'  python scripts/plot_prediction_distributions.py --ckpt_dir {run_dir}/{cname} --tune_weights -o dist_{cname}.png --gpu 0')


if __name__ == '__main__':
    main()
