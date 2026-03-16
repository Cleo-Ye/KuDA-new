"""
run_grcf_deep_sweep.py
======================
在 grcf_deep 架构上做小范围超参搜索：lr × grcf_stage1_epochs。

网格：
  lr: 2e-5, 3e-5, 5e-5
  grcf_stage1_epochs: 45, 50, 55

共 9 个配置，每个 1 个 seed（可指定多 seed 更稳）。

用法：
    python scripts/run_grcf_deep_sweep.py --seeds 1111,42 --gpus 0,1,2,3,4,5,6,7,8 --run_dir ./checkpoints/grcf_deep_sweep
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
    lambda_aux=0.05,
    lambda_ortho=0.0025,
    lambda_rank=0.12,
    use_grcf=True,
    grcf_stage1_regression_weight=0.08,
    n_epochs=60,
    batch_size=32,
    dropout=0.35,
    loss_fn='smoothl1',
    path_layers=3,
    hidden_size=512,
    freeze_bert=True,
    weight_decay='1e-4',
    use_dynamic_margin_rank=True,
    rank_margin_gamma=0.7,
    ranking_threshold=0.1,
)

# 网格: lr × grcf_stage1_epochs
LR_OPTIONS = ['2e-5', '3e-5', '5e-5']
STAGE1_OPTIONS = [45, 50, 55]


def build_cmd(lr, stage1, seed, run_dir, gpu_id):
    name = f"lr{lr.replace('e-', 'e')}_s1{stage1}"
    ckpt = os.path.join(run_dir, name, f"seed_{seed}")
    # stage1=55 时 n_epochs=60 仅 5ep 校准，提高到 70 保证至少 15ep
    n_epochs = max(60, stage1 + 15)
    cmd = [
        sys.executable, 'train.py',
        '--datasetName', BASE['datasetName'],
        '--model_type', BASE['model_type'],
        '--use_batch_pid_prior', str(BASE['use_batch_pid_prior']),
        '--pid_warmup_epochs', str(BASE['pid_warmup_epochs']),
        '--router_tau', str(BASE['router_tau']),
        '--lambda_alpha_var', str(BASE['lambda_alpha_var']),
        '--lambda_aux', str(BASE['lambda_aux']),
        '--lambda_ortho', str(BASE['lambda_ortho']),
        '--lambda_rank', str(BASE['lambda_rank']),
        '--use_grcf', str(BASE['use_grcf']),
        '--grcf_stage1_epochs', str(stage1),
        '--grcf_stage1_regression_weight', str(BASE['grcf_stage1_regression_weight']),
        '--n_epochs', str(n_epochs),
        '--batch_size', str(BASE['batch_size']),
        '--lr', lr,
        '--dropout', str(BASE['dropout']),
        '--loss_fn', BASE['loss_fn'],
        '--path_layers', str(BASE['path_layers']),
        '--hidden_size', str(BASE['hidden_size']),
        '--freeze_bert', str(BASE['freeze_bert']),
        '--weight_decay', BASE['weight_decay'],
        '--use_dynamic_margin_rank', str(BASE['use_dynamic_margin_rank']),
        '--rank_margin_gamma', str(BASE['rank_margin_gamma']),
        '--ranking_threshold', str(BASE['ranking_threshold']),
        '--seed', str(seed),
        '--checkpoint_dir', ckpt,
        '--gpu', str(gpu_id),
    ]
    return cmd, ckpt, name


def load_summary(ckpt_dir):
    p = os.path.join(ckpt_dir, 'summary.json')
    if not os.path.isfile(p):
        return None
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='1111', help='逗号分隔，单 seed 快速扫，多 seed 更稳')
    ap.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    ap.add_argument('--run_dir', default='./checkpoints/grcf_deep_sweep')
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    seeds = [s.strip() for s in args.seeds.split(',') if s.strip()]
    gpus = [s.strip() for s in args.gpus.split(',')]
    run_dir = os.path.abspath(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)

    tasks = []
    for lr in LR_OPTIONS:
        for stage1 in STAGE1_OPTIONS:
            for seed in seeds:
                gpu = gpus[len(tasks) % len(gpus)]
                cmd, ckpt, name = build_cmd(lr, stage1, seed, run_dir, gpu)
                tasks.append({
                    'name': f"{name}_seed{seed}",
                    'cfg': name,
                    'lr': lr,
                    'stage1': stage1,
                    'seed': seed,
                    'cmd': cmd,
                    'ckpt': ckpt,
                    'gpu': gpu,
                    'proc': None,
                    'done': False,
                    'mae': 9.0, 'corr': 0.0,
                })
                if args.dry_run:
                    print(f"[dry-run] {name} seed={seed} GPU={gpu}")
                    print('  ' + ' '.join(cmd))

    if args.dry_run:
        return

    print(f'[grcf_deep sweep] {len(tasks)} 任务')
    print(f'  lr: {LR_OPTIONS}, grcf_stage1_epochs: {STAGE1_OPTIONS}')
    print(f'  seeds={seeds}, gpus={gpus}\n')

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

    # 汇总：按 (lr, stage1) 分组
    print('\n' + '=' * 70)
    print('grcf_deep 超参搜索汇总 (test_at_best_mae)')
    print('=' * 70)
    print(f"{'lr':<10} {'stage1':<8} {'MAE mean±std':<18} {'best':<8} {'Corr':<8}  n")
    print('-' * 70)
    best_overall = 9.0
    best_cfg = None
    for lr in LR_OPTIONS:
        for stage1 in STAGE1_OPTIONS:
            name = f"lr{lr.replace('e-', 'e')}_s1{stage1}"
            subset = [t for t in tasks if t['cfg'] == name and t.get('done')]
            if not subset:
                print(f'{lr:<10} {stage1:<8} (无完成)')
                continue
            mae_list = [t['mae'] for t in subset]
            corr_list = [t['corr'] for t in subset]
            mae_mean = sum(mae_list) / len(mae_list)
            mae_std = (sum((x - mae_mean) ** 2 for x in mae_list) / len(mae_list)) ** 0.5 if len(mae_list) > 1 else 0
            corr_mean = sum(corr_list) / len(corr_list)
            best_mae = min(mae_list)
            if best_mae < best_overall:
                best_overall = best_mae
                best_cfg = f"lr={lr}, stage1={stage1}"
            print(f'{lr:<10} {stage1:<8} {mae_mean:.4f}±{mae_std:.4f}     {best_mae:.4f}   {corr_mean:.4f}   {len(subset)}')
    print('=' * 70)
    print(f'\n最佳配置: {best_cfg}  MAE={best_overall:.4f}')


if __name__ == '__main__':
    main()
