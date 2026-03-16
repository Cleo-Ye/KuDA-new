"""
run_aggressive_mae.py
=====================
以 MAE 最优为目标（不保守），尝试多种激进配置冲击 0.41 以下。

配置思路：
  1. baseline_ranking_strong: baseline 全量 aux/ortho + 强动态 ranking（lambda_rank=0.12, gamma=0.7）
  2. grcf_aggressive: 更长 Stage1(50ep)、更强 ranking(0.15)、更少回归干扰(reg_weight=0.05)
  3. grcf_long: 80 轮总训练，50ep ranking + 30ep 校准
  4. grcf_deep: path_layers=3, hidden_size=512（论文 CH-SIMS Table 2 更大容量）

用法：
    python scripts/run_aggressive_mae.py --seeds 1111,42,2024 --gpus 0,1,2,3,4,5 --run_dir ./checkpoints/aggressive_mae
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
    hidden_size=256,
)

CONFIGS = [
    # 0. 对照：当前最佳稳定配置
    dict(name='ref_dynamic', lambda_aux=0.05, lambda_ortho=0.0025, lambda_rank=0.08,
         use_dynamic_margin_rank=True, rank_margin_gamma=0.6, ranking_threshold=0.1),
    # 1. baseline + 强动态 ranking（保持全量 aux/ortho）
    dict(name='baseline_ranking_strong',
         lambda_aux=0.1, lambda_ortho=0.005, lambda_rank=0.12,
         use_dynamic_margin_rank=True, rank_margin_gamma=0.7, ranking_threshold=0.08),
    # 2. GRCF 激进：50ep ranking，lambda_rank=0.15，reg_weight=0.05
    dict(name='grcf_aggressive',
         lambda_aux=0.05, lambda_ortho=0.0025, lambda_rank=0.15,
         use_grcf=True, grcf_stage1_epochs=50, grcf_stage1_regression_weight=0.05,
         use_dynamic_margin_rank=True, rank_margin_gamma=0.7, ranking_threshold=0.08),
    # 3. GRCF 长训：80 轮，50 ranking + 30 校准
    dict(name='grcf_long',
         lambda_aux=0.05, lambda_ortho=0.0025, lambda_rank=0.12,
         use_grcf=True, grcf_stage1_epochs=50, grcf_stage1_regression_weight=0.08,
         n_epochs=80, use_dynamic_margin_rank=True, rank_margin_gamma=0.7),
    # 4. GRCF 大模型：path_layers=3, hidden_size=512
    dict(name='grcf_deep',
         lambda_aux=0.05, lambda_ortho=0.0025, lambda_rank=0.12,
         use_grcf=True, grcf_stage1_epochs=45, grcf_stage1_regression_weight=0.08,
         path_layers=3, hidden_size=512, dropout=0.35,
         use_dynamic_margin_rank=True, rank_margin_gamma=0.7),
    # 5. GRCF + L1 损失（Stage2 用 MAE 更直接）
    dict(name='grcf_l1',
         lambda_aux=0.05, lambda_ortho=0.0025, lambda_rank=0.12,
         use_grcf=True, grcf_stage1_epochs=45, grcf_stage1_regression_weight=0.08,
         loss_fn='l1', use_dynamic_margin_rank=True, rank_margin_gamma=0.7),
    # 6. grcf_deep_long: grcf_deep 架构 + 80ep（55 ranking + 25 calibration）
    dict(name='grcf_deep_long',
         lambda_aux=0.05, lambda_ortho=0.0025, lambda_rank=0.12,
         use_grcf=True, grcf_stage1_epochs=55, grcf_stage1_regression_weight=0.08,
         n_epochs=80, path_layers=3, hidden_size=512, dropout=0.35,
         use_dynamic_margin_rank=True, rank_margin_gamma=0.7),
]


def build_cmd(cfg, seed, run_dir, gpu_id):
    c = {**BASE, **cfg}
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
        '--loss_fn', str(c.get('loss_fn', 'smoothl1')),
        '--path_layers', str(c['path_layers']),
        '--hidden_size', str(c.get('hidden_size', 256)),
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
    ap.add_argument('--seeds', default='1111,42,2024')
    ap.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    ap.add_argument('--run_dir', default='./checkpoints/aggressive_mae')
    ap.add_argument('--dry_run', action='store_true')
    ap.add_argument('--verbose', action='store_true',
                    help='显示各任务的训练日志（默认隐藏，因 8 任务并行会混在一起）')
    ap.add_argument('--configs', type=str, default='',
                    help='只跑指定配置，逗号分隔，如 grcf_deep,grcf_deep_long；空则跑全部')
    args = ap.parse_args()

    seeds = [s.strip() for s in args.seeds.split(',') if s.strip()]
    gpus = [s.strip() for s in args.gpus.split(',')]
    run_dir = os.path.abspath(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)

    config_filter = [c.strip() for c in args.configs.split(',') if c.strip()] if getattr(args, 'configs', '') else []
    configs_to_run = [c for c in CONFIGS if c['name'] in config_filter] if config_filter else CONFIGS

    tasks = []
    for cfg in configs_to_run:
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

    t_start = time.time()
    print(f'[aggressive MAE] {len(tasks)} 任务, seeds={seeds}, gpus={gpus}')
    if config_filter:
        print(f'  仅跑配置: {[c["name"] for c in configs_to_run]}')
    if not args.verbose:
        print('  (训练日志已隐藏，加 --verbose 可查看；每个任务约 20–60 分钟)')
    print('  0. ref_dynamic: 对照 (ablation_half_ranking_dynamic)')
    print('  1. baseline_ranking_strong: 全量 aux/ortho + lambda_rank=0.12, gamma=0.7')
    print('  2. grcf_aggressive: 50ep ranking, lambda_rank=0.15, reg_weight=0.05')
    print('  3. grcf_long: 80ep (50 ranking + 30 calibration)')
    print('  4. grcf_deep: path_layers=3, hidden_size=512')
    print('  5. grcf_l1: loss_fn=L1 (Stage2 MAE 更直接)')
    print('  6. grcf_deep_long: grcf_deep + 80ep (55 ranking + 25 calibration)\n')

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
                    print(f'\n[完成-无summary] {t["name"]} (可能训练异常退出，可用 --verbose 重跑查看日志)')
            else:
                still.append(t)
        running.clear()
        running.extend(still)

    while pending or running:
        collect()
        while pending and len(running) < max_par:
            t = pending.pop(0)
            print(f'[启动] {t["name"]}  GPU={t["gpu"]}')
            # --verbose 时显示训练日志；默认隐藏（8 任务并行时混在一起难以阅读）
            out_err = None if getattr(args, 'verbose', False) else subprocess.DEVNULL
            t['proc'] = subprocess.Popen(t['cmd'], cwd=ROOT, stdout=out_err, stderr=out_err)
            running.append(t)
        time.sleep(5)

    print('\n' + '=' * 70)
    print('Aggressive MAE 汇总 (test_at_best_mae)')
    print('=' * 70)
    cfg_names = [c['name'] for c in configs_to_run]
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
        best_mae = min(mae_list)
        print(f'  {cfg_name}:  MAE={mae_mean:.4f}±{mae_std:.4f}  Corr={corr_mean:.4f}  best={best_mae:.4f}  (n={len(subset)})')
        for t in subset:
            print(f'    seed_{t["seed"]}: MAE={t["mae"]:.4f}  Corr={t["corr"]:.4f}')
    print('=' * 70)
    elapsed = time.time() - t_start
    print(f'\n总耗时: {elapsed/60:.1f} 分钟 ({elapsed/3600:.2f} 小时)')


if __name__ == '__main__':
    main()
