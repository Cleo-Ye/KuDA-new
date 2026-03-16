"""
run_sims_single_best.py
========================
SIMS 单模型最佳配置搜索。双分支 pid_dualpath 模型，多配置多 seed 并行训练，
checkpoint 统一存于 ./checkpoints/SIMS，与其它数据集隔离。

目标：超越 grcf_deep seed_2024 的 MAE 0.4104。

用法：
    # 默认 3 seed × 8 配置，8 GPU 并行
    python scripts/run_sims_single_best.py --seeds 1111,42,2024 --gpus 0,1,2,3,4,5,6,7

    # 只跑指定配置
    python scripts/run_sims_single_best.py --configs grcf_deep,grcf_deep_long,grcf_aggressive

    # 干跑（仅打印命令）
    python scripts/run_sims_single_best.py --dry_run
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
RUN_DIR = "./checkpoints/SIMS"
# RoBERTa 文本特征 pkl（SIMS）
ROBERTA_DATA_PATH = "/18T/yechenlu/MSA_datasets/SIMS/Processed/unaligned_39_roberta.pkl"

BASE = dict(
    datasetName='sims',
    model_type='pid_dualpath',
    # 显式指定为 RoBERTa pkl，避免被默认 BERT 覆盖
    dataPath=ROBERTA_DATA_PATH,
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
    # 0. 对照
    dict(name='ref_dynamic', lambda_aux=0.05, lambda_ortho=0.0025, lambda_rank=0.08,
         use_dynamic_margin_rank=True, rank_margin_gamma=0.6, ranking_threshold=0.1),
    # 1. baseline 强 ranking
    dict(name='baseline_ranking_strong',
         lambda_aux=0.1, lambda_ortho=0.005, lambda_rank=0.12,
         use_dynamic_margin_rank=True, rank_margin_gamma=0.7, ranking_threshold=0.08),
    # 2. GRCF 激进
    dict(name='grcf_aggressive',
         lambda_aux=0.05, lambda_ortho=0.0025, lambda_rank=0.15,
         use_grcf=True, grcf_stage1_epochs=50, grcf_stage1_regression_weight=0.05,
         use_dynamic_margin_rank=True, rank_margin_gamma=0.7, ranking_threshold=0.08),
    # 3. GRCF 长训 80ep
    dict(name='grcf_long',
         lambda_aux=0.05, lambda_ortho=0.0025, lambda_rank=0.12,
         use_grcf=True, grcf_stage1_epochs=50, grcf_stage1_regression_weight=0.08,
         n_epochs=80, use_dynamic_margin_rank=True, rank_margin_gamma=0.7),
    # 4. GRCF 大模型（历史最佳 0.4104 的配置）
    dict(name='grcf_deep',
         lambda_aux=0.05, lambda_ortho=0.0025, lambda_rank=0.12,
         use_grcf=True, grcf_stage1_epochs=45, grcf_stage1_regression_weight=0.08,
         path_layers=3, hidden_size=512, dropout=0.35,
         use_dynamic_margin_rank=True, rank_margin_gamma=0.7),
    # 5. grcf_deep + 更长 Stage1
    dict(name='grcf_deep_stage1_55',
         lambda_aux=0.05, lambda_ortho=0.0025, lambda_rank=0.12,
         use_grcf=True, grcf_stage1_epochs=55, grcf_stage1_regression_weight=0.08,
         path_layers=3, hidden_size=512, dropout=0.35,
         use_dynamic_margin_rank=True, rank_margin_gamma=0.7),
    # 6. grcf_deep + 80ep
    dict(name='grcf_deep_long',
         lambda_aux=0.05, lambda_ortho=0.0025, lambda_rank=0.12,
         use_grcf=True, grcf_stage1_epochs=55, grcf_stage1_regression_weight=0.08,
         n_epochs=80, path_layers=3, hidden_size=512, dropout=0.35,
         use_dynamic_margin_rank=True, rank_margin_gamma=0.7),
    # 7. grcf_deep + 激进 ranking
    dict(name='grcf_deep_aggressive',
         lambda_aux=0.05, lambda_ortho=0.0025, lambda_rank=0.15,
         use_grcf=True, grcf_stage1_epochs=50, grcf_stage1_regression_weight=0.05,
         path_layers=3, hidden_size=512, dropout=0.35,
         use_dynamic_margin_rank=True, rank_margin_gamma=0.7, ranking_threshold=0.08),
    # 8. L1 损失
    dict(name='grcf_l1',
         lambda_aux=0.05, lambda_ortho=0.0025, lambda_rank=0.12,
         use_grcf=True, grcf_stage1_epochs=45, grcf_stage1_regression_weight=0.08,
         loss_fn='l1', use_dynamic_margin_rank=True, rank_margin_gamma=0.7),
    # 9. Synergy sweep 1: 更强 S 方差 + 略低 router_tau
    dict(name='grcf_deep_sweep_s1',
         lambda_aux=0.05, lambda_ortho=0.0025, lambda_rank=0.12,
         use_grcf=True, grcf_stage1_epochs=45, grcf_stage1_regression_weight=0.08,
         path_layers=3, hidden_size=512, dropout=0.35,
         use_dynamic_margin_rank=True, rank_margin_gamma=0.7,
         lambda_S_var=0.15, lambda_S_diverse=0.2, sigmoid_scale=2.5, router_tau=0.25),
    # 10. Synergy sweep 2: 更激进 S 方差 + 更尖锐路由
    dict(name='grcf_deep_sweep_s2',
         lambda_aux=0.05, lambda_ortho=0.0025, lambda_rank=0.12,
         use_grcf=True, grcf_stage1_epochs=45, grcf_stage1_regression_weight=0.08,
         path_layers=3, hidden_size=512, dropout=0.35,
         use_dynamic_margin_rank=True, rank_margin_gamma=0.7,
         lambda_S_var=0.2, lambda_S_diverse=0.25, sigmoid_scale=3.0, router_tau=0.25),
    # 11. Synergy sweep 3: 更低 router_tau，强化高/低 S 两端
    dict(name='grcf_deep_sweep_s3',
         lambda_aux=0.05, lambda_ortho=0.0025, lambda_rank=0.12,
         use_grcf=True, grcf_stage1_epochs=50, grcf_stage1_regression_weight=0.08,
         path_layers=3, hidden_size=512, dropout=0.35,
         use_dynamic_margin_rank=True, rank_margin_gamma=0.7,
         lambda_S_var=0.2, lambda_S_diverse=0.25, sigmoid_scale=3.0, router_tau=0.2),
]


def build_cmd(cfg, seed, run_dir, gpu_id):
    c = {**BASE, **cfg}
    ckpt = os.path.join(run_dir, c['name'], f"seed_{seed}")
    cmd = [
        sys.executable, 'train.py',
        '--datasetName', c['datasetName'],
        '--dataPath', c.get('dataPath', ROBERTA_DATA_PATH),
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
    ap.add_argument('--run_dir', default=RUN_DIR, help=f'checkpoint 根目录，默认 {RUN_DIR}')
    ap.add_argument('--dry_run', action='store_true')
    ap.add_argument('--verbose', action='store_true', help='显示训练日志')
    ap.add_argument('--configs', type=str, default='',
                    help='只跑指定配置，逗号分隔；空则跑全部')
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
        print(f"\n共 {len(tasks)} 个任务，run_dir={run_dir}")
        return

    t_start = time.time()
    print(f'[SIMS 单模型最佳] {len(tasks)} 任务, seeds={seeds}, gpus={gpus}')
    print(f'  run_dir={run_dir}')
    if config_filter:
        print(f'  仅跑配置: {[c["name"] for c in configs_to_run]}')
    if not args.verbose:
        print('  (加 --verbose 可查看训练日志)\n')

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
                    print(f'\n[完成-无summary] {t["name"]} (可能异常退出，加 --verbose 重跑)')
            else:
                still.append(t)
        running.clear()
        running.extend(still)

    while pending or running:
        collect()
        while pending and len(running) < max_par:
            t = pending.pop(0)
            print(f'[启动] {t["name"]}  GPU={t["gpu"]}')
            out_err = None if getattr(args, 'verbose', False) else subprocess.DEVNULL
            t['proc'] = subprocess.Popen(t['cmd'], cwd=ROOT, stdout=out_err, stderr=out_err)
            running.append(t)
        time.sleep(5)

    print('\n' + '=' * 70)
    print('SIMS 单模型汇总 (test_at_best_mae)')
    print('=' * 70)
    cfg_names = [c['name'] for c in configs_to_run]
    all_mae = []
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
        best_seed = [t['seed'] for t in subset if t['mae'] == best_mae][0]
        all_mae.append((cfg_name, best_mae, best_seed))
        print(f'  {cfg_name}:  MAE={mae_mean:.4f}±{mae_std:.4f}  Corr={corr_mean:.4f}  best={best_mae:.4f}(seed_{best_seed})  (n={len(subset)})')
        for t in subset:
            print(f'    seed_{t["seed"]}: MAE={t["mae"]:.4f}  Corr={t["corr"]:.4f}')
    print('=' * 70)
    if all_mae:
        best_cfg, best_val, best_s = min(all_mae, key=lambda x: x[1])
        print(f'\n最佳单模型: {best_cfg} seed_{best_s}  MAE={best_val:.4f}')
        print(f'  评估: python scripts/eval_sims_single.py --ckpt_dir {run_dir}/{best_cfg}/seed_{best_s} --gpu 0')
    elapsed = time.time() - t_start
    print(f'\n总耗时: {elapsed/60:.1f} 分钟')


if __name__ == '__main__':
    main()
