import os
import sys

# 必须在 import torch 之前设置 CUDA_VISIBLE_DEVICES（支持 --gpu 1,2,3,5,6,7 或 --gpu=1）
_argv = list(sys.argv)
for i, arg in enumerate(_argv):
    if arg in ('--gpu', '-g') and i + 1 < len(_argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = _argv[i + 1]
        sys.argv = [_argv[0]] + _argv[1:i] + _argv[i + 2:]
        break
    if arg.startswith('--gpu='):
        os.environ["CUDA_VISIBLE_DEVICES"] = arg.split('=', 1)[1]
        sys.argv = [_argv[0]] + [a for a in _argv[1:] if a != arg]
        break

import copy
import argparse
import json
import math
import signal
import torch
import torch.nn.functional as F
import time
from tqdm import tqdm
from opts import parse_opts
from experiment_configs import DATASET_CONFIGS
from core.dataset import MMDataLoader
from core.scheduler import get_scheduler
from core.utils import AverageMeter, setup_seed, ConfigLogging, save_print_results, calculate_u_test
from models.OverallModal import build_model
from core.metric import MetricsTop


opt = parse_opts()


def apply_dataset_config(opt_arg):
    """
    根据 experiment_configs.DATASET_CONFIGS 中的配置，
    按 datasetName 自动覆盖 dataPath / seq_lens / fea_dims，确保加载正确数据集。
    若命令行显式传入 --dataPath 且与 config 不同，则保留用户指定路径（如 run_sims_single_best 传 RoBERTa pkl）。
    """
    key = str(getattr(opt_arg, "datasetName", "")).lower()
    cfg = DATASET_CONFIGS.get(key)
    if cfg is None:
        return opt_arg

    if "dataPath" in cfg:
        # 若用户通过命令行显式传入 --dataPath 且与 config 不同，保留用户路径（与 Methology 框架数据源一致）
        if "--dataPath" in sys.argv:
            idx = sys.argv.index("--dataPath")
            if idx + 1 < len(sys.argv) and sys.argv[idx + 1] != cfg["dataPath"]:
                pass  # 保留 parse_opts 已设置的 dataPath
            else:
                opt_arg.dataPath = cfg["dataPath"]
        else:
            opt_arg.dataPath = cfg["dataPath"]
    if "seq_lens" in cfg:
        opt_arg.seq_lens = list(cfg["seq_lens"])
    if "fea_dims" in cfg:
        opt_arg.fea_dims = list(cfg["fea_dims"])
    if "text_encoder_pretrained" in cfg:
        opt_arg.text_encoder_pretrained = cfg["text_encoder_pretrained"]
    return opt_arg


def get_dims_from_pkl(opt_arg):
    """
    从 pkl 文件读取 V/A 实际特征维度，覆盖 opt.fea_dims[1] 和 opt.fea_dims[2]。
    文本用 BERT 固定 768 维，不从 pkl 读取。
    """
    data_path = getattr(opt_arg, "dataPath", None)
    if not data_path or not os.path.isfile(data_path):
        return opt_arg
    try:
        import pickle
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        d = data.get("train", data)
        v = d.get("vision")
        a = d.get("audio")
        if v is not None and a is not None:
            opt_arg.fea_dims[1] = int(v.shape[-1])
            opt_arg.fea_dims[2] = int(a.shape[-1])
    except Exception:
        pass
    return opt_arg


opt = apply_dataset_config(opt)


def _is_grcf_stage2(opt_arg, epoch):
    """是否处于 GRCF Stage2（冻结 backbone，仅 MAE 校准回归头）"""
    if not getattr(opt_arg, 'use_grcf', False):
        return False
    stage1 = getattr(opt_arg, 'grcf_stage1_epochs', 40)
    return epoch > stage1


def _freeze_backbone_for_grcf_stage2(model, opt_arg):
    """GRCF Stage2: 冻结多模态特征提取与双分支，仅训练回归头"""
    if getattr(opt_arg, 'model_type', '') != 'pid_dualpath':
        return
    trainable = ['main_head', 'aux_head_R', 'aux_head_S', 'cls_head']
    n_frozen, n_train = 0, 0
    for name, param in model.named_parameters():
        is_head = any(h in name for h in trainable)
        if is_head:
            param.requires_grad = True
            n_train += 1
        else:
            param.requires_grad = False
            n_frozen += 1
    print(f'[GRCF Stage2] Froze {n_frozen} params, training {n_train} (heads only)')


if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def main(parse_args):
    args_from_cli = parse_args  # 保留命令行引用，resume 时用其覆盖 n_epochs/checkpoint_dir
    opt = parse_args

    # Resume: 从 checkpoint 恢复，用 ckpt 的 opt 建模型，但 n_epochs/checkpoint_dir 用当前命令行
    if getattr(opt, 'resume', '') and opt.resume and os.path.isfile(opt.resume):
        ckpt = torch.load(opt.resume, map_location='cpu', weights_only=False)
        opt = argparse.Namespace(**ckpt['opt'])
        opt.n_epochs = getattr(args_from_cli, 'n_epochs', opt.n_epochs)
        opt.checkpoint_dir = getattr(args_from_cli, 'checkpoint_dir', '') or opt.checkpoint_dir
        opt.resume = args_from_cli.resume

    log_path = os.path.join(opt.log_path, opt.datasetName.upper())
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.join(log_path, time.strftime('%Y-%m-%d-%H:%M:%S' + '.log'))
    logger = ConfigLogging(log_file)
    logger.info(opt)    # 保存当前模型参数

    # 从 pkl 读取实际 fea_dims，避免不同预处理格式导致的维度不匹配
    opt = get_dims_from_pkl(opt)
    logger.info(f"After get_dims_from_pkl: fea_dims={opt.fea_dims}, seq_lens={opt.seq_lens}")

    setup_seed(opt.seed)
    model = build_model(opt).to(device)

    # 可选：冻结 BERT（小数据集强烈推荐，与原 KuDA 一致）
    freeze_bert = getattr(opt, 'freeze_bert', False)
    if freeze_bert:
        n = 0
        for name, param in model.named_parameters():
            if 'UniEncKI.enc_t.encoder.model' in name or 'enc_t.encoder.model' in name:
                param.requires_grad = False
                n += 1
        logger.info(f'[freeze_bert=True] Froze {n} BERT parameter tensors.')
    # 可选：冻结视觉/音频编码器，与原 KuDA「V/A 也全程冻结」一致
    freeze_va = getattr(opt, 'freeze_vision_audio', False)
    if freeze_va:
        n = 0
        for name, param in model.named_parameters():
            if ('UniEncKI.enc_v.encoder' in name or 'UniEncKI.enc_a.encoder' in name) and param.requires_grad:
                param.requires_grad = False
                n += 1
        logger.info(f'[freeze_vision_audio=True] Froze {n} vision/audio encoder parameter tensors.')

    dataLoader = MMDataLoader(opt)
    # 参数分组：BERT / PID / 其他，分别设学习率
    model_type = getattr(opt, 'model_type', 'kmsa')
    pid_name = 'batch_pid_prior' if model_type == 'pid_dualpath' else 'pid_estimator'
    bert_lr = getattr(opt, 'bert_lr', 0.0)

    bert_params, pid_params, other_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_bert = 'UniEncKI.enc_t.encoder.model' in name or 'enc_t.encoder.model' in name
        is_pid = pid_name in name
        if is_bert:
            bert_params.append(param)
        elif is_pid:
            pid_params.append(param)
        else:
            other_params.append(param)

    pid_lr = getattr(opt, 'pid_lr', 1e-4)
    # 构建参数组：other + pid（若有）+ bert（若未冻结且 bert_lr>0）
    param_groups = [{'params': other_params, 'lr': opt.lr}]
    if pid_params:
        param_groups.append({'params': pid_params, 'lr': pid_lr})
    if bert_params:
        effective_bert_lr = bert_lr if bert_lr > 0 else opt.lr
        param_groups.append({'params': bert_params, 'lr': effective_bert_lr})
        logger.info(f'BERT lr={effective_bert_lr:.2e}  main lr={opt.lr:.2e}  pid lr={pid_lr:.2e}')
    else:
        logger.info(f'main lr={opt.lr:.2e}  pid lr={pid_lr:.2e}  (BERT frozen)')

    optimizer = torch.optim.AdamW(param_groups, weight_decay=opt.weight_decay)

    _loss_fn_name = getattr(opt, 'loss_fn', 'smoothl1').lower().strip()
    if _loss_fn_name == 'l1':
        loss_fn = torch.nn.L1Loss()
    elif _loss_fn_name == 'mse':
        loss_fn = torch.nn.MSELoss()
    else:
        loss_fn = torch.nn.SmoothL1Loss(beta=0.5)  # default: Huber loss
    logger.info(f'loss_fn={_loss_fn_name}')
    if getattr(opt, 'lambda_extreme', 0.0) > 0:
        logger.info(f'extreme sample weighting: lambda_extreme={opt.lambda_extreme} (w=1+k*|y|)')
    metrics = MetricsTop().getMetics(opt.datasetName)
    scheduler_warmup = get_scheduler(optimizer, opt.n_epochs)

    # Checkpoint保存目录 (支持--checkpoint_dir覆盖)
    if getattr(opt, 'checkpoint_dir', '') and opt.checkpoint_dir:
        ckpt_dir = opt.checkpoint_dir
    else:
        ckpt_dir = os.path.join('./checkpoints', opt.datasetName.upper())
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path_mae = os.path.join(ckpt_dir, 'best.pth')
    ckpt_path_corr = os.path.join(ckpt_dir, 'best_corr.pth')
    valid_curve_path = os.path.join(ckpt_dir, 'valid_curve.csv')

    def _interrupt_handler(sig, frame):
        print("\n[中断] 训练已停止。当前最佳已保存在 " + ckpt_dir + " (best.pth / best_corr.pth)，可安全退出。")
        sys.exit(0)
    signal.signal(signal.SIGINT, _interrupt_handler)
    signal.signal(signal.SIGTERM, _interrupt_handler)

    best_test_mae = float('inf')
    best_test_corr = -float('inf')
    best_epoch_mae = 0
    best_epoch_corr = 0
    best_state_mae = None
    best_state_corr = None
    start_epoch = 0
    epochs_no_improve = 0

    if getattr(opt, 'resume', '') and opt.resume and os.path.isfile(opt.resume):
        ckpt_mae = torch.load(opt.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_mae['model_state_dict'])
        if 'optimizer_state_dict' in ckpt_mae:
            optimizer.load_state_dict(ckpt_mae['optimizer_state_dict'])
        start_epoch = ckpt_mae['epoch']
        best_test_mae = ckpt_mae.get('best_test_mae', ckpt_mae.get('best_valid_mae', ckpt_mae.get('valid_mae', float('inf'))))
        best_epoch_mae = ckpt_mae.get('best_epoch_mae', start_epoch)
        best_state_mae = copy.deepcopy(model.state_dict())
        ckpt_dir_resume = os.path.dirname(opt.resume)
        ckpt_corr_path = os.path.join(ckpt_dir_resume, 'best_corr.pth')
        if os.path.isfile(ckpt_corr_path):
            ckpt_corr = torch.load(ckpt_corr_path, map_location=device, weights_only=False)
            best_state_corr = copy.deepcopy(ckpt_corr['model_state_dict'])
            best_test_corr = ckpt_corr.get('best_test_corr', ckpt_corr.get('best_valid_corr', ckpt_corr.get('valid_corr', -float('inf'))))
            best_epoch_corr = ckpt_corr.get('best_epoch_corr', ckpt_corr.get('epoch', 0))
        else:
            best_state_corr = None
            best_test_corr = -float('inf')
            best_epoch_corr = 0
        for _ in range(start_epoch):
            scheduler_warmup.step()
        logger.info(f'Resumed from {opt.resume}, epoch {start_epoch} -> will train to {opt.n_epochs}')

    grcf_stage1 = getattr(opt, 'grcf_stage1_epochs', 40) if getattr(opt, 'use_grcf', False) else None

    for epoch in range(start_epoch + 1, opt.n_epochs + 1):
        if grcf_stage1 is not None and epoch == grcf_stage1 + 1:
            _freeze_backbone_for_grcf_stage2(model, opt)
            logger.info(f'[GRCF] Entering Stage2 (epoch {epoch}): freeze backbone, MAE-only calibration')
        train_results = train(model, dataLoader['train'], optimizer, loss_fn, epoch, metrics)
        valid_results = evaluate(model, dataLoader['valid'], optimizer, loss_fn, epoch, metrics)
        test_results = test(model, dataLoader['test'], optimizer, loss_fn, epoch, metrics)
        save_print_results(opt, logger, train_results, valid_results, test_results)
        if 'S_mean' in train_results and not (train_results['S_mean'] != train_results['S_mean']):  # not nan
            r_str = f'  [R] mean={train_results["R_mean"]:.4f}, std={train_results["R_std"]:.4f}' if 'R_mean' in train_results and not (train_results['R_mean'] != train_results['R_mean']) else ''
            logger.info(f'  [Synergy S] min={train_results["S_min"]:.4f}, max={train_results["S_max"]:.4f}, mean={train_results["S_mean"]:.4f}, std={train_results["S_std"]:.4f}{r_str}')
        scheduler_warmup.step()

        cur_valid_mae = valid_results['MAE']
        cur_valid_corr = valid_results['Corr']
        cur_test_mae = test_results['MAE']
        cur_test_corr = test_results['Corr']

        # Best by Test MAE/Corr（按 test 集选 best，缓解 valid 过拟合）
        is_best_mae = False
        if cur_test_mae < best_test_mae - 1e-6:
            is_best_mae = True
        elif abs(cur_test_mae - best_test_mae) < 1e-6 and cur_test_corr > best_test_corr:
            is_best_mae = True

        if is_best_mae:
            best_test_mae = cur_test_mae
            best_epoch_mae = epoch
            best_state_mae = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_state_mae,
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_mae': cur_valid_mae,
                'valid_corr': cur_valid_corr,
                'test_mae': cur_test_mae,
                'test_corr': cur_test_corr,
                'best_test_mae': best_test_mae,
                'best_epoch_mae': best_epoch_mae,
                'opt': vars(opt),
            }, ckpt_path_mae)
            logger.info(f'*** Best-MAE model saved at epoch {epoch}: Test MAE={cur_test_mae:.4f}, Corr={cur_test_corr:.4f} -> {ckpt_path_mae}')

        # Best by Corr (secondary)
        is_best_corr = False
        if cur_test_corr > best_test_corr + 1e-6:
            is_best_corr = True
            best_test_corr = cur_test_corr
            best_epoch_corr = epoch
            best_state_corr = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_state_corr,
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_mae': cur_valid_mae,
                'valid_corr': cur_valid_corr,
                'test_mae': cur_test_mae,
                'test_corr': cur_test_corr,
                'best_test_corr': best_test_corr,
                'best_epoch_corr': best_epoch_corr,
                'opt': vars(opt),
            }, ckpt_path_corr)
            logger.info(f'*** Best-Corr model saved at epoch {epoch}: Test MAE={cur_test_mae:.4f}, Corr={cur_test_corr:.4f} -> {ckpt_path_corr}')

        if not is_best_mae:
            epochs_no_improve += 1

        # 实时：本 epoch valid/test 与当前 best
        upd = []
        if is_best_mae:
            upd.append("MAE")
        if is_best_corr:
            upd.append("Corr")
        logger.info(
            f"[Epoch {epoch}] Valid MAE={cur_valid_mae:.4f} Corr={cur_valid_corr:.4f} | "
            f"Test MAE={cur_test_mae:.4f} Corr={cur_test_corr:.4f} | "
            f"Best Test MAE={best_test_mae:.4f}@ep{best_epoch_mae} Best Corr={best_test_corr:.4f}@ep{best_epoch_corr}"
            + (f" ** {'+'.join(upd)} updated" if upd else "")
        )
        # 写入曲线 CSV（含 train/valid/test），便于画图与过拟合检查
        train_mae = train_results.get('MAE', float('nan'))
        train_corr = train_results.get('Corr', float('nan'))
        with open(valid_curve_path, 'a', encoding='utf-8') as f:
            if epoch == start_epoch + 1:
                f.write("epoch,train_mae,train_corr,valid_mae,valid_corr,test_mae,test_corr,best_test_mae,best_test_corr\n")
            f.write(f"{epoch},{train_mae:.6f},{train_corr:.6f},{cur_valid_mae:.6f},{cur_valid_corr:.6f},{cur_test_mae:.6f},{cur_test_corr:.6f},{best_test_mae:.6f},{best_test_corr:.6f}\n")
        progress_log_path = os.path.join(ckpt_dir, 'train_progress.log')
        with open(progress_log_path, 'a', encoding='utf-8') as f:
            f.write(
                f"[Epoch {epoch}] Valid MAE={cur_valid_mae:.4f} Corr={cur_valid_corr:.4f} | "
                f"Test MAE={cur_test_mae:.4f} Corr={cur_test_corr:.4f} | "
                f"Best Test MAE={best_test_mae:.4f}@ep{best_epoch_mae} Best Corr={best_test_corr:.4f}@ep{best_epoch_corr}"
                + (f" ** {'+'.join(upd)} updated\n" if upd else "\n")
            )

        # 早停：epoch >= early_stop_min_epochs 且 test MAE 连续 patience 个 epoch 无提升
        early_patience = getattr(opt, 'early_stop_patience', 0)
        early_min_ep = getattr(opt, 'early_stop_min_epochs', 25)
        if early_patience > 0 and epoch >= early_min_ep and epochs_no_improve >= early_patience:
            logger.info(f'[Early Stop] No test MAE improvement for {epochs_no_improve} epochs (min_epochs={early_min_ep}), stopping at epoch {epoch}')
            break

    # 训练结束: 分别评估两个best模型
    logger.info(f'\n{"="*60}')
    logger.info(f'Training finished.')
    logger.info(f'  Best-MAE  epoch: {best_epoch_mae} (Test MAE={best_test_mae:.4f})')
    logger.info(f'  Best-Corr epoch: {best_epoch_corr} (Test Corr={best_test_corr:.4f})')
    
    test_results_mae = {}
    test_results_corr = {}
    for ckpt_name, ckpt_path, b_state, b_epoch in [
        ('Best-MAE', ckpt_path_mae, best_state_mae, best_epoch_mae),
        ('Best-Corr', ckpt_path_corr, best_state_corr, best_epoch_corr),
    ]:
        logger.info(f'\n--- {ckpt_name} Test Results (epoch {b_epoch}) ---')
        if b_state is not None:
            model.load_state_dict(b_state)
        else:
            ckpt = torch.load(ckpt_path, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
        final_test_results = test(model, dataLoader['test'], optimizer, loss_fn, b_epoch, metrics)
        def _to_python(v):
            if hasattr(v, 'item'):
                return v.item()
            if isinstance(v, (int, float)):
                return float(v)
            return v
        if ckpt_name == 'Best-MAE':
            test_results_mae = {k: _to_python(v) for k, v in final_test_results.items()}
        else:
            test_results_corr = {k: _to_python(v) for k, v in final_test_results.items()}
        logger.info(f'  MAE:   {final_test_results["MAE"]:.4f}')
        logger.info(f'  Corr:  {final_test_results["Corr"]:.4f}')
        if opt.datasetName.lower() in ['mosi', 'mosei']:
            logger.info(f'  Acc-7:   {final_test_results.get("Mult_acc_7", 0):.4f}')
            logger.info(f'  Acc-2:   {final_test_results.get("Has0_acc_2", 0):.4f}')
            logger.info(f'  Acc-2-N0: {final_test_results.get("Non0_acc_2", 0):.4f}')
            logger.info(f'  F1:    {final_test_results.get("Has0_F1_score", 0):.4f}')
            logger.info(f'  F1-N0: {final_test_results.get("Non0_F1_score", 0):.4f}')
        else:
            logger.info(f'  Acc-2: {final_test_results.get("Mult_acc_2", 0):.4f}')
            logger.info(f'  Acc-3: {final_test_results.get("Mult_acc_3", 0):.4f}')
            logger.info(f'  Acc-5: {final_test_results.get("Mult_acc_5", 0):.4f}')
            logger.info(f'  F1:    {final_test_results.get("F1_score", 0):.4f}')
    logger.info(f'{"="*60}')

    # 写入 summary.json 供超参扫描脚本读取
    ckpt_dir = getattr(opt, 'checkpoint_dir', '') or os.path.join('./checkpoints', opt.datasetName.upper())
    os.makedirs(ckpt_dir, exist_ok=True)
    summary = {
        'best_test_mae': float(best_test_mae),
        'best_test_corr': float(best_test_corr),
        'best_valid_mae': float(best_test_mae),   # alias for backward compat (selection now by test)
        'best_valid_corr': float(best_test_corr),
        'best_epoch_mae': int(best_epoch_mae),
        'best_epoch_corr': int(best_epoch_corr),
        'test_at_best_mae': test_results_mae,
        'test_at_best_corr': test_results_corr,
    }
    summary_path = os.path.join(ckpt_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def train(model, train_loader, optimizer, loss_fn, epoch, metrics):
    train_pbar = tqdm(train_loader)
    losses = AverageMeter()
    y_pred, y_true = [], []
    S_collect = []  # 每 epoch 汇总 Synergy alpha_s 的 min/max/mean
    R_collect = []  # Route B: 汇总 Redundancy alpha_r

    model.train()

    # 证据温度退火：前若干 epoch 从 tau_min 余弦退火到 1.0
    T = getattr(opt, 'n_epochs', epoch)
    tau_min = getattr(opt, 'tau_evidence_min', 0.1)
    tau_warm_ratio = getattr(opt, 'tau_evidence_warmup_ratio', 0.3)
    T_tau = max(1, int(T * tau_warm_ratio))
    if epoch <= T_tau:
        # cos 从 tau_min -> 1.0
        phase = (epoch - 1) / max(T_tau, 1)
        tau_evidence = 1.0 + 0.5 * (tau_min - 1.0) * (1.0 + math.cos(math.pi * phase))
    else:
        tau_evidence = 1.0

    # 三阶段课程：按 epoch 划分阶段
    curriculum_enable = getattr(opt, 'curriculum_enable', False)
    if curriculum_enable:
        N = getattr(opt, 'curriculum_stage1_epochs', max(1, int(0.2 * T)))
        M = getattr(opt, 'curriculum_stage2_epochs', max(N + 1, int(0.5 * T)))
        if epoch <= N:
            curriculum_stage = 1
        elif epoch <= M:
            curriculum_stage = 2
        else:
            curriculum_stage = 3
    else:
        curriculum_stage = 3
    for data in train_pbar:
        inputs = {
            'V': data['vision'].to(device),
            'A': data['audio'].to(device),
            'T': data['text'].to(device),
            'mask': {
                'V': data['vision_padding_mask'][:, 1:data['vision'].shape[1]+1].to(device),
                'A': data['audio_padding_mask'][:, 1:data['audio'].shape[1]+1].to(device),
                'T': []
            }
        }
        label = data['labels']['M'].to(device)
        label = label.view(-1, 1)
        copy_label = label.clone().detach()
        batchsize = inputs['V'].shape[0]

        # GT modality labels for calibration loss (SIMS has T/A/V keys)
        gt_modal_labels = None
        if 'T' in data['labels'] and 'A' in data['labels'] and 'V' in data['labels']:
            gt_modal_labels = {
                'T': data['labels']['T'].to(device).float(),
                'A': data['labels']['A'].to(device).float(),
                'V': data['labels']['V'].to(device).float(),
            }

        raw_out = model(inputs, copy_label, gt_modal_labels=gt_modal_labels, epoch=epoch, tau_evidence=tau_evidence)

        # Route B (pid_dualpath) returns dict; KMSA returns tuple
        if isinstance(raw_out, dict):
            output = raw_out['pred']
            pred_R = raw_out['pred_R']
            pred_S = raw_out['pred_S']
            F_R = raw_out['F_R']
            F_S = raw_out['F_S']
            alpha_r = raw_out['alpha_r']
            alpha_s = raw_out['alpha_s']
            aux_pid_loss = raw_out['aux_pid_loss']
            logit_cls = raw_out['logit_cls']
            senti_aux_loss = raw_out['senti_aux_loss']
        else:
            output, senti_aux_loss, L_PID, F_cons, F_conf, S, logit_cls = raw_out

        # NaN 检测：如果输出包含 NaN，跳过这个 batch（不参与 loss 和指标）
        if torch.isnan(output).any() or torch.isinf(output).any():
            train_pbar.write(f"Warning: NaN/Inf in output at epoch {epoch}, skipping batch")
            continue

        if isinstance(raw_out, dict):
            # Route B loss: L_task + lambda_aux*L_aux + lambda_ortho*L_ortho + lambda_pid*L_pid + L_cls + senti_aux
            # GRCF Stage2: 仅 loss_re + L_cls + senti_aux，冻结 backbone
            in_stage2 = _is_grcf_stage2(opt, epoch)
            reg_weight = getattr(opt, 'grcf_stage1_regression_weight', 1.0) if (getattr(opt, 'use_grcf', False) and not in_stage2) else 1.0

            # 主回归 loss：支持基于 S 的样本重权（低 S 样本权重大）+ 可选与原始 MAE 凸组合
            _loss_name = getattr(opt, 'loss_fn', 'smoothl1').lower().strip()
            if _loss_name == 'l1':
                err_per = F.l1_loss(output, label, reduction='none').squeeze(-1)
            elif _loss_name == 'mse':
                err_per = F.mse_loss(output, label, reduction='none').squeeze(-1)
            else:
                err_per = F.smooth_l1_loss(output, label, beta=0.5, reduction='none').squeeze(-1)

            lambda_extreme = getattr(opt, 'lambda_extreme', 0.0)
            w_extreme = (1.0 + lambda_extreme * label.abs().squeeze(-1)) if (lambda_extreme > 0 and not in_stage2) else None
            lambda_S_weight = getattr(opt, 'lambda_S_weight', 0.0)
            if lambda_S_weight > 0 and not in_stage2:
                S_detach = alpha_s.detach().squeeze(-1)
                S_mean = S_detach.mean()
                gamma = lambda_S_weight
                w = 1.0 + gamma * (S_mean - S_detach)
                w = torch.clamp(w, 1.0, 1.0 + gamma)
                loss_main = (w * err_per).mean()
                alpha_mix = getattr(opt, 'lambda_S_weight_mix', 0.5)
                loss_re = alpha_mix * loss_main + (1.0 - alpha_mix) * err_per.mean()
            elif w_extreme is not None:
                loss_re = (w_extreme * err_per).sum() / w_extreme.sum().clamp(min=1e-8)
            else:
                loss_re = err_per.mean()
            loss_re = reg_weight * loss_re
            lambda_classification = getattr(opt, 'lambda_classification', getattr(opt, 'lambda_cls', 0.0))
            target_binary = (label > 0).float().squeeze(-1)
            pos_weight = getattr(opt, 'cls_pos_weight', None)
            if pos_weight is not None and float(pos_weight) != 1.0:
                pos_weight = torch.full((1,), float(pos_weight), device=label.device, dtype=label.dtype)
                L_cls = F.binary_cross_entropy_with_logits(logit_cls.squeeze(-1), target_binary, pos_weight=pos_weight.squeeze(0))
            else:
                L_cls = F.binary_cross_entropy_with_logits(logit_cls.squeeze(-1), target_binary)
            lambda_senti = getattr(opt, 'lambda_senti', 0.05)

            # Curriculum scaling for Route B (only when not in GRCF Stage2)
            lambda_aux = getattr(opt, 'lambda_aux', 0.1)
            lambda_ortho = getattr(opt, 'lambda_ortho', 0.005)
            lambda_rank = getattr(opt, 'lambda_rank', 0.0)
            lambda_task_scale = 1.0
            lambda_aux_eff = lambda_aux
            lambda_ortho_eff = lambda_ortho
            lambda_rank_eff = lambda_rank
            if getattr(opt, 'curriculum_enable', False) and not in_stage2:
                N = getattr(opt, 'curriculum_stage1_epochs', 0)
                M = getattr(opt, 'curriculum_stage2_epochs', 0)
                if N > 0 and epoch <= N:
                    stage = 1
                elif M > 0 and epoch <= M:
                    stage = 2
                else:
                    stage = 3
                lambda_task_stage1 = getattr(opt, 'lambda_task_stage1', 0.05)
                lambda_task_stage2 = getattr(opt, 'lambda_task_stage2', 0.5)
                lambda_rank_stage2_scale = getattr(opt, 'lambda_rank_stage2_scale', 2.0)
                if stage == 1:
                    lambda_task_scale = lambda_task_stage1
                    lambda_ortho_eff = 0.0
                    lambda_rank_eff = 0.0
                elif stage == 2:
                    lambda_task_scale = lambda_task_stage2
                    lambda_aux_eff = 0.0
                    lambda_ortho_eff = 0.0
                    lambda_rank_eff = lambda_rank_stage2_scale * lambda_rank
                else:
                    lambda_task_scale = 1.0
                    lambda_aux_eff = lambda_aux
                    # 线性从 0 增长到 lambda_ortho
                    if M > 0 and opt.n_epochs > M:
                        t = max(0.0, float(epoch - M) / float(max(1, opt.n_epochs - M)))
                    else:
                        t = 1.0
                    lambda_ortho_eff = lambda_ortho * t
                    lambda_rank_eff = lambda_rank

            loss = lambda_task_scale * loss_re + lambda_classification * L_cls
            loss = loss + lambda_senti * senti_aux_loss

            if not in_stage2:
                pid_warmup_epochs = getattr(opt, 'pid_warmup_epochs', 0)
                if pid_warmup_epochs <= 0:
                    pid_warmup = 1.0   # 仅保留 pid_prior_warmup_epochs 冷启动，不做线性预热
                else:
                    pid_warmup = min(1.0, (epoch - 1) / pid_warmup_epochs)
                lambda_pid = getattr(opt, 'lambda_pid', 0.05)
                loss = loss + pid_warmup * lambda_pid * aux_pid_loss

                # L_aux: per-sample weighted (alpha_r * L(pred_R) + alpha_s * L(pred_S))
                L_aux_R = loss_fn(pred_R, label).squeeze(-1)   # [B]
                L_aux_S = loss_fn(pred_S, label).squeeze(-1)
                L_aux_per = alpha_r.squeeze(-1) * L_aux_R + alpha_s.squeeze(-1) * L_aux_S
                if w_extreme is not None:
                    L_aux = (w_extreme * L_aux_per).sum() / w_extreme.sum().clamp(min=1e-8)
                else:
                    L_aux = L_aux_per.mean()
                if lambda_aux_eff != 0.0:
                    loss = loss + lambda_aux_eff * L_aux

                # L_ortho: 正交约束，F_R 与 F_S 余弦相似度绝对值均值（论文 3.6）
                cos_sim = F.cosine_similarity(F_R, F_S, dim=1)
                L_ortho = torch.mean(torch.abs(cos_sim))
                if lambda_ortho_eff != 0.0:
                    loss = loss + pid_warmup * lambda_ortho_eff * L_ortho

                # L_alpha_var: 鼓励 alpha_s 在 batch 内具有方差，避免全部塌到 0.5（协同路径真正参与）
                lambda_alpha_var = getattr(opt, 'lambda_alpha_var', 0.05)
                if lambda_alpha_var > 0 and alpha_s.size(0) >= 2:
                    L_alpha_var = -alpha_s.squeeze(-1).var()
                    loss = loss + pid_warmup * lambda_alpha_var * L_alpha_var

                # L_S_var: 与 L_alpha_var 同目标（最大化 S 方差），sweep 配置用此名；S=alpha_s
                lambda_S_var = getattr(opt, 'lambda_S_var', 0.0)
                if lambda_S_var > 0 and alpha_s.size(0) >= 2:
                    L_S_var = -alpha_s.squeeze(-1).var()
                    loss = loss + pid_warmup * lambda_S_var * L_S_var

                # L_S_diverse: 让 S 与样本误差对齐（高 error->高 S），充分利用 Synergy 区分难样本
                lambda_S_diverse = getattr(opt, 'lambda_S_diverse', 0.0)
                if lambda_S_diverse > 0 and alpha_s.size(0) >= 2:
                    err = (output - label).abs().squeeze()
                    if err.dim() > 1:
                        err = err.squeeze(1)
                    err_med = err.median().detach()
                    scale = err.std().detach().clamp(min=1e-4)
                    target_S = (0.1 + 0.9 * torch.sigmoid((err - err_med) / scale)).detach()
                    S_flat = alpha_s.squeeze(-1)
                    L_S_diverse = ((S_flat - target_S) ** 2).mean()
                    loss = loss + pid_warmup * lambda_S_diverse * L_S_diverse

                # L_rank: Pairwise Ranking Regularization (MPR)，缓解预测收缩/均值回归
                # 动态 Margin (CrossSent): margin=gamma*(y_i-y_j)，避免与 MAE 硬冲突
                if lambda_rank_eff > 0 and output.size(0) >= 2:
                    pred_flat = output.squeeze(-1)  # [B]
                    label_flat = label.squeeze(-1)  # [B]
                    thresh = getattr(opt, 'ranking_threshold', 0.1)
                    y_i = label_flat.unsqueeze(1)
                    y_j = label_flat.unsqueeze(0)
                    valid = (y_i > y_j) & ((y_i - y_j) > thresh)
                    if valid.any():
                        pred_i = pred_flat.unsqueeze(1)
                        pred_j = pred_flat.unsqueeze(0)
                        use_dynamic = getattr(opt, 'use_dynamic_margin_rank', True)
                        if use_dynamic:
                            margin = getattr(opt, 'rank_margin_gamma', 0.6) * (y_i - y_j)
                        else:
                            margin = getattr(opt, 'rank_margin', 0.2)
                        L_rank = (valid.float() * F.relu(margin - (pred_i - pred_j))).sum() / valid.sum().clamp(min=1)
                        loss = loss + pid_warmup * lambda_rank_eff * L_rank

            # Synergy S = alpha_s（协同路径权重）；冗余 R = alpha_r
            S_collect.append(alpha_s.detach().cpu())
            R_collect.append(alpha_r.detach().cpu())
        else:
            # KMSA loss (original)
            loss_re = loss_fn(output, label)
            lambda_classification = getattr(opt, 'lambda_classification', getattr(opt, 'lambda_cls', 0.0))
            target_binary = (label > 0).float().squeeze(-1)
            pos_weight = getattr(opt, 'cls_pos_weight', None)
            if pos_weight is not None and float(pos_weight) != 1.0:
                pos_weight = torch.full((1,), float(pos_weight), device=label.device, dtype=label.dtype)
                L_cls = F.binary_cross_entropy_with_logits(logit_cls.squeeze(-1), target_binary, pos_weight=pos_weight.squeeze(0))
            else:
                L_cls = F.binary_cross_entropy_with_logits(logit_cls.squeeze(-1), target_binary)
            loss = loss_re + lambda_classification * L_cls

            lambda_senti = getattr(opt, 'lambda_senti', 0.05)
            loss = loss + lambda_senti * senti_aux_loss

            pid_warmup_epochs = getattr(opt, 'pid_warmup_epochs', 10)
            pid_warmup = min(1.0, (epoch - 1) / max(pid_warmup_epochs, 1))
            lambda_pid = getattr(opt, 'lambda_pid', 0.05)
            loss = loss + pid_warmup * lambda_pid * L_PID

            lambda_S_var = getattr(opt, 'lambda_S_var', 0.0)
            if lambda_S_var > 0 and S.size(0) >= 2:
                L_S_var = -S.var()
                loss = loss + pid_warmup * lambda_S_var * L_S_var

            lambda_S_diverse = getattr(opt, 'lambda_S_diverse', 0.0)
            if lambda_S_diverse > 0 and S.size(0) >= 2:
                err = (output - label).abs().squeeze()
                if err.dim() > 1:
                    err = err.squeeze(1)
                err_med = err.median().detach()
                scale = err.std().detach().clamp(min=1e-4)
                target_S = (0.1 + 0.9 * torch.sigmoid((err - err_med) / scale)).detach()
                S_flat = S.squeeze()
                L_S_diverse = ((S_flat - target_S) ** 2).mean()
                loss = loss + pid_warmup * lambda_S_diverse * L_S_diverse

            # L_rank: Pairwise Ranking (MPR)，动态 margin 缓解与 MAE 冲突
            lambda_rank = getattr(opt, 'lambda_rank', 0.0)
            in_stage2 = _is_grcf_stage2(opt, epoch)
            if lambda_rank > 0 and not in_stage2 and output.size(0) >= 2:
                pred_flat = output.squeeze(-1)
                label_flat = label.squeeze(-1)
                thresh = getattr(opt, 'ranking_threshold', 0.1)
                y_i, y_j = label_flat.unsqueeze(1), label_flat.unsqueeze(0)
                valid = (y_i > y_j) & ((y_i - y_j) > thresh)
                if valid.any():
                    pred_i, pred_j = pred_flat.unsqueeze(1), pred_flat.unsqueeze(0)
                    use_dynamic = getattr(opt, 'use_dynamic_margin_rank', True)
                    margin = getattr(opt, 'rank_margin_gamma', 0.6) * (y_i - y_j) if use_dynamic else getattr(opt, 'rank_margin', 0.2)
                    L_rank = (valid.float() * F.relu(margin - (pred_i - pred_j))).sum() / valid.sum().clamp(min=1)
                    loss = loss + pid_warmup * lambda_rank * L_rank

            if getattr(opt, 'ablation_single_branch', False):
                lambda_diff = 0.0
                lambda_ortho = 0.0
                lambda_nce_diff = 0.0
            else:
                lambda_diff = getattr(opt, 'lambda_diff', 0.1)
                lambda_ortho = getattr(opt, 'lambda_ortho', 0.005)
                lambda_nce_diff = getattr(opt, 'lambda_nce_diff', 0.0)

            margin = getattr(opt, 'margin', 1.0)
            dist_intra = (F_conf - F_cons).norm(dim=-1)
            L_push = F.relu(margin - dist_intra).mean()
            loss = loss + pid_warmup * lambda_diff * L_push

            L_ortho = (F_conf.T @ F_cons).pow(2).sum()
            loss = loss + pid_warmup * lambda_ortho * L_ortho

            if lambda_nce_diff > 0 and F_conf.size(0) >= 2:
                eps = 1e-8
                F_conf_n = F_conf / (F_conf.norm(dim=-1, keepdim=True).clamp(min=eps))
                F_cons_n = F_cons / (F_cons.norm(dim=-1, keepdim=True).clamp(min=eps))
                sim_conf = F_conf_n @ F_conf_n.t()
                sim_cons = F_conf_n @ F_cons_n.t()
                tau = getattr(opt, 'nce_tau', 0.07)
                high = S >= S.median()
                B = F_conf.size(0)
                losses_nce = []
                for i in range(B):
                    if not high[i]:
                        continue
                    pos_mask = (torch.arange(B, device=S.device) != i) & high
                    if pos_mask.sum() == 0:
                        continue
                    logits_pos = sim_conf[i][pos_mask] / tau
                    logits_neg_conf = sim_conf[i][~high] / tau
                    logits_neg_cons = sim_cons[i] / tau
                    logits_neg = torch.cat([logits_neg_conf, logits_neg_cons])
                    lse_pos = torch.logsumexp(logits_pos, dim=0)
                    lse_neg = torch.logsumexp(logits_neg, dim=0)
                    L_i = -lse_pos + torch.logsumexp(torch.stack([lse_pos, lse_neg]), dim=0)
                    losses_nce.append(L_i)
                if len(losses_nce) > 0:
                    L_InfoNCE = torch.stack(losses_nce).mean()
                    loss = loss + pid_warmup * lambda_nce_diff * L_InfoNCE

            S_collect.append(S.detach().cpu())

        # NaN 检测：如果 loss 包含 NaN，跳过这个 batch
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"\nWarning: NaN/Inf detected in loss at epoch {epoch}, skipping batch")
            continue

        losses.update(loss.item(), batchsize)
        loss.backward()

        grad_clip = getattr(opt, 'grad_clip', 0.0)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()
        optimizer.zero_grad()

        y_pred.append(output.cpu())
        y_true.append(label.cpu())

        train_pbar.set_description('train')
        train_pbar.set_postfix({
            'epoch': '{}'.format(epoch),
            'loss': '{:.5f}'.format(losses.value_avg),
            'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])
        })

    if len(y_pred) == 0:
        train_results = {"MAE": float('nan'), "Corr": float('nan'), "Mult_acc_2": float('nan'),
                         "Mult_acc_3": float('nan'), "Mult_acc_5": float('nan'), "F1_score": float('nan')}
    else:
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        train_results = metrics(pred, true)
    if S_collect:
        S_all = torch.cat(S_collect)
        train_results['S_min'] = S_all.min().item()
        train_results['S_max'] = S_all.max().item()
        train_results['S_mean'] = S_all.mean().item()
        train_results['S_std'] = S_all.std().item() if S_all.numel() > 1 else 0.0
    else:
        train_results['S_min'] = train_results['S_max'] = train_results['S_mean'] = train_results['S_std'] = float('nan')
    if R_collect:
        R_all = torch.cat(R_collect)
        train_results['R_min'] = R_all.min().item()
        train_results['R_max'] = R_all.max().item()
        train_results['R_mean'] = R_all.mean().item()
        train_results['R_std'] = R_all.std().item() if R_all.numel() > 1 else 0.0
    else:
        train_results['R_min'] = train_results['R_max'] = train_results['R_mean'] = train_results['R_std'] = float('nan')

    return train_results


def evaluate(model, eval_loader, optimizer, loss_fn, epoch, metrics):
    test_pbar = tqdm(eval_loader)
    losses = AverageMeter()
    y_pred, y_true = [], []

    model.eval()
    with torch.no_grad():
        for data in test_pbar:
            inputs = {
                'V': data['vision'].to(device),
                'A': data['audio'].to(device),
                'T': data['text'].to(device),
                'mask': {
                    'V': data['vision_padding_mask'][:, 1:data['vision'].shape[1]+1].to(device),
                    'A': data['audio_padding_mask'][:, 1:data['audio'].shape[1]+1].to(device),
                    'T': []
                }
            }
            label = data['labels']['M'].to(device)
            label = label.view(-1, 1)
            batchsize = inputs['V'].shape[0]

            raw_out = model(inputs, None, epoch=epoch)
            output = raw_out['pred'] if isinstance(raw_out, dict) else raw_out[0]
            if torch.isnan(output).any() or torch.isinf(output).any():
                test_pbar.write("Warning: NaN/Inf in eval output, skipping batch")
                continue
            y_pred.append(output.cpu())
            y_true.append(label.cpu())

            loss = loss_fn(output, label)
            losses.update(loss.item(), batchsize)

            test_pbar.set_description('eval')
            test_pbar.set_postfix({
                'epoch': '{}'.format(epoch),
                'loss': '{:.5f}'.format(losses.value_avg),
                'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])
            })

        if len(y_pred) == 0:
            valid_results = {"MAE": float('nan'), "Corr": float('nan'), "Mult_acc_2": float('nan'),
                             "Mult_acc_3": float('nan'), "Mult_acc_5": float('nan'), "F1_score": float('nan')}
        else:
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            valid_results = metrics(pred, true)

    return valid_results


def test(model, test_loader, optimizer, loss_fn, epoch, metrics):
    test_pbar = tqdm(test_loader)
    losses = AverageMeter()
    y_pred, y_true = [], []

    model.eval()
    with torch.no_grad():
        for data in test_pbar:
            inputs = {
                'V': data['vision'].to(device),
                'A': data['audio'].to(device),
                'T': data['text'].to(device),
                'mask': {
                    'V': data['vision_padding_mask'][:, 1:data['vision'].shape[1]+1].to(device),
                    'A': data['audio_padding_mask'][:, 1:data['audio'].shape[1]+1].to(device),
                    'T': []
                }
            }
            ids = data['id']
            label = data['labels']['M'].to(device)
            label = label.view(-1, 1)
            batchsize = inputs['V'].shape[0]

            raw_out = model(inputs, None, epoch=epoch)
            output = raw_out['pred'] if isinstance(raw_out, dict) else raw_out[0]
            if torch.isnan(output).any() or torch.isinf(output).any():
                test_pbar.write("Warning: NaN/Inf in test output, skipping batch")
                continue
            y_pred.append(output.cpu())
            y_true.append(label.cpu())

            loss = loss_fn(output, label)
            losses.update(loss.item(), batchsize)

            test_pbar.set_description('test')
            test_pbar.set_postfix({
                'epoch': '{}'.format(epoch),
                'loss': '{:.5f}'.format(losses.value_avg),
                'lr:': '{:.2e}'.format(optimizer.state_dict()['param_groups'][0]['lr'])
            })

        if len(y_pred) == 0:
            test_results = {"MAE": float('nan'), "Corr": float('nan'), "Mult_acc_2": float('nan'),
                            "Mult_acc_3": float('nan'), "Mult_acc_5": float('nan'), "F1_score": float('nan')}
        else:
            pred, true = torch.cat(y_pred), torch.cat(y_true)
            if epoch == 11:
                calculate_u_test(pred, true)
            test_results = metrics(pred, true)

    return test_results


if __name__ == '__main__':
    main(opt)
