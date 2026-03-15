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
    """
    key = str(getattr(opt_arg, "datasetName", "")).lower()
    cfg = DATASET_CONFIGS.get(key)
    if cfg is None:
        return opt_arg

    if "dataPath" in cfg:
        opt_arg.dataPath = cfg["dataPath"]
    if "seq_lens" in cfg:
        opt_arg.seq_lens = list(cfg["seq_lens"])
    if "fea_dims" in cfg:
        opt_arg.fea_dims = list(cfg["fea_dims"])
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
    
    dataLoader = MMDataLoader(opt)
    # PID 估计器 / BatchPIDPrior 单独大学习率，避免协同度坍塌
    model_type = getattr(opt, 'model_type', 'kmsa')
    pid_name = 'batch_pid_prior' if model_type == 'pid_dualpath' else 'pid_estimator'
    pid_params = [p for n, p in model.named_parameters() if p.requires_grad and pid_name in n]
    other_params = [p for n, p in model.named_parameters() if p.requires_grad and pid_name not in n]
    pid_lr = getattr(opt, 'pid_lr', 1e-4)
    if len(pid_params) > 0 and len(other_params) > 0:
        optimizer = torch.optim.AdamW(
            [{'params': other_params, 'lr': opt.lr}, {'params': pid_params, 'lr': pid_lr}],
            weight_decay=opt.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=opt.lr,
            weight_decay=opt.weight_decay
        )

    loss_fn = torch.nn.SmoothL1Loss(beta=0.5)  # Huber loss, beta=0.5 对 SIMS 标签范围 [-1,1] 合适
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

    best_valid_mae = float('inf')
    best_valid_corr = -float('inf')
    best_epoch_mae = 0
    best_epoch_corr = 0
    best_state_mae = None
    best_state_corr = None
    start_epoch = 0

    if getattr(opt, 'resume', '') and opt.resume and os.path.isfile(opt.resume):
        ckpt_mae = torch.load(opt.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_mae['model_state_dict'])
        if 'optimizer_state_dict' in ckpt_mae:
            optimizer.load_state_dict(ckpt_mae['optimizer_state_dict'])
        start_epoch = ckpt_mae['epoch']
        best_valid_mae = ckpt_mae.get('best_valid_mae', ckpt_mae.get('valid_mae', float('inf')))
        best_epoch_mae = ckpt_mae.get('best_epoch_mae', start_epoch)
        best_state_mae = copy.deepcopy(model.state_dict())
        ckpt_dir_resume = os.path.dirname(opt.resume)
        ckpt_corr_path = os.path.join(ckpt_dir_resume, 'best_corr.pth')
        if os.path.isfile(ckpt_corr_path):
            ckpt_corr = torch.load(ckpt_corr_path, map_location=device, weights_only=False)
            best_state_corr = copy.deepcopy(ckpt_corr['model_state_dict'])
            best_valid_corr = ckpt_corr.get('best_valid_corr', ckpt_corr.get('valid_corr', -float('inf')))
            best_epoch_corr = ckpt_corr.get('best_epoch_corr', ckpt_corr.get('epoch', 0))
        else:
            best_state_corr = None
            best_valid_corr = -float('inf')
            best_epoch_corr = 0
        for _ in range(start_epoch):
            scheduler_warmup.step()
        logger.info(f'Resumed from {opt.resume}, epoch {start_epoch} -> will train to {opt.n_epochs}')

    for epoch in range(start_epoch + 1, opt.n_epochs + 1):
        train_results = train(model, dataLoader['train'], optimizer, loss_fn, epoch, metrics)
        valid_results = evaluate(model, dataLoader['valid'], optimizer, loss_fn, epoch, metrics)
        test_results = test(model, dataLoader['test'], optimizer, loss_fn, epoch, metrics)
        save_print_results(opt, logger, train_results, valid_results, test_results)
        if 'S_mean' in train_results and not (train_results['S_mean'] != train_results['S_mean']):  # not nan
            r_str = f'  [R] mean={train_results["R_mean"]:.4f}, std={train_results["R_std"]:.4f}' if 'R_mean' in train_results and not (train_results['R_mean'] != train_results['R_mean']) else ''
            logger.info(f'  [Synergy S] min={train_results["S_min"]:.4f}, max={train_results["S_max"]:.4f}, mean={train_results["S_mean"]:.4f}, std={train_results["S_std"]:.4f}{r_str}')
        scheduler_warmup.step()
        
        cur_mae = valid_results['MAE']
        cur_corr = valid_results['Corr']
        
        # Best by MAE (primary)
        is_best_mae = False
        if cur_mae < best_valid_mae - 1e-6:
            is_best_mae = True
        elif abs(cur_mae - best_valid_mae) < 1e-6 and cur_corr > best_valid_corr:
            is_best_mae = True
        
        if is_best_mae:
            best_valid_mae = cur_mae
            best_epoch_mae = epoch
            best_state_mae = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_state_mae,
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_mae': cur_mae,
                'valid_corr': cur_corr,
                'best_valid_mae': best_valid_mae,
                'best_epoch_mae': best_epoch_mae,
                'opt': vars(opt),
            }, ckpt_path_mae)
            logger.info(f'*** Best-MAE model saved at epoch {epoch}: MAE={cur_mae:.4f}, Corr={cur_corr:.4f} -> {ckpt_path_mae}')
        
        # Best by Corr (secondary)
        is_best_corr = False
        if cur_corr > best_valid_corr + 1e-6:
            is_best_corr = True
            best_valid_corr = cur_corr
            best_epoch_corr = epoch
            best_state_corr = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_state_corr,
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_mae': cur_mae,
                'valid_corr': cur_corr,
                'best_valid_corr': best_valid_corr,
                'best_epoch_corr': best_epoch_corr,
                'opt': vars(opt),
            }, ckpt_path_corr)
            logger.info(f'*** Best-Corr model saved at epoch {epoch}: MAE={cur_mae:.4f}, Corr={cur_corr:.4f} -> {ckpt_path_corr}')

        # 实时：本 epoch valid 与当前 best，以及是否刚更新（便于 tail -f 观察曲线与刷新）
        upd = []
        if is_best_mae:
            upd.append("MAE")
        if is_best_corr:
            upd.append("Corr")
        logger.info(
            f"[Epoch {epoch}] Valid MAE={cur_mae:.4f} Corr={cur_corr:.4f} | "
            f"Best MAE={best_valid_mae:.4f}@ep{best_epoch_mae} Best Corr={best_valid_corr:.4f}@ep{best_epoch_corr}"
            + (f" ** {'+'.join(upd)} updated" if upd else "")
        )
        # 写入曲线 CSV（含 train/valid），便于画图与过拟合检查
        train_mae = train_results.get('MAE', float('nan'))
        train_corr = train_results.get('Corr', float('nan'))
        with open(valid_curve_path, 'a', encoding='utf-8') as f:
            if epoch == start_epoch + 1:
                f.write("epoch,train_mae,train_corr,valid_mae,valid_corr,best_mae,best_corr\n")
            f.write(f"{epoch},{train_mae:.6f},{train_corr:.6f},{cur_mae:.6f},{cur_corr:.6f},{best_valid_mae:.6f},{best_valid_corr:.6f}\n")
        # 同一行追加到单独 progress 日志，方便 tail -f 查看与绘图
        progress_log_path = os.path.join(ckpt_dir, 'train_progress.log')
        with open(progress_log_path, 'a', encoding='utf-8') as f:
            f.write(
                f"[Epoch {epoch}] Valid MAE={cur_mae:.4f} Corr={cur_corr:.4f} | "
                f"Best MAE={best_valid_mae:.4f}@ep{best_epoch_mae} Best Corr={best_valid_corr:.4f}@ep{best_epoch_corr}"
                + (f" ** {'+'.join(upd)} updated\n" if upd else "\n")
            )

    # 训练结束: 分别评估两个best模型
    logger.info(f'\n{"="*60}')
    logger.info(f'Training finished.')
    logger.info(f'  Best-MAE  epoch: {best_epoch_mae} (Valid MAE={best_valid_mae:.4f})')
    logger.info(f'  Best-Corr epoch: {best_epoch_corr} (Valid Corr={best_valid_corr:.4f})')
    
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
        'best_valid_mae': float(best_valid_mae),
        'best_valid_corr': float(best_valid_corr),
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

        raw_out = model(inputs, copy_label, gt_modal_labels=gt_modal_labels)

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
            loss = loss + pid_warmup * lambda_pid * aux_pid_loss

            # L_aux: per-sample weighted (alpha_r * L(pred_R) + alpha_s * L(pred_S)).mean()
            L_aux_R = loss_fn(pred_R, label).squeeze(-1)   # [B]
            L_aux_S = loss_fn(pred_S, label).squeeze(-1)
            L_aux = (alpha_r.squeeze(-1) * L_aux_R + alpha_s.squeeze(-1) * L_aux_S).mean()
            lambda_aux = getattr(opt, 'lambda_aux', 0.1)
            loss = loss + lambda_aux * L_aux

            # L_ortho: 正交约束，F_R 与 F_S 余弦相似度绝对值均值（论文 3.6）
            cos_sim = F.cosine_similarity(F_R, F_S, dim=1)
            L_ortho = torch.mean(torch.abs(cos_sim))
            lambda_ortho = getattr(opt, 'lambda_ortho', 0.005)
            loss = loss + pid_warmup * lambda_ortho * L_ortho

            # L_alpha_var: 鼓励 alpha_s 在 batch 内具有方差，避免全部塌到 0.5（协同路径真正参与）
            lambda_alpha_var = getattr(opt, 'lambda_alpha_var', 0.05)
            if lambda_alpha_var > 0 and alpha_s.size(0) >= 2:
                L_alpha_var = -alpha_s.squeeze(-1).var()
                loss = loss + pid_warmup * lambda_alpha_var * L_alpha_var

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

            raw_out = model(inputs, None)
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

            raw_out = model(inputs, None)
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
