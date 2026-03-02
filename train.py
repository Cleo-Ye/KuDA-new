import os
import copy
import torch
import torch.nn.functional as F
import time
from tqdm import tqdm
from opts import parse_opts
from core.dataset import MMDataLoader
from core.scheduler import get_scheduler
from core.utils import AverageMeter, setup_seed, ConfigLogging, save_print_results, calculate_u_test
from models.OverallModal import build_model
from core.metric import MetricsTop


opt = parse_opts()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(parse_args):
    opt = parse_args

    log_path = os.path.join(opt.log_path, opt.datasetName.upper())
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.join(log_path, time.strftime('%Y-%m-%d-%H:%M:%S' + '.log'))
    logger = ConfigLogging(log_file)
    logger.info(opt)    # 保存当前模型参数

    setup_seed(opt.seed)
    model = build_model(opt).to(device)
    
    # Phase 1: KI加载变为可选(通过--use_ki参数控制)
    if opt.use_ki:
        model.preprocess_model(pretrain_path={
            'T': "./pretrainedModel/KnowledgeInjectPretraining/SIMS/SIMS_T_MAE-0.278_Corr-0.765.pth",
            'V': "./pretrainedModel/KnowledgeInjectPretraining/SIMS/SIMS_V_MAE-0.522_Corr-0.520.pth",
            'A': "./pretrainedModel/KnowledgeInjectPretraining/SIMS/SIMS_A_MAE-0.516_Corr-0.261.pth"
        })      # 加载预训练权重并冻结参数
    
    # Phase 3: 冻结ConflictJS模块(只微调token筛选+融合层)
    if getattr(opt, 'freeze_conflict_js', False):
        frozen_count = 0
        for name, param in model.named_parameters():
            if 'conflict_js' in name or 'senti_proj' in name:
                param.requires_grad = False
                frozen_count += 1
        logger.info(f'Phase3 freeze: frozen {frozen_count} params in conflict_js/senti_proj')
    
    dataLoader = MMDataLoader(opt)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=opt.lr,
        weight_decay=opt.weight_decay
    )

    loss_fn = torch.nn.MSELoss()
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
    
    best_valid_mae = float('inf')
    best_valid_corr = -float('inf')
    best_epoch_mae = 0
    best_epoch_corr = 0
    best_state_mae = None
    best_state_corr = None

    for epoch in range(1, opt.n_epochs+1):
        train_results = train(model, dataLoader['train'], optimizer, loss_fn, epoch, metrics)
        valid_results = evaluate(model, dataLoader['valid'], optimizer, loss_fn, epoch, metrics)
        test_results = test(model, dataLoader['test'], optimizer, loss_fn, epoch, metrics)
        save_print_results(opt, logger, train_results, valid_results, test_results)
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
                'opt': vars(opt),
            }, ckpt_path_mae)
            logger.info(f'*** Best-MAE model saved at epoch {epoch}: MAE={cur_mae:.4f}, Corr={cur_corr:.4f} -> {ckpt_path_mae}')
        
        # Best by Corr (secondary)
        if cur_corr > best_valid_corr + 1e-6:
            best_valid_corr = cur_corr
            best_epoch_corr = epoch
            best_state_corr = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_state_corr,
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_mae': cur_mae,
                'valid_corr': cur_corr,
                'opt': vars(opt),
            }, ckpt_path_corr)
            logger.info(f'*** Best-Corr model saved at epoch {epoch}: MAE={cur_mae:.4f}, Corr={cur_corr:.4f} -> {ckpt_path_corr}')
    
    # 训练结束: 分别评估两个best模型
    logger.info(f'\n{"="*60}')
    logger.info(f'Training finished.')
    logger.info(f'  Best-MAE  epoch: {best_epoch_mae} (Valid MAE={best_valid_mae:.4f})')
    logger.info(f'  Best-Corr epoch: {best_epoch_corr} (Valid Corr={best_valid_corr:.4f})')
    
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
        logger.info(f'  MAE:   {final_test_results["MAE"]:.4f}')
        logger.info(f'  Corr:  {final_test_results["Corr"]:.4f}')
        logger.info(f'  Acc-2: {final_test_results["Mult_acc_2"]:.4f}')
        logger.info(f'  Acc-3: {final_test_results["Mult_acc_3"]:.4f}')
        logger.info(f'  Acc-5: {final_test_results["Mult_acc_5"]:.4f}')
        logger.info(f'  F1:    {final_test_results["F1_score"]:.4f}')
    logger.info(f'{"="*60}')


def train(model, train_loader, optimizer, loss_fn, epoch, metrics):
    train_pbar = tqdm(train_loader)
    losses = AverageMeter()
    y_pred, y_true = [], []

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

        output, nce_loss, senti_aux_loss, js_loss, con_loss, cal_loss, polar_logits = model(inputs, copy_label, gt_modal_labels=gt_modal_labels)

        # NaN 检测：如果输出包含 NaN，跳过这个 batch
        if torch.isnan(output).any() or torch.isinf(output).any():
            print(f"\nWarning: NaN/Inf detected in output at epoch {epoch}, skipping batch")
            continue

        loss_re = loss_fn(output, label)
        # 改进损失权重平衡: 提升js/con/cal权重,使冲突路由更重要
        lambda_nce = getattr(opt, 'lambda_nce', 0.08)      # 从0.1降到0.08
        lambda_senti = getattr(opt, 'lambda_senti', 0.03)  # 从0.05降到0.03
        lambda_js = getattr(opt, 'lambda_js', 0.15)        # 从0.1提升到0.15
        lambda_con = getattr(opt, 'lambda_con', 0.12)      # 从0.1提升到0.12
        lambda_cal = getattr(opt, 'lambda_cal', 0.12)      # 从0.1提升到0.12
        loss = (loss_re + lambda_nce * nce_loss + lambda_senti * senti_aux_loss
                + lambda_js * js_loss + lambda_con * con_loss + lambda_cal * cal_loss)

        # NaN 检测：如果 loss 包含 NaN，跳过这个 batch
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"\nWarning: NaN/Inf detected in loss at epoch {epoch}, skipping batch")
            print(f"  loss_re: {loss_re.item():.4f}, nce: {nce_loss.item():.4f}, senti: {senti_aux_loss.item():.4f}")
            print(f"  js: {js_loss.item():.4f}, con: {con_loss.item():.4f}, cal: {cal_loss.item():.4f}")
            continue

        # P2 招1: 边界敏感极性损失 L_cls = w(y)*BCE(sigmoid(z), I(y>0)), w(y)=min(1, |y|/delta)
        if polar_logits is not None:
            lambda_cls = getattr(opt, 'lambda_cls', 0.1)
            delta = getattr(opt, 'polar_delta', 0.3)
            y_flat = label.view(-1)
            target = (y_flat > 0).float().unsqueeze(1)
            w = torch.clamp(torch.abs(y_flat).unsqueeze(1) / delta, max=1.0)
            bce = F.binary_cross_entropy_with_logits(polar_logits, target, reduction='none')
            loss = loss + lambda_cls * (w * bce).mean()

        # P2 招2: 排序损失 L_rank, 采样 K_pair=32 对 (plan: margin=0.2)
        lambda_rank = getattr(opt, 'lambda_rank', 0.05)
        if lambda_rank > 0 and output.shape[0] > 1:
            margin = getattr(opt, 'rank_margin', 0.2)
            pred_flat = output.view(-1)
            y_flat = label.view(-1)
            n = len(y_flat)
            l_rank = 0.0
            n_pairs = 0
            k_pair = min(32, n * (n - 1) // 2)
            for _ in range(k_pair):
                i, j = torch.randint(0, n, (2,)).tolist()
                if i == j:
                    continue
                if y_flat[i].item() > y_flat[j].item() + 0.1:
                    l_rank = l_rank + F.relu(margin - (pred_flat[i] - pred_flat[j]))
                    n_pairs += 1
            if n_pairs > 0:
                loss = loss + lambda_rank * (l_rank / n_pairs)
        
        # 最终 NaN 检查
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"\nWarning: Final NaN/Inf in loss, skipping batch")
            continue
            
        losses.update(loss.item(), batchsize)
        loss.backward()

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

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    train_results = metrics(pred, true)

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

            output, _, _, _, _, _, _ = model(inputs, None)
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

            output, _, _, _, _, _, _ = model(inputs, None)
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

        pred, true = torch.cat(y_pred), torch.cat(y_true)
        if epoch == 11:
            calculate_u_test(pred, true)
        test_results = metrics(pred, true)

    return test_results


if __name__ == '__main__':
    main(opt)
