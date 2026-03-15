"""
完整实验运行脚本
按照消融实验表逐个运行所有实验组
每个实验: 构建模型 → 训练N个epoch → 在测试集上评估
"""
import os
import sys
import json
import copy
import torch
import torch.nn.functional as F
from tqdm import tqdm
from opts import parse_opts
from core.dataset import MMDataLoader
from core.scheduler import get_scheduler
from core.utils import setup_seed
from models.OverallModal import build_model
from core.metric import MetricsTop
from evaluate_experiments import run_ablation_experiments, print_ablation_table, visualize_conflict_intensity_distribution


def train_model(model, dataLoader, opt, device):
    """
    完整训练循环
    
    Args:
        model: 模型
        dataLoader: 数据加载器dict (train/valid/test)
        opt: 配置
        device: 设备
    Returns:
        model: 训练后的模型
        best_valid_mae: 最佳验证MAE
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt.lr,
        weight_decay=opt.weight_decay
    )
    loss_fn = torch.nn.MSELoss()
    metrics = MetricsTop().getMetics(opt.datasetName)
    scheduler = get_scheduler(optimizer, opt.n_epochs)
    
    best_valid_mae = float('inf')
    best_state = None
    
    for epoch in range(1, opt.n_epochs + 1):
        # --- Train ---
        model.train()
        train_pbar = tqdm(dataLoader['train'], desc=f'train epoch {epoch}')
        train_losses = []
        
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
            label = data['labels']['M'].to(device).view(-1, 1)
            copy_label = label.clone().detach()
            
            output, senti_aux_loss, L_PID, F_cons, F_conf, S, _ = model(inputs, copy_label)
            
            loss_re = loss_fn(output, label)
            lambda_senti = getattr(opt, 'lambda_senti', 0.05)
            pid_warmup = min(1.0, (epoch - 1) / max(getattr(opt, 'pid_warmup_epochs', 10), 1))
            loss = loss_re + lambda_senti * senti_aux_loss + pid_warmup * getattr(opt, 'lambda_pid', 0.05) * L_PID
            dist_intra = (F_conf - F_cons).norm(dim=-1)
            loss = loss + pid_warmup * getattr(opt, 'lambda_diff', 0.1) * F.relu(getattr(opt, 'margin', 1.0) - dist_intra).mean()
            loss = loss + pid_warmup * getattr(opt, 'lambda_ortho', 0.01) * (F_conf.T @ F_cons).pow(2).sum()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            train_losses.append(loss.item())
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # --- Validate ---
        model.eval()
        valid_preds, valid_labels = [], []
        with torch.no_grad():
            for data in dataLoader['valid']:
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
                label = data['labels']['M'].to(device).view(-1, 1)
                output, _, _, _, _, _, _ = model(inputs, None)
                valid_preds.append(output.cpu())
                valid_labels.append(label.cpu())
        
        valid_preds = torch.cat(valid_preds)
        valid_labels = torch.cat(valid_labels)
        valid_mae = torch.mean(torch.abs(valid_preds - valid_labels)).item()
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        print(f"  Epoch {epoch}/{opt.n_epochs} | Train Loss: {avg_train_loss:.4f} | Valid MAE: {valid_mae:.4f}")
        
        # 保存最佳模型
        if valid_mae < best_valid_mae:
            best_valid_mae = valid_mae
            best_state = copy.deepcopy(model.state_dict())
        
        scheduler.step()
    
    # 恢复最佳模型
    if best_state is not None:
        model.load_state_dict(best_state)
    
    print(f"  Best Valid MAE: {best_valid_mae:.4f}")
    return model, best_valid_mae


def run_single_experiment(opt, exp_name, exp_config, device):
    """
    运行单个实验: 构建 → 训练 → 评估
    """
    print(f"\n{'='*60}")
    print(f"Running Experiment: {exp_name}")
    print(f"Config: {exp_config}")
    print(f"{'='*60}\n")
    
    # 更新配置
    for key, value in exp_config.items():
        setattr(opt, key, value)
    
    # 设置随机种子确保可复现
    setup_seed(opt.seed)
    
    # 构建模型和数据
    model = build_model(opt).to(device)
    dataLoader = MMDataLoader(opt)
    
    # KuDA原版需要加载预训练权重
    if getattr(opt, 'use_ki', False):
        model.preprocess_model(pretrain_path={
            'T': "./pretrainedModel/KnowledgeInjectPretraining/SIMS/SIMS_T_MAE-0.278_Corr-0.765.pth",
            'V': "./pretrainedModel/KnowledgeInjectPretraining/SIMS/SIMS_V_MAE-0.522_Corr-0.520.pth",
            'A': "./pretrainedModel/KnowledgeInjectPretraining/SIMS/SIMS_A_MAE-0.516_Corr-0.261.pth"
        })
    
    # 训练模型
    model, best_valid_mae = train_model(model, dataLoader, opt, device)
    
    # 在测试集上评估
    results = run_ablation_experiments(
        model, 
        dataLoader['test'], 
        device, 
        opt,
        logger=None
    )
    results['best_valid_mae'] = best_valid_mae
    
    return results


def main():
    opt = parse_opts()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 定义所有消融实验配置
    experiments = {
        'KuDA(original)': {
            'use_ki': True,
            'use_cmvn': False,
            'use_conflict_js': False,
            'use_routing': False,
            'use_vision_pruning': False,
            'n_epochs': 50
        },
        'Baseline(Phase1)': {
            'use_ki': False,
            'use_cmvn': True,
            'use_conflict_js': False,
            'use_routing': False,
            'use_vision_pruning': False,
            'n_epochs': 50
        },
        '+EvidenceSplit': {
            'use_ki': False,
            'use_cmvn': True,
            'use_conflict_js': True,
            'use_routing': False,
            'use_vision_pruning': False,
            'n_epochs': 50
        },
        '+EvidenceJS': {
            'use_ki': False,
            'use_cmvn': True,
            'use_conflict_js': True,
            'use_routing': False,
            'use_vision_pruning': False,
            'n_epochs': 50
        },
        'Full(Phase2)': {
            'use_ki': False,
            'use_cmvn': True,
            'use_conflict_js': True,
            'use_routing': True,
            'use_vision_pruning': False,
            'n_epochs': 50
        },
        '+TokenPruning(Phase3)': {
            'use_ki': False,
            'use_cmvn': True,
            'use_conflict_js': True,
            'use_routing': True,
            'use_vision_pruning': True,
            'n_epochs': 50
        }
    }
    
    # 运行所有实验
    all_results = {}
    for exp_name, exp_config in experiments.items():
        try:
            results = run_single_experiment(opt, exp_name, exp_config, device)
            all_results[exp_name] = results
            
            # 保存中间结果
            os.makedirs('./results', exist_ok=True)
            with open(f'./results/{exp_name.replace(" ", "_")}.json', 'w') as f:
                json.dump({k: str(v) for k, v in results.items()}, f, indent=2)
            
            print(f"\n✅ {exp_name} completed: MAE={results['overall']['MAE']:.4f}, Corr={results['overall']['Corr']:.4f}")
        except Exception as e:
            import traceback
            print(f"\n❌ Error in {exp_name}: {e}")
            traceback.print_exc()
            all_results[exp_name] = {'error': str(e)}
    
    # 打印汇总表格
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    # 过滤掉错误的实验
    valid_results = {k: v for k, v in all_results.items() if 'error' not in v}
    if valid_results:
        print_ablation_table(valid_results)
    
    # 保存最终结果
    with open('./results/ablation_summary.json', 'w') as f:
        json.dump({k: str(v) for k, v in all_results.items()}, f, indent=2)
    
    print(f"\nResults saved to ./results/")


if __name__ == '__main__':
    main()
