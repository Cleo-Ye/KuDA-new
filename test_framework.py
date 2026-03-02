"""
快速测试脚本: 验证Phase 1 Baseline是否能正常运行
"""
import torch
from opts import parse_opts
from core.dataset import MMDataLoader
from models.OverallModal import build_model
from core.metric import MetricsTop
from core.utils import setup_seed


def test_phase1_baseline():
    """
    测试Phase 1: 去KI + CMVN是否能正常运行
    """
    print("="*60)
    print("Testing Phase 1 Baseline (No KI + CMVN)")
    print("="*60)
    
    # 解析参数
    opt = parse_opts()
    
    # 强制设置Phase 1配置
    opt.use_ki = False
    opt.use_cmvn = True
    opt.use_conflict_js = False
    opt.use_vision_pruning = False
    opt.batch_size = 4  # 小batch测试
    
    print(f"\nDataset: {opt.datasetName}")
    print(f"Use KI: {opt.use_ki}")
    print(f"Use CMVN: {opt.use_cmvn}")
    print(f"Batch size: {opt.batch_size}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    try:
        # 设置随机种子
        setup_seed(opt.seed)
        
        # 构建模型
        print("Building model...")
        model = build_model(opt).to(device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        
        # 加载数据
        print("\nLoading data...")
        dataLoader = MMDataLoader(opt)
        print(f"Train samples: {len(dataLoader['train'].dataset)}")
        print(f"Valid samples: {len(dataLoader['valid'].dataset)}")
        print(f"Test samples: {len(dataLoader['test'].dataset)}")
        
        # 测试forward pass
        print("\nTesting forward pass...")
        model.eval()
        test_batch = next(iter(dataLoader['test']))
        
        inputs = {
            'V': test_batch['vision'].to(device),
            'A': test_batch['audio'].to(device),
            'T': test_batch['text'].to(device),
            'mask': {
                'V': test_batch['vision_padding_mask'][:, 1:test_batch['vision'].shape[1]+1].to(device),
                'A': test_batch['audio_padding_mask'][:, 1:test_batch['audio'].shape[1]+1].to(device),
                'T': []
            }
        }
        labels = test_batch['labels']['M'].to(device)
        
        with torch.no_grad():
            output, nce_loss, senti_aux_loss, js_loss, con_loss, cal_loss, _ = model(inputs, None)
        
        print(f"Input shapes:")
        print(f"  Text: {inputs['T'].shape}")
        print(f"  Audio: {inputs['A'].shape}")
        print(f"  Vision: {inputs['V'].shape}")
        print(f"Output shape: {output.shape}")
        print(f"NCE loss: {nce_loss.item():.4f}")
        print(f"Senti aux loss: {senti_aux_loss.item():.4f}")
        print(f"JS loss: {js_loss.item():.4f}")
        print(f"Sample prediction: {output[0].item():.3f}, Label: {labels[0].item():.3f}")
        
        # 测试backward pass
        print("\nTesting backward pass...")
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr)
        
        output, nce_loss, senti_aux_loss, js_loss, con_loss, cal_loss, _ = model(inputs, labels)
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(output, labels.view(-1, 1)) + 0.1 * nce_loss + 0.05 * senti_aux_loss + 0.1 * js_loss + 0.1 * con_loss + 0.1 * cal_loss
        
        loss.backward()
        optimizer.step()
        
        print(f"Loss: {loss.item():.4f}")
        print("\n✅ Phase 1 Baseline test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Phase 1 Baseline test FAILED!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_phase2_conflict_js():
    """
    测试Phase 2: ConflictJS模块是否能正常运行
    """
    print("\n" + "="*60)
    print("Testing Phase 2 Conflict-JS Module")
    print("="*60)
    
    opt = parse_opts()
    opt.use_ki = False
    opt.use_cmvn = True
    opt.use_conflict_js = True
    opt.use_routing = False  # 先测试不带路由
    opt.use_vision_pruning = False
    opt.batch_size = 4
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        setup_seed(opt.seed)
        model = build_model(opt).to(device)
        dataLoader = MMDataLoader(opt)
        
        print("\nTesting ConflictJS forward pass...")
        model.eval()
        test_batch = next(iter(dataLoader['test']))
        
        inputs = {
            'V': test_batch['vision'].to(device),
            'A': test_batch['audio'].to(device),
            'T': test_batch['text'].to(device),
            'mask': {
                'V': test_batch['vision_padding_mask'][:, 1:test_batch['vision'].shape[1]+1].to(device),
                'A': test_batch['audio_padding_mask'][:, 1:test_batch['audio'].shape[1]+1].to(device),
                'T': []
            }
        }
        
        with torch.no_grad():
            output, nce_loss, senti_aux_loss, js_loss, con_loss, cal_loss, _ = model(inputs, None)
            
            # 检查是否生成了冲突强度
            if hasattr(model, 'last_conflict_intensity'):
                C = model.last_conflict_intensity
                print(f"Conflict intensity C: shape={C.shape}, mean={C.mean().item():.4f}")
                print(f"C range: [{C.min().item():.4f}, {C.max().item():.4f}]")
                
                # 检查证据masks
                if hasattr(model, 'last_con_masks'):
                    for m in ['T', 'A', 'V']:
                        con_count = model.last_con_masks[m].sum(dim=1).float().mean().item()
                        conf_count = model.last_conf_masks[m].sum(dim=1).float().mean().item()
                        print(f"{m} - Congruent tokens: {con_count:.1f}, Conflict tokens: {conf_count:.1f}")
                
                print("\n✅ Phase 2 Conflict-JS test PASSED!")
                return True
            else:
                print("❌ Conflict intensity not recorded")
                return False
                
    except Exception as e:
        print(f"\n❌ Phase 2 test FAILED!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import sys
    
    # 测试Phase 1
    phase1_ok = test_phase1_baseline()
    
    if phase1_ok:
        # 测试Phase 2
        phase2_ok = test_phase2_conflict_js()
        
        if phase2_ok:
            print("\n" + "="*60)
            print("✅ All tests PASSED! Ready to run full experiments.")
            print("="*60)
            print("\nNext steps:")
            print("1. Run: python train.py --use_ki False --use_cmvn True")
            print("2. Run: python run_all_experiments.py")
            print("3. Run: python visualize_results.py")
        else:
            print("\n⚠️ Phase 2 test failed. Please fix before proceeding.")
            sys.exit(1)
    else:
        print("\n⚠️ Phase 1 test failed. Please fix before proceeding.")
        sys.exit(1)
