"""
评估脚本: 加载训练好的模型进行温度缩放校准和ECE评估
"""
import os
import torch
from opts import parse_opts
from core.dataset import MMDataLoader
from models.OverallModal import build_model
from models.TemperatureScaling import calibrate_and_evaluate
from core.metric import MetricsTop


def evaluate_with_calibration():
    opt = parse_opts()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = build_model(opt).to(device)
    
    # 加载checkpoint
    ckpt_path = os.path.join('./checkpoints', opt.datasetName.upper(), 'best.pth')
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return
    
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Loaded checkpoint from {ckpt_path}")
    print(f"  Epoch: {ckpt['epoch']}, Valid MAE: {ckpt['valid_mae']:.4f}, Valid Corr: {ckpt['valid_corr']:.4f}")
    
    # 加载数据
    dataLoader = MMDataLoader(opt)
    
    # 温度缩放校准和ECE评估
    print("\nPerforming temperature scaling calibration...")
    temperature, ece_before, ece_after = calibrate_and_evaluate(
        model, dataLoader['valid'], num_classes=opt.senti_num_classes
    )
    
    print(f"\n{'='*60}")
    print(f"Calibration Results:")
    print(f"  Optimal Temperature: {temperature:.4f}")
    print(f"  ECE Before Calibration: {ece_before:.4f}")
    print(f"  ECE After Calibration: {ece_after:.4f}")
    print(f"  ECE Improvement: {(ece_before - ece_after):.4f} ({(ece_before - ece_after)/ece_before*100:.1f}%)")
    print(f"{'='*60}\n")
    
    # 在测试集上评估
    metrics = MetricsTop().getMetics(opt.datasetName)
    model.eval()
    y_pred, y_true = [], []
    
    with torch.no_grad():
        for data in dataLoader['test']:
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
            y_pred.append(output.cpu())
            y_true.append(label.cpu())
    
    pred = torch.cat(y_pred)
    true = torch.cat(y_true)
    test_results = metrics(pred, true)
    
    print("Test Results:")
    print(f"  MAE:   {test_results['MAE']:.4f}")
    print(f"  Corr:  {test_results['Corr']:.4f}")
    print(f"  Acc-2: {test_results.get('Mult_acc_2', 0):.4f}")
    print(f"  F1:    {test_results.get('F1_score', 0):.4f}")


if __name__ == '__main__':
    evaluate_with_calibration()
