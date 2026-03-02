"""
快速维度测试：验证SentimentProjector的输入维度是否正确
"""
import torch
from opts import parse_opts
from models.OverallModal import build_model

def test_dimensions():
    """测试模型各模块的维度"""
    opt = parse_opts()
    opt.use_ki = False
    opt.use_cmvn = True
    opt.use_conflict_js = True
    opt.batch_size = 2  # 小batch测试
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Building model...")
    model = build_model(opt).to(device)
    
    print("\nChecking SentimentProjector dimensions:")
    print(f"Text projector input dim: {model.UniEncKI.senti_proj_t.projector[0].in_features}")
    print(f"Vision projector input dim: {model.UniEncKI.senti_proj_v.projector[0].in_features}")
    print(f"Audio projector input dim: {model.UniEncKI.senti_proj_a.projector[0].in_features}")
    
    # 创建dummy输入测试
    print("\nTesting with dummy inputs...")
    B = 2
    dummy_inputs = {
        'T': torch.randn(B, 50, 768).to(device),
        'A': torch.randn(B, 400, 25).to(device),
        'V': torch.randn(B, 55, 177).to(device),
        'mask': {
            'T': [],
            'A': torch.zeros(B, 401, dtype=torch.bool).to(device),
            'V': torch.zeros(B, 56, dtype=torch.bool).to(device)
        }
    }
    
    try:
        model.eval()
        with torch.no_grad():
            output, nce_loss = model(dummy_inputs, None)
        
        print(f"✅ Forward pass successful!")
        print(f"Output shape: {output.shape}")
        print(f"NCE loss: {nce_loss.item():.4f}")
        
        if hasattr(model, 'last_conflict_intensity'):
            print(f"Conflict intensity C: {model.last_conflict_intensity}")
        
        return True
    except Exception as e:
        print(f"❌ Forward pass failed!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_dimensions()
    if success:
        print("\n✅ All dimension checks passed! Ready to train.")
    else:
        print("\n❌ Dimension check failed. Please review the error above.")
