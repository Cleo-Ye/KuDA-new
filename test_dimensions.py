"""
快速维度测试：验证重构后模型的输入输出维度是否正确
PIDEstimator + DualBranchExtractor + regressor 全路径
"""
import torch
from opts import parse_opts
from models.OverallModal import build_model


def test_dimensions():
    """测试重构模型各模块的维度"""
    opt = parse_opts()
    opt.use_ki = False
    opt.use_cmvn = True
    opt.batch_size = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Building model...")
    model = build_model(opt).to(device)

    print("\nChecking SentimentProjector dimensions:")
    print(f"Text projector input dim: {model.UniEncKI.senti_proj_t.projector[0].in_features}")
    print(f"Vision projector input dim: {model.UniEncKI.senti_proj_v.projector[0].in_features}")
    print(f"Audio projector input dim: {model.UniEncKI.senti_proj_a.projector[0].in_features}")

    # Create dummy inputs with correct format
    # Text T: [B, 3, L] — channel 0 = input_ids (long, in [0, vocab_size)),
    #                      channel 1 = attention_mask (float 0/1),
    #                      channel 2 = token_type_ids (long 0)
    print("\nTesting with dummy inputs (correct BERT format)...")
    B = 2
    seq_t, seq_v, seq_a = opt.seq_lens  # e.g. [50, 55, 400]
    # Dynamically get vocab size from the loaded BERT model to avoid out-of-range ids
    VOCAB_SIZE = model.UniEncKI.enc_t.encoder.model.embeddings.word_embeddings.num_embeddings
    print(f"BERT vocab size: {VOCAB_SIZE}")

    # Build BERT-style text input: [B, 3, seq_t]
    t_ids    = torch.randint(1, VOCAB_SIZE - 1, (B, seq_t)).long()   # input_ids
    t_mask   = torch.ones(B, seq_t).float()                           # attention_mask
    t_segs   = torch.zeros(B, seq_t).long()                           # token_type_ids
    t_input  = torch.stack([t_ids.float(), t_mask, t_segs.float()], dim=1)  # [B, 3, seq_t]

    dummy_inputs = {
        'T': t_input.to(device),
        'A': torch.randn(B, seq_a, opt.fea_dims[2]).to(device),
        'V': torch.randn(B, seq_v, opt.fea_dims[1]).to(device),
        'mask': {
            'T': [],
            'A': torch.zeros(B, seq_a, dtype=torch.bool).to(device),
            'V': torch.zeros(B, seq_v, dtype=torch.bool).to(device),
        }
    }

    try:
        model.eval()
        with torch.no_grad():
            output, senti_aux_loss, L_PID, F_cons, F_conf, S, _ = model(dummy_inputs, None)

        print(f"✅ Forward pass successful!")
        print(f"Output shape: {output.shape}")
        print(f"F_cons shape: {F_cons.shape}, F_conf shape: {F_conf.shape}")
        print(f"L_PID: {L_PID.item():.4f}")
        print(f"S shape: {S.shape}, S mean: {S.mean().item():.4f}, S range: [{S.min().item():.4f}, {S.max().item():.4f}]")

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
