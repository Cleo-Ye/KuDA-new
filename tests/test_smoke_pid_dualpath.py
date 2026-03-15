"""
Smoke test: full forward + backward for PIDDualPathMSA (Route B).
Run: pytest tests/test_smoke_pid_dualpath.py -v
Uses a mock encoder to avoid BERT/data dependency; exercises aligner -> router -> dual path -> heads.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn


class _FakeUniEncKI(nn.Module):
    """nn.Module that returns fixed (uni_fea, uni_senti, posteriors, senti_scores) for smoke tests."""
    def __init__(self, B, L_t, L_v, L_a, device=None):
        super().__init__()
        self.B, self.L_t, self.L_v, self.L_a = B, L_t, L_v, L_a
        self._device = device

    def forward(self, inputs_data_mask):
        device = self._device or next(self.parameters()).device
        B, L_t, L_v, L_a = self.B, self.L_t, self.L_v, self.L_a
        uni_fea = {
            'T': torch.randn(B, L_t, 768, device=device),
            'V': torch.randn(B, L_v, 128, device=device),
            'A': torch.randn(B, L_a, 128, device=device),
        }
        uni_senti = {'T': torch.randn(B, 1, device=device), 'V': torch.randn(B, 1, device=device), 'A': torch.randn(B, 1, device=device)}
        C = 7
        posteriors = {
            'T': torch.softmax(torch.randn(B, L_t, C, device=device), dim=-1),
            'V': torch.softmax(torch.randn(B, L_v, C, device=device), dim=-1),
            'A': torch.softmax(torch.randn(B, L_a, C, device=device), dim=-1),
        }
        senti_scores = {'T': torch.randn(B, L_t, device=device), 'V': torch.randn(B, L_v, device=device), 'A': torch.randn(B, L_a, device=device)}
        return uni_fea, uni_senti, posteriors, senti_scores


def _make_fake_opt():
    class Opt:
        datasetName = 'sims'
        hidden_size = 256
        model_type = 'pid_dualpath'
        use_batch_pid_prior = False
        dropout = 0.1
        path_nhead = 4
        path_layers = 2
        lambda_r = 0.5
        lambda_s = 0.5
        router_tau = 1.0
        fea_dims = [768, 177, 25]
        seq_lens = [50, 55, 400]
        senti_num_classes = 7
        hf_cache_dir = None
        use_pairwise_joint_path = True
        use_residual_gate_in_joint_path = True
        return_pairwise_debug = False
        debug_joint_path = False
    return Opt()


def _make_fake_input(B=2, L_t=10, L_v=12, L_a=20):
    return {
        'T': torch.randn(B, L_t, 768),
        'V': torch.randn(B, L_v, 177),
        'A': torch.randn(B, L_a, 25),
        'mask': {
            'T': torch.zeros(B, L_t).bool(),
            'V': torch.zeros(B, L_v).bool(),
            'A': torch.zeros(B, L_a).bool(),
        }
    }


def test_smoke_pid_dualpath_forward_pairwise_true():
    """Full forward with use_pairwise_joint_path=True; check shapes and backward."""
    from models.pid_dualpath_msa import PIDDualPathMSA
    opt = _make_fake_opt()
    opt.use_pairwise_joint_path = True
    model = PIDDualPathMSA(opt, dataset='sims', use_batch_pid_prior=False)
    B, L_t, L_v, L_a = 2, 10, 12, 20
    device = next(model.parameters()).device
    model.UniEncKI = _FakeUniEncKI(B, L_t, L_v, L_a, device).to(device)
    model.eval()
    inp = _make_fake_input(B, L_t, L_v, L_a)
    inp = {k: v.to(device) if torch.is_tensor(v) else {kk: vv.to(device) for kk, vv in v.items()} for k, v in inp.items()}
    label = torch.randn(B, 1, device=device)
    with torch.no_grad():
        out = model(inp, label)
    assert out['pred'].shape == (B, 1)
    assert out['pred_R'].shape == (B, 1)
    assert out['pred_S'].shape == (B, 1)
    assert out['F_R'].shape == (B, 256)
    assert out['F_S'].shape == (B, 256)
    assert out['alpha_r'].shape == (B, 1)
    assert out['alpha_s'].shape == (B, 1)
    assert out['aux_pid_loss'].dim() == 0
    # backward
    model.train()
    out = model(inp, label)
    loss = out['pred'].sum() + out['aux_pid_loss']
    loss.backward()


def test_smoke_pid_dualpath_forward_pairwise_false():
    """Full forward with use_pairwise_joint_path=False (encoder fallback)."""
    from models.pid_dualpath_msa import PIDDualPathMSA
    opt = _make_fake_opt()
    opt.use_pairwise_joint_path = False
    model = PIDDualPathMSA(opt, dataset='sims', use_batch_pid_prior=False)
    B, L_t, L_v, L_a = 2, 10, 12, 20
    device = next(model.parameters()).device
    model.UniEncKI = _FakeUniEncKI(B, L_t, L_v, L_a, device).to(device)
    model.eval()
    inp = _make_fake_input(B, L_t, L_v, L_a)
    inp = {k: v.to(device) if torch.is_tensor(v) else {kk: vv.to(device) for kk, vv in v.items()} for k, v in inp.items()}
    label = torch.randn(B, 1, device=device)
    with torch.no_grad():
        out = model(inp, label)
    assert out['pred'].shape == (B, 1)
    assert out['F_S'].shape == (B, 256)
    model.train()
    out = model(inp, label)
    loss = out['pred'].sum()
    loss.backward()


def test_smoke_return_pairwise_debug():
    """When return_pairwise_debug=True, dict contains F_TA, F_TV, F_AV."""
    from models.pid_dualpath_msa import PIDDualPathMSA
    opt = _make_fake_opt()
    opt.use_pairwise_joint_path = True
    opt.return_pairwise_debug = True
    model = PIDDualPathMSA(opt, dataset='sims', use_batch_pid_prior=False)
    B, L_t, L_v, L_a = 2, 10, 12, 20
    device = next(model.parameters()).device
    model.UniEncKI = _FakeUniEncKI(B, L_t, L_v, L_a, device).to(device)
    inp = _make_fake_input(B, L_t, L_v, L_a)
    inp = {k: v.to(device) if torch.is_tensor(v) else {kk: vv.to(device) for kk, vv in v.items()} for k, v in inp.items()}
    with torch.no_grad():
        out = model(inp, torch.randn(B, 1, device=device))
    assert 'F_TA' in out and 'F_TV' in out and 'F_AV' in out
    assert out['F_TA'].shape == (B, 256)
