"""
Unit tests for JointGainPath (pairwise vs encoder fallback).
Run: pytest tests/test_joint_gain_path.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn


def test_joint_gain_path_pairwise_shape_and_backward():
    """Pairwise mode: forward returns [B, d], backward works."""
    from models.modules.joint_gain_path import JointGainPath
    B, L_total, d = 4, 50, 256
    len_t, len_v, len_a = 20, 15, 15
    assert len_t + len_v + len_a == L_total
    H_S_in = torch.randn(B, L_total, d)
    alpha_s = torch.rand(B, 1)
    m = JointGainPath(hidden_dim=d, nhead=4, num_layers=2, dropout=0.0, use_pairwise=True)
    F_S = m(H_S_in, alpha_s, len_t=len_t, len_v=len_v, len_a=len_a)
    assert F_S.shape == (B, d)
    loss = F_S.sum()
    loss.backward()


def test_joint_gain_path_encoder_fallback_shape_and_backward():
    """Encoder fallback: forward returns [B, d], backward works."""
    from models.modules.joint_gain_path import JointGainPath
    B, L_total, d = 4, 50, 256
    H_S_in = torch.randn(B, L_total, d)
    alpha_s = torch.rand(B, 1)
    m = JointGainPath(hidden_dim=d, nhead=4, num_layers=2, dropout=0.0, use_pairwise=False)
    F_S = m(H_S_in, alpha_s)
    assert F_S.shape == (B, d)
    loss = F_S.sum()
    loss.backward()


def test_joint_gain_path_pairwise_return_debug():
    """Pairwise + return_pairwise_debug: returns (F_S, dict)."""
    from models.modules.joint_gain_path import JointGainPath
    B, L_total, d = 2, 30, 256
    len_t, len_v, len_a = 10, 10, 10
    m = JointGainPath(hidden_dim=d, use_pairwise=True, return_pairwise_debug=True)
    out = m(torch.randn(B, L_total, d), torch.rand(B, 1), len_t=len_t, len_v=len_v, len_a=len_a)
    assert isinstance(out, tuple)
    F_S, debug = out
    assert F_S.shape == (B, d)
    assert 'F_TA' in debug and 'F_TV' in debug and 'F_AV' in debug
    assert debug['F_TA'].shape == (B, d)


def test_joint_gain_path_length_assert():
    """Length mismatch raises AssertionError with message."""
    from models.modules.joint_gain_path import JointGainPath
    B, L_total, d = 2, 30, 256
    m = JointGainPath(hidden_dim=d, use_pairwise=True)
    try:
        m(torch.randn(B, L_total, d), torch.rand(B, 1), len_t=10, len_v=10, len_a=5)
    except AssertionError as e:
        assert "len_t" in str(e) or "Length" in str(e) or str(L_total) in str(e)
    else:
        raise AssertionError("Expected AssertionError for length mismatch")


def test_alpha_s_affects_output():
    """Changing alpha_s changes F_S (gate effect)."""
    from models.modules.joint_gain_path import JointGainPath
    B, L_total, d = 2, 30, 256
    len_t, len_v, len_a = 10, 10, 10
    H_S_in = torch.randn(B, L_total, d)
    m = JointGainPath(hidden_dim=d, use_pairwise=True, use_residual_gate=True)
    F_S_low = m(H_S_in, torch.zeros(B, 1), len_t=len_t, len_v=len_v, len_a=len_a)
    F_S_high = m(H_S_in, torch.ones(B, 1), len_t=len_t, len_v=len_v, len_a=len_a)
    assert not torch.allclose(F_S_low, F_S_high), "alpha_s should affect F_S"
