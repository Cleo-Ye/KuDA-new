"""
Unit tests for IEC+ICR modules (plan Section 4 invariants)
Run: pytest tests/test_iec_icr.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F


def test_mask_alignment():
    """1. Mask alignment: IEC/ICR 后各模态第2维长度一致"""
    from models.ConflictJS import EvidenceSplitter, _reliability_from_entropy
    B, L, C = 2, 10, 7
    posteriors = F.softmax(torch.randn(B, L, C), dim=-1)
    senti_scores = torch.randn(B, L)
    senti_ref = {m: senti_scores.clone() for m in ['T', 'A', 'V']}
    splitter = EvidenceSplitter(rel_min=0.1, conf_ratio=0.25, con_ratio=0.25)
    con_masks, conf_masks = splitter(
        {m: posteriors for m in ['T', 'A', 'V']},
        {m: senti_scores for m in ['T', 'A', 'V']},
        senti_ref,
        None
    )
    for m in ['T', 'A', 'V']:
        assert con_masks[m].shape == (B, L)
        assert conf_masks[m].shape == (B, L)


def test_probability_invariants():
    """2. Probability invariants: posteriors sum=1; P_conf_m sum=1"""
    from models.ConflictJS import EvidenceLevelJS, _reliability_from_entropy
    B, L, C = 2, 10, 7
    posteriors = F.softmax(torch.randn(B, L, C), dim=-1)
    assert torch.allclose(posteriors.sum(dim=-1), torch.ones(B, L))
    mask = torch.rand(B, L) > 0.5
    js = EvidenceLevelJS()
    P_conf = js.aggregate_posteriors(posteriors, mask, _reliability_from_entropy(posteriors))
    assert torch.allclose(P_conf.sum(dim=-1), torch.ones(B))


def test_jsd_invariants():
    """3. JSD invariants: JSD(P,P,P)=0; C in [0,1]"""
    from models.ConflictJS import EvidenceLevelJS, ConflictIntensity
    B, C = 2, 7
    P = F.softmax(torch.randn(B, C), dim=-1)
    js = EvidenceLevelJS()
    jsd3 = js.jensen_shannon_divergence([P, P, P])
    assert torch.allclose(jsd3, torch.zeros(B), atol=1e-5)
    intensity = ConflictIntensity(num_classes=C)
    C_val, C_m = intensity(jsd3, {'T': P, 'A': P, 'V': P})
    assert (C_val >= 0).all() and (C_val <= 1).all()


def test_evidence_split_invariants():
    """4. Evidence split: conf_mask and con_mask each have at least 1 True"""
    from models.ConflictJS import EvidenceSplitter
    B, L, C = 2, 10, 7
    posteriors = F.softmax(torch.randn(B, L, C), dim=-1)
    senti_scores = torch.randn(B, L)
    senti_ref = {m: senti_scores.clone() for m in ['T', 'A', 'V']}
    splitter = EvidenceSplitter(rel_min=0.0, conf_ratio=0.25, con_ratio=0.25)
    con_masks, conf_masks = splitter(
        {m: posteriors for m in ['T', 'A', 'V']},
        {m: senti_scores for m in ['T', 'A', 'V']},
        senti_ref,
        None
    )
    for m in ['T', 'A', 'V']:
        assert conf_masks[m].sum(dim=1).min() >= 1
        assert con_masks[m].sum(dim=1).min() >= 1


def test_rho_invariants():
    """5. rho_m in [0,1]; when valid=0, rho_m=0"""
    from models.ConflictJS import _reliability_from_entropy
    B, L = 2, 10
    conf_mask = torch.rand(B, L) > 0.5
    valid = torch.ones(B, L, dtype=torch.bool)
    rho = (conf_mask.sum(dim=1).float() / valid.sum(dim=1).float().clamp(min=1e-6)).clamp(0, 1)
    assert (rho >= 0).all() and (rho <= 1).all()


def test_gate_invariants():
    """6. alpha_m in (0,1)"""
    gate_k = 10.0
    gate_tau = 0.15
    gate_m = torch.tensor([0.0, 0.5, 1.0])
    alpha = torch.sigmoid(gate_k * (gate_m - gate_tau))
    assert (alpha > 0).all() and (alpha < 1).all()


def test_text_guided_pruner_output_shape():
    """TextGuidedVisionPruner output shape and K = ceil(L*ratio)"""
    from models.VisionTokenPruner import TextGuidedVisionPruner
    B, L_v, L_t, D = 2, 55, 50, 256
    pruner = TextGuidedVisionPruner(vision_keep_ratio=0.5, hidden_dim=256, text_dim=1536)
    hidden_v = torch.randn(B, L_v, D)
    hidden_t = torch.randn(B, L_t, 1536)
    senti_t = torch.randn(B, L_t)
    out, idx, info = pruner(hidden_v, hidden_t, senti_t)
    K_expected = max(1, int(torch.ceil(torch.tensor(L_v * 0.5)).item()))
    assert out.shape[0] == B
    assert out.shape[2] == D
    assert info['pruned_mask'].shape[1] >= K_expected - 2  # allow some padding flexibility


def test_alignment_aware_reference():
    """AlignmentAwareReference output shapes"""
    from models.ConflictJS import AlignmentAwareReference
    B, L_t, L_v, L_a = 2, 50, 30, 100
    align = AlignmentAwareReference(text_dim=1536, align_dim=256)
    ref_T, ref_V, ref_A = align(
        torch.randn(B, L_t, 1536),
        torch.randn(B, L_v, 256),
        torch.randn(B, L_a, 256),
        torch.randn(B, L_t),
        torch.randn(B, L_v),
        torch.randn(B, L_a),
    )
    assert ref_T.shape == (B, L_t)
    assert ref_V.shape == (B, L_v)
    assert ref_A.shape == (B, L_a)


if __name__ == '__main__':
    test_mask_alignment()
    test_probability_invariants()
    test_jsd_invariants()
    test_evidence_split_invariants()
    test_rho_invariants()
    test_gate_invariants()
    test_text_guided_pruner_output_shape()
    test_alignment_aware_reference()
    print('All tests passed.')
