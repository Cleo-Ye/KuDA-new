"""
消融实验配置文件
定义所有实验组的配置
"""

# 消融实验表中的所有实验组配置
ABLATION_CONFIGS = {
    # 原始KuDA(对比用)
    'KuDA_original': {
        'use_ki': True,
        'use_cmvn': False,
        'use_conflict_js': False,
        'use_routing': False,
        'use_vision_pruning': False,
        'description': 'Original KuDA with Knowledge Injection'
    },
    
    # Phase 1: Baseline
    'baseline_phase1': {
        'use_ki': False,
        'use_cmvn': True,
        'use_conflict_js': False,
        'use_routing': False,
        'use_vision_pruning': False,
        'description': 'Baseline: Remove KI + Add CMVN'
    },
    
    # Phase 2 渐进式消融
    'phase2_evidence_split_only': {
        'use_ki': False,
        'use_cmvn': True,
        'use_conflict_js': True,
        'use_routing': False,  # 只拆分,不做路由
        'use_vision_pruning': False,
        'description': 'Phase2: Evidence split only (no routing)'
    },
    
    'phase2_with_js_no_routing': {
        'use_ki': False,
        'use_cmvn': True,
        'use_conflict_js': True,
        'compute_C': True,  # 计算C但不用于路由
        'use_routing': False,
        'use_vision_pruning': False,
        'description': 'Phase2: With Evidence-JS but no routing'
    },
    
    'phase2_full': {
        'use_ki': False,
        'use_cmvn': True,
        'use_conflict_js': True,
        'use_routing': True,
        'use_vision_pruning': False,
        'description': 'Phase2 Full: Evidence split + JS + Routing'
    },
    
    # Phase 3: Token筛选
    'phase3_full': {
        'use_ki': False,
        'use_cmvn': True,
        'use_conflict_js': True,
        'use_routing': True,
        'use_vision_pruning': True,
        'vision_target_ratio': 0.3,
        'vision_conf_ratio': 0.5,
        'description': 'Phase3 Full: All modules + Vision token pruning'
    },
    
    # Token筛选的不同配置(用于Phase3内部消融)
    'phase3_pruning_aggressive': {
        'use_ki': False,
        'use_cmvn': True,
        'use_conflict_js': True,
        'use_routing': True,
        'use_vision_pruning': True,
        'vision_target_ratio': 0.2,  # 更激进的压缩
        'vision_conf_ratio': 0.6,
        'description': 'Phase3: Aggressive pruning (20% retention)'
    },
    
    'phase3_pruning_conservative': {
        'use_ki': False,
        'use_cmvn': True,
        'use_conflict_js': True,
        'use_routing': True,
        'use_vision_pruning': True,
        'vision_target_ratio': 0.5,  # 保守的压缩
        'vision_conf_ratio': 0.4,
        'description': 'Phase3: Conservative pruning (50% retention)'
    },

    # Phase 1 必须产出: 保留率 r vs 性能/计算 曲线 (多档 vision_target_ratio)
    'phase3_ratio_02': {
        'use_ki': False,
        'use_cmvn': True,
        'use_conflict_js': True,
        'use_routing': True,
        'use_vision_pruning': True,
        'vision_target_ratio': 0.2,
        'description': 'IEC r=0.2: Vision retention 20%'
    },
    'phase3_ratio_04': {
        'use_ki': False,
        'use_cmvn': True,
        'use_conflict_js': True,
        'use_routing': True,
        'use_vision_pruning': True,
        'vision_target_ratio': 0.4,
        'description': 'IEC r=0.4: Vision retention 40%'
    },
    'phase3_ratio_06': {
        'use_ki': False,
        'use_cmvn': True,
        'use_conflict_js': True,
        'use_routing': True,
        'use_vision_pruning': True,
        'vision_target_ratio': 0.6,
        'description': 'IEC r=0.6: Vision retention 60%'
    },
    'phase3_ratio_08': {
        'use_ki': False,
        'use_cmvn': True,
        'use_conflict_js': True,
        'use_routing': True,
        'use_vision_pruning': True,
        'vision_target_ratio': 0.8,
        'description': 'IEC r=0.8: Vision retention 80%'
    },
}


# 超参数搜索范围(可选)
HYPERPARAMETER_SEARCH = {
    'tau_conf': [0.2, 0.3, 0.4],      # 冲突阈值
    'tau_con': [0.05, 0.1, 0.15],     # 一致阈值
    'tau_rel': [0.4, 0.5, 0.6],       # 置信度阈值
    'vision_target_ratio': [0.2, 0.3, 0.4, 0.5],
    'vision_conf_ratio': [0.3, 0.4, 0.5, 0.6]
}


# 数据集配置
DATASET_CONFIGS = {
    'sims': {
        'dataPath': '/18T/yechenlu/MSA_datasets/SIMS-v2/ch-sims2s/unaligned.pkl',
        'seq_lens': [50, 55, 400],
        'fea_dims': [768, 177, 25],
        'bert_pretrained': './pretrainedModel/BERT/bert-base-chinese'
    },
    'simsv2': {
        'dataPath': '/18T/yechenlu/MSA_datasets/SIMS-v2/ch-sims2s/unaligned.pkl',
        'seq_lens': [50, 55, 400],
        'fea_dims': [768, 177, 25],
        'bert_pretrained': './pretrainedModel/BERT/bert-base-chinese'
    },
    'mosi': {
        'dataPath': 'path_to_mosi.pkl',
        'seq_lens': [50, 500, 500],  # 需要根据实际数据调整
        'fea_dims': [768, 709, 33],
        'bert_pretrained': './pretrainedModel/BERT/bert-base-uncased'
    },
    'mosei': {
        'dataPath': 'path_to_mosei.pkl',
        'seq_lens': [50, 500, 500],
        'fea_dims': [768, 709, 33],
        'bert_pretrained': './pretrainedModel/BERT/bert-base-uncased'
    }
}
