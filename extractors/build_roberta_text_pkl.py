import os
import argparse
import pickle
from copy import deepcopy

import numpy as np
from transformers import AutoTokenizer


class SimpleRobertaTokenizer:
    """
    只实现 KuDA 生成 text_bert 所需的 tokenize 功能，
    使用 AutoTokenizer 自动匹配模型（支持 BERT、RoBERTa、chinese-roberta-wwm-ext 等）。
    生成 pkl 时的 pretrained 必须与 experiment_configs.text_encoder_pretrained 一致。
    """

    def __init__(self, pretrained: str = "hfl/chinese-roberta-wwm-ext"):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained)

    def tokenize_to_text_bert(self, text: str) -> np.ndarray:
        """
        模仿 MMSA-FET 的 robertaExtractor.tokenize 输出：
        返回形状 [L, 3] 的 numpy 数组，每一列分别是
        [input_id, attention_mask, token_type_id]。
        """
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            return_tensors="np",
        )
        # encoded["input_ids"]:  [1, L]
        # encoded["attention_mask"]: [1, L]
        input_ids = encoded["input_ids"].squeeze(0)        # [L]
        attention_mask = encoded["attention_mask"].squeeze(0)  # [L]

        # RoBERTa 没有 segment 概念，统一用 0 即可
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)  # [L]

        # 拼成 [L, 3]
        stacked = np.stack(
            [input_ids.astype(np.int64),
             attention_mask.astype(np.int64),
             token_type_ids],
            axis=1,
        )  # [L, 3]
        return stacked


def build_roberta_extractor(pretrained: str = "hfl/chinese-roberta-wwm-ext") -> SimpleRobertaTokenizer:
    """
    构建一个轻量级 RoBERTa tokenizer 封装，仅用于生成 text_bert。
    """
    return SimpleRobertaTokenizer(pretrained=pretrained)


def tokenize_split(raw_text_list, extractor: SimpleRobertaTokenizer):
    """
    给一个 split（train/valid/test）的 raw_text 列表，生成新的 text_bert 数组。

    返回:
        text_bert: np.ndarray, shape [N, 3, L_max]
    """
    text_bert_list = []

    for i, txt in enumerate(raw_text_list):
        if not isinstance(txt, str):
            # 某些数据集 raw_text 可能是 bytes 或 list，按需要清洗
            if isinstance(txt, bytes):
                txt = txt.decode("utf-8", errors="ignore")
            else:
                txt = str(txt)

        # SimpleRobertaTokenizer.tokenize_to_text_bert 返回 [L, 3]：
        # 每行 = [input_id, attention_mask, token_type_id]
        tb = extractor.tokenize_to_text_bert(txt)  # [L, 3]
        text_bert_list.append(tb)

        if (i + 1) % 100 == 0:
            print(f"  tokenized {i + 1} samples...")

    # 对变长序列做 padding，堆成 [N, 3, L_max]
    lens = [tb.shape[0] for tb in text_bert_list]
    max_len = max(lens)
    N = len(text_bert_list)

    text_bert = np.zeros((N, 3, max_len), dtype=np.int64)
    for idx, tb in enumerate(text_bert_list):
        L = tb.shape[0]
        # tb: [L, 3] -> [3, L]
        tb_T = tb.T
        text_bert[idx, :, :L] = tb_T

    print(f"  built text_bert: shape = {text_bert.shape} (N={N}, max_len={max_len})")
    return text_bert


def rebuild_pkl_with_roberta(input_pkl: str, output_pkl: str, device: str = "cuda", pretrained: str = "hfl/chinese-roberta-wwm-ext"):
    print(f"Loading original pkl from: {input_pkl}")
    with open(input_pkl, "rb") as f:
        data = pickle.load(f)

    # 深拷贝一份，避免误改原数据
    new_data = deepcopy(data)

    # 当前 SimpleRobertaTokenizer 只在 CPU 上做分词，这里忽略 device 参数
    # pretrained 必须与 experiment_configs.text_encoder_pretrained 一致
    extractor = build_roberta_extractor(pretrained=pretrained)

    for split in ["train", "valid", "test"]:
        if split not in data:
            print(f"[WARN] split '{split}' not found in pkl, skip.")
            continue

        print(f"\nProcessing split: {split}")
        raw_text_list = data[split]["raw_text"]
        text_bert_new = tokenize_split(raw_text_list, extractor)  # [N, 3, L_max]

        # 覆盖/新增 text_bert 字段，其它字段保持不变
        new_data[split]["text_bert"] = text_bert_new.astype("float32")

        print(
            f"  {split}: raw_text num = {len(raw_text_list)}, "
            f"text_bert_new shape = {new_data[split]['text_bert'].shape}"
        )

    print(f"\nSaving new pkl to: {output_pkl}")
    with open(output_pkl, "wb") as f:
        pickle.dump(new_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="从原有 KuDA 文本 raw_text 生成 RoBERTa text_bert，并写入新 pkl"
    )
    parser.add_argument(
        "--input_pkl",
        type=str,
        required=True,
        help="原始 KuDA 数据 pkl 路径（如 unaligned_39.pkl）",
    )
    parser.add_argument(
        "--output_pkl",
        type=str,
        default="",
        help="输出新 pkl 路径（默认在文件名后加 _roberta.pkl）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="运行 RoBERTa 的设备：cuda 或 cpu",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="hfl/chinese-roberta-wwm-ext",
        help="tokenizer 与模型路径，需与 experiment_configs.text_encoder_pretrained 一致",
    )

    args = parser.parse_args()

    input_pkl = os.path.abspath(args.input_pkl)
    if args.output_pkl:
        output_pkl = os.path.abspath(args.output_pkl)
    else:
        root, ext = os.path.splitext(input_pkl)
        output_pkl = root + "_roberta" + ext

    rebuild_pkl_with_roberta(input_pkl=input_pkl, output_pkl=output_pkl, device=args.device, pretrained=args.pretrained)


if __name__ == "__main__":
    main()