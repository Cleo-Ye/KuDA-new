import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


__all__ = ['MMDataLoader']


class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        DATA_MAP = {
            'mosi': self.__init_mosi,
            'mosei': self.__init_mosei,
            'sims': self.__init_sims,
            'simsv2': self.__init_simsv2,
            'external_knowledge': self.__init_external_knowledge
        }
        dataset_key = str(args.datasetName).lower()
        if dataset_key not in DATA_MAP:
            raise KeyError(f"Unknown datasetName '{args.datasetName}'. Supported: {list(DATA_MAP.keys())}")
        DATA_MAP[dataset_key]()

    def __init_mosi(self):
        with open(self.args.dataPath, 'rb') as f:
            data = pickle.load(f)

        self.args.use_bert = True
        self.args.need_truncated = True
        self.args.need_data_aligned = False

        if self.args.use_bert:
            self.text = data[self.mode]['text_bert'].astype(np.float32)
        else:
            self.text = data[self.mode]['text'].astype(np.float32)

        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)

        # 动态同步特征维度，避免 fea_dims 与真实特征维度不一致导致 Linear 输入维度错误
        # 形状约定: text [B, L_t, D_t], vision [B, L_v, D_v], audio [B, L_a, D_a]
        try:
            text_dim = int(self.text.shape[-1])
            vision_dim = int(self.vision.shape[-1])
            audio_dim = int(self.audio.shape[-1])
            self.args.fea_dims = [text_dim, vision_dim, audio_dim]
        except Exception:
            # 若遇到非标准形状（例如某模态不存在），保持原 fea_dims，不中断训练
            pass

        self.rawText = data[self.mode]['raw_text']
        self.ids = data[self.mode]['id']
        self.labels = {
            'M': data[self.mode][self.args.train_mode+'_labels'].astype(np.float32)
        }
        if 'sims' in self.args.datasetName.lower():
            for m in "TAV":
                self.labels[m] = data[self.mode][self.args.train_mode+'_labels_'+m]

        if not self.args.need_data_aligned:
            self.audio_lengths = data[self.mode]['audio_lengths']
            self.vision_lengths = data[self.mode]['vision_lengths']

        # Clear dirty data: remove NaN, +inf, -inf
        self.audio[~np.isfinite(self.audio)] = 0
        self.vision[~np.isfinite(self.vision)] = 0

        # Phase 1: CMVN for both audio and vision
        if self.args.use_cmvn:
            self._apply_audio_cmvn()
            self._apply_vision_cmvn()

        self.__gen_mask(data[self.mode])
        if self.args.need_truncated:
            self.__truncated()

    def __init_mosei(self):
        return self.__init_mosi()

    def __init_sims(self):
        return self.__init_mosi()

    def __init_simsv2(self):
        return self.__init_mosi()

    def __init_external_knowledge(self):
        with open(self.args.dataPath, 'rb') as f:
            data = pickle.load(f)

        if self.args.datasetName in ['mosi', 'mosei']:
            with open('./pretrainedModel/pretrained_text.pkl', 'rb') as f2:
                data_t_en = pickle.load(f2)

        self.text = data[self.mode]['text_bert'].astype(np.float32)

        self.rawText = data[self.mode]['raw_text']

        '''The segmentation of the following dataset is transformed according to MOSI and MOSEI
        '''
        if self.args.datasetName in ['mosi', 'mosei']:
            if self.mode == 'train':
                self.rawText = data_t_en['en'][0:1368]
            elif self.mode == 'valid':
                self.rawText = data_t_en['en'][1368:1824]
            else:
                self.rawText = data_t_en['en'][1824:]

        self.vision = data[self.mode]['vision'].astype(np.float32)
        self.audio = data[self.mode]['audio'].astype(np.float32)

        self.ids = data[self.mode]['id']
        self.labels = {
            'M': data[self.mode][self.args.train_mode+'_labels'].astype(np.float32)
        }
        for m in "TAV":
            self.labels[m] = data[self.mode][self.args.train_mode+'_labels_'+m]

        self.audio_lengths = data[self.mode]['audio_lengths']
        self.vision_lengths = data[self.mode]['vision_lengths']

        # Clear dirty data
        self.audio[self.audio == -np.inf] = 0
        self.vision[self.vision == -np.inf] = 0

        self.__gen_mask(data[self.mode])
        self.__truncated()

    def __truncated(self):
        # NOTE: Here for dataset we manually cut the input into specific length.
        def Truncated(modal_features, length):
            if length == modal_features.shape[1]:
                return modal_features
            truncated_feature = []
            padding = np.array([0 for i in range(modal_features.shape[2])])
            for instance in modal_features:
                for index in range(modal_features.shape[1]):
                    if ((instance[index] == padding).all()):
                        if (index + length >= modal_features.shape[1]):
                            truncated_feature.append(instance[index:index+length])
                            break
                    else:
                        truncated_feature.append(instance[index:index+length])
                        break
            truncated_feature = np.array(truncated_feature)
            return truncated_feature

        text_length, video_length, audio_length = self.args.seq_lens

        self.vision = Truncated(self.vision, video_length)
        self.audio = Truncated(self.audio, audio_length)

    def __gen_mask(self, data):
        vision_tmp = torch.tensor([[True for i in range(data['vision'].shape[1])] for j in range(data['vision'].shape[0])])
        for i in range(len(vision_tmp)):
            vision_tmp[i][0:data['vision_lengths'][i]] = False

        vision_mask = torch.cat((vision_tmp[:, 0:1], vision_tmp), dim=-1)
        for i in range(self.__len__()):
            vision_mask[i][0] = False
        self.vision_padding_mask = vision_mask

        audio_tmp = torch.tensor([[True for i in range(data['audio'].shape[1])] for j in range(data['audio'].shape[0])])
        for i in range(len(audio_tmp)):
            audio_tmp[i][0:data['audio_lengths'][i]] = False

        audio_mask = torch.cat((audio_tmp[:, 0:1], audio_tmp), dim=-1)
        for i in range(self.__len__()):
            audio_mask[i][0] = False
        self.audio_padding_mask = audio_mask

    def __len__(self):
        return len(self.labels['M'])
    
    def _apply_audio_cmvn(self):
        """
        Phase 1: 对音频特征应用CMVN(Cepstral Mean and Variance Normalization)
        对每条样本的有效音频长度做z-score归一化
        """
        for i in range(len(self.audio)):
            valid_len = self.audio_lengths[i]
            if valid_len > 0:
                audio_valid = self.audio[i, :valid_len, :]
                mean = audio_valid.mean(axis=0, keepdims=True)
                std = audio_valid.std(axis=0, keepdims=True) + 1e-8
                self.audio[i, :valid_len, :] = (audio_valid - mean) / std

    def _apply_vision_cmvn(self):
        """
        对视觉特征应用 CMVN：对每条样本的有效帧做 z-score 归一化。
        SIMS 视觉特征维度为 709，数值范围可能非常大，不归一化极易导致 Transformer 溢出。
        """
        for i in range(len(self.vision)):
            valid_len = self.vision_lengths[i]
            if valid_len > 0:
                vis_valid = self.vision[i, :valid_len, :]
                mean = vis_valid.mean(axis=0, keepdims=True)
                std = vis_valid.std(axis=0, keepdims=True) + 1e-8
                normalized = (vis_valid - mean) / std
                # 再 clamp 防止极端离群点
                normalized = np.clip(normalized, -10.0, 10.0)
                self.vision[i, :valid_len, :] = normalized

    def __getitem__(self, index):
        sample = {
            'raw_text': self.rawText[index],
            'text': torch.Tensor(self.text[index]),
            'audio': torch.Tensor(self.audio[index]),
            'vision': torch.Tensor(self.vision[index]),
            'index': index,
            'id': self.ids[index],
            'labels': {k: torch.Tensor(v[index].reshape(-1)) for k, v in self.labels.items()},
            'mask': self.mask[index] if self.mode == 'train_mix' else []
        }
        sample['audio_lengths'] = self.audio_lengths[index]
        sample['vision_lengths'] = self.vision_lengths[index]
        sample['vision_padding_mask'] = self.vision_padding_mask[index]
        sample['audio_padding_mask'] = self.audio_padding_mask[index]
        return sample


def MMDataLoader(args):
    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    dataLoader = {
        ds: DataLoader(
            datasets[ds],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True
        )
        for ds in datasets.keys()
    }

    return dataLoader
