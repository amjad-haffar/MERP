import numpy as np
from torch.utils.data import Dataset


class rdm_dataset(Dataset):

    def __init__(self, feat_dict, exps, seq_len=10, seed=42):

        self.random = np.random.RandomState(seed)
        self.seq_len = seq_len

        songlist = list(feat_dict.keys())

        exps = exps[
            exps["songurl"].isin(songlist)
        ].reset_index(drop=True)

        self.feat_dict = feat_dict
        self.exps = exps

    def __getitem__(self, index):

        row = self.exps.iloc[index]

        songurl = row["songurl"]
        audio_feat_full = self.feat_dict[songurl]
        label_full = row["labels"]

        if self.seq_len:
            audio_length = len(audio_feat_full)

            start_idx = self.random.randint(
                audio_length - self.seq_len
            )

            end_idx = start_idx + self.seq_len

            audio_feat = audio_feat_full[
                start_idx:end_idx
            ]

            label = label_full[
                start_idx:end_idx
            ]

        else:
            audio_feat = audio_feat_full
            label = label_full

        return audio_feat, label

    def __len__(self):
        return len(self.exps)