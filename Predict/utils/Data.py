from torch.utils.data import Dataset
import pandas as pd
import json
import torch


class HouseDataset(Dataset):
    def __init__(self, df_path: str, feature_list: [str]):
        self.source_df = pd.read_csv(df_path, sep='\t')
        self.selected_feature = feature_list

    def get_shape(self):
        row = self.source_df.iloc[0]
        re_list = []
        for f in self.selected_feature:
            re_list += json.loads(row[f])
        return len(re_list)

    def __len__(self):
        return max(self.source_df.count())

    def __getitem__(self, idx):
        row = self.source_df.iloc[idx]
        re_list = []
        for f in self.selected_feature:
            re_list += json.loads(row[f])
        price = row['price'] / 1e4
        return torch.tensor(re_list), torch.tensor([price], dtype=torch.float32)
