from torch.utils.data import DataLoader
from Predict.utils.Data import HouseDataset
from Predict.utils.Constant import FEATURE_LIST
from visualization.utils.DrawNetworkResult import visualize
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import json
import torch
import os


class Train:
    def __init__(self, test_path: str, feature_list: list):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_data = HouseDataset(test_path, feature_list)
        self.test_data_loader = DataLoader(test_data, batch_size=20, shuffle=False)
        self.loss = nn.MSELoss()
        self.store_dir = ''

    def eval(self, model_file_path):
        self.store_dir = f'data/{model_file_path.split("/")[-1]}'
        self.create_storage_dirs(self.store_dir)
        df_list = []
        for f_dir in os.listdir(model_file_path):
            if '.pt' in f_dir:
                df_list.append(self.eval_epoch(os.path.join(model_file_path, f_dir)))
        visualize(df_list, self.store_dir)

    def eval_epoch(self, model_path):
        model = torch.load(model_path, map_location=self.device)
        model.eval()
        desc = '  - (Evaluation)   '
        pred_list = []
        labels_list = []
        for batch in tqdm(self.test_data_loader, mininterval=2, desc=desc, leave=False):
            feature, labels = batch
            feature = feature.to(self.device)
            labels = labels.to(self.device) * 1e4

            with torch.no_grad():
                pred = model(feature) * 1e4
                pred_list.extend(pred.squeeze(-1).tolist())
                labels_list.extend(labels.squeeze(-1).tolist())

        return pd.DataFrame({
            'pred': pred_list,
            'label': labels_list
        })

    def create_storage_dirs(self, name):
        if not os.path.exists(f'./{name}'):
            os.mkdir(f'./{name}')

    @staticmethod
    def prepare_dataloader(train_path: str, test_path: str, feature_list: [str], batch_size=20):
        train_data = HouseDataset(train_path, feature_list)
        test_data = HouseDataset(test_path, feature_list)
        train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        return train_data_loader, test_data_loader, train_data.get_shape()


if __name__ == "__main__":
    feature_list = FEATURE_LIST
    feature_list.remove('avg_reco_price')
    feature_list.remove('avg_nearby_price')
    feature_list.remove('geo_lat')
    feature_list.remove('geo_lng')
    config = {
        'feature_list': feature_list
    }
    a = Train('../FeatureGeneration/data/feature_test.tsv', feature_list)
    a.eval('../Predict/storage/Attention_256_200_SGD_11')
    a.eval('../Predict/storage/Attention_256_200_Adam_11')
    a.eval('../Predict/storage/Attention_Sigmoid_256_200_SGD_11')
    a.eval('../Predict/storage/Attention_Sigmoid_256_200_Adam_11')
    a.eval('../Predict/storage/MLP_Sigmoid_256_200_SGD_11')
    a.eval('../Predict/storage/MLP_Sigmoid_256_200_Adam_11')
    a.eval('../Predict/storage/MLP_256_200_SGD_11')
    a.eval('../Predict/storage/MLP_256_200_Adam_11')
