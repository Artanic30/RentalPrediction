from torch.utils.tensorboard import SummaryWriter
from Predict.Models.MLP import MLP
from Predict.Models.Transformer import Transformer
from torch.utils.data import DataLoader
from Predict.utils.Data import HouseDataset
from Predict.utils.Constant import FEATURE_LIST
import torch.nn as nn
from tqdm import tqdm
import json
import torch
import os


class Train:
    def __init__(self, train_path: str, test_path: str, config: dict, trial_name: str,
                 model_path=''):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = trial_name
        self.writer = SummaryWriter(f'runs/{trial_name}')
        self.train_data, self.test_data, feature_size = self.prepare_dataloader(train_path, test_path,
                                                                                config['feature_list'])

        self.create_storage_dirs(config)
        if model_path:
            self.model = torch.load(model_path, map_location=self.device)
        else:
            self.model = MLP(feature_size, config['hidden_layer_size'], 1).to(
                self.device)

        if config['optimizer']['name'] == "Adam":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), betas=(0.9, 0.98),
                                               lr=config['optimizer']['init_lr'],
                                               weight_decay=config['optimizer']['weight_decay'],
                                               eps=1e-09)
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config['optimizer']['init_lr'],
                                             weight_decay=config['optimizer']['weight_decay'],
                                             momentum=config['optimizer']['momentum'])

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, config['restart_begin'])
        self.loss = nn.MSELoss()

    def train(self, epoch_times: int):
        min_eval_loss = 1e4
        for epoch_i in range(epoch_times):
            print('[ Epoch', epoch_i, ']')
            loss_train = self.train_epoch(self.train_data)
            print(f'Train Loss in epoch {epoch_i}: {loss_train}m lr: {self.scheduler.get_last_lr()}')

            loss_test = self.eval_epoch(self.test_data)
            if min_eval_loss > loss_test:
                min_eval_loss = loss_test
                print('new minimum loss found, saving model')
                torch.save(self.model, f'storage/{self.name}/trained_{epoch_i}.pt')
            print(f'Evaluation Loss in epoch {epoch_i}: {loss_test}')

            self.writer.add_scalar(f'Train loss', loss_train, epoch_i)
            self.writer.add_scalar(f'Evaluation loss', loss_test, epoch_i)
            # if epoch_i % 3 == 0:
            #     torch.save(self.model, f'storage/{self.name}/trained_{epoch_i}.pt')
        self.writer.close()

    def train_epoch(self, training_data: DataLoader):
        model = self.model
        optimizer = self.optimizer
        scheduler = self.scheduler
        model.train()
        total_loss = 0

        desc = '  - (Training)   '
        for batch in tqdm(training_data, mininterval=2, desc=desc, leave=False):
            feature, labels = batch
            feature = feature.to(self.device)
            labels = labels.to(self.device)

            pred = model(feature)
            loss = self.loss(pred, labels)
            if pred.sum() == 0:
                print('all zero', end='')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
        return total_loss

    def eval_epoch(self, test_data: DataLoader):
        model = self.model
        model.eval()
        total_loss = 0
        desc = '  - (Evaluation)   '
        for batch in tqdm(test_data, mininterval=2, desc=desc, leave=False):
            feature, labels = batch
            feature = feature.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                pred = model(feature)
                loss = self.loss(pred, labels)
            total_loss += loss.item()

        return total_loss

    @staticmethod
    def prepare_dataloader(train_path: str, test_path: str, feature_list: [str], batch_size=20):
        train_data = HouseDataset(train_path, feature_list)
        test_data = HouseDataset(test_path, feature_list)
        train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        return train_data_loader, test_data_loader, train_data.get_shape()

    def create_storage_dirs(self, config):
        if not os.path.exists(f'./storage'):
            os.mkdir('./storage')
        if not os.path.exists(f'storage/{self.name}'):
            os.mkdir(f'storage/{self.name}')
        with open(f'storage/{self.name}/config.json', 'w') as f:
            f.write(json.dumps(config))


if __name__ == "__main__":
    feature_list = FEATURE_LIST
    feature_list.remove('avg_reco_price')
    feature_list.remove('avg_nearby_price')
    feature_list.remove('geo_lat')
    feature_list.remove('geo_lng')
    hyperParameters = {
        'hidden_layer_size': 256,
        'batch_size': 200,
        'restart_begin': 1000,
        'optimizer': {
            'name': "Adam",
            'init_lr': 1e-3,
            'momentum': 1e-5,
            'weight_decay': 1e-5
        },
        'feature_list': feature_list
    }

    name = f'Attention_{hyperParameters["hidden_layer_size"]}_{hyperParameters["batch_size"]}_{hyperParameters["optimizer"]["name"]}_{len(hyperParameters["feature_list"])}'

    a = Train('../FeatureGeneration/data/feature_train.tsv', '../FeatureGeneration/data/feature_test.tsv',
              hyperParameters, name)
    a.train(20)
