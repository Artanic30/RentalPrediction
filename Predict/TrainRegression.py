from Predict.utils.Constant import FEATURE_LIST
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge
import sklearn.metrics.pairwise as pairwise
from sklearn.tree import DecisionTreeRegressor
from mlxtend.feature_selection import ExhaustiveFeatureSelector, SequentialFeatureSelector
from sklearn.svm import SVR
import pandas as pd
import numpy as np
import json
import os


class Train:
    def __init__(self, train_path: str, test_path: str, config: dict, summary_path: str, select_path: str):
        self.feature_list = config['feature_list']
        self.summary_path = summary_path
        self.train_x, self.train_y = self.prepare_data(train_path)
        self.test_x, self.test_y = self.prepare_data(test_path)
        self.test_df = pd.read_csv(test_path, sep='\t')

        self.save_or_load_summary(self.summary_path)

        # self.run()
        # self.predict()
        # self.feature_select()
        print(self.predict_select())
        split_loss = []
        list_idx = range(5, 65, 5)
        for i in list_idx:
            self.train_x, self.train_y = self.prepare_data(train_path, select_path, i)
            self.test_x, self.test_y = self.prepare_data(test_path, select_path, i)
            loss = 0
            for j in range(10):
                loss += self.predict_select()
                print(f'feature {i}_{j} loss: {loss}')
            loss /= 10
            print(f'feature {i} loss: {loss}')

            split_loss.append(loss)
        re_df = pd.DataFrame(data=zip(list_idx, split_loss))
        re_df.to_csv('./storage/selection_predict_result.csv')

        # self.save_or_load_summary(self.summary_path, is_load=False)

    def predict_select(self):
        reg = DecisionTreeRegressor()
        reg.fit(self.train_x, self.train_y)
        return self.reg_eval(reg)
        # self.test_df.to_csv('./storage/predict_result.tsv', sep='\t')

    def feature_select(self):
        # reg = ExhaustiveFeatureSelector(KernelRidge(alpha=0.5, kernel='laplacian'),
        #                                 min_features=10,
        #                                 max_features=58,
        #                                 print_progress=True,
        #                                 cv=5,
        #                                 scoring='neg_mean_squared_error')
        reg = SequentialFeatureSelector(DecisionTreeRegressor(random_state=30),
                                        k_features='best',
                                        forward=True,
                                        cv=4,
                                        n_jobs=1,
                                        scoring='neg_mean_squared_error'
                                        )

        feature_names = ['layout_feature_room', 'layout_feature_hall', 'size_feature', 'image_num_feature',
                         'house_info_dict_east', 'house_info_dict_west', 'house_info_dict_south',
                         'house_info_dict_north', 'house_info_dict_checkin', 'house_info_dict_high',
                         'house_info_dict_medium', 'house_info_low', 'house_info_dict_floor',
                         'house_info_dict_elevator', 'house_info_dict_park_rent', 'house_info_dict_park_no',
                         'house_info_dict_park_free', 'house_info_dict_water', 'house_info_dict_elect',
                         'house_info_dict_heat', 'facility_info_feature_1', 'facility_info_feature_2',
                         'facility_info_feature_3', 'facility_info_feature_4', 'facility_info_feature_5',
                         'facility_info_feature_6', 'facility_info_feature_7', 'facility_info_feature_8',
                         'facility_info_feature_9', 'facility_info_feature_10', 'house_description_1',
                         'house_description_2', 'house_description_3', 'agent_feature', 'tag_list_1', 'tag_list_2',
                         'tag_list_3', 'tag_list_4', 'tag_list_5', 'tag_list_6', 'tag_list_7', 'tag_list_8',
                         'tag_list_9', 'tag_list_10', 'geo_feature_1', 'geo_feature_2', 'subway_feature_dis',
                         'subway_feature_num', 'image_feature_1', 'image_feature_2', 'image_feature_3',
                         'image_feature_4', 'image_feature_5', 'image_feature_6', 'image_feature_7', 'image_feature_8',
                         'image_feature_9', 'image_feature_10']

        reg = reg.fit(self.train_x, self.train_y, custom_feature_names=feature_names)
        subset = reg.subsets_
        rank = []
        for idx, value in subset.items():
            rank.append(list(set(value['feature_idx']).difference(set(rank)))[0])

        rank_names = []
        for i in rank:
            rank_names.append([feature_names[i], i])

        re_df = pd.DataFrame(data=dict(zip([i for i in range(len(rank_names))], rank_names)))
        re_df.to_csv('./storage/selection_result.tsv')

    def predict(self):
        reg = KernelRidge(alpha=0.5, kernel='laplacian')
        reg.fit(self.train_x, self.train_y)

        pred = reg.predict(self.test_x) * 1e4
        square_loss = (pred - self.test_y * 1e4) ** 2
        print(self.reg_eval(reg))
        self.test_df['pred'] = pred
        self.test_df['label'] = self.test_y * 1e4
        self.test_df['square_loss'] = square_loss
        self.test_df.to_csv('./storage/predict_result.tsv', sep='\t')

    def reg_eval(self, reg):
        reg.fit(self.train_x, self.train_y)
        pred = reg.predict(self.test_x)
        return mean_squared_error(pred, self.test_y)

    def save_or_load_summary(self, path, is_load=True):
        if is_load:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    self.summary = json.loads(f.read())
            else:
                return {}
        else:
            with open(path, 'w') as f:
                f.write(json.dumps(self.summary))

    def run(self):
        self.summary['KernelRidge_loss'] = {}
        self.summary['SVR_loss'] = {}
        self.summary['decision_tree'] = self.decision_tree()
        print(f'decision_tree_loss: {self.summary["decision_tree"]}')

        for kernel in pairwise.PAIRWISE_KERNEL_FUNCTIONS:
            for al in [0.5, 1.0, 1.5]:
                try:
                    KR_loss = self.kernel_ridge(al, kernel)
                    print(f'KG_{al}_{kernel}_loss: {KR_loss}')
                    self.summary['KernelRidge_loss'][f'{al}_{kernel}'] = KR_loss
                except ValueError:
                    print(f'KG_{al}_{kernel}_loss is not feasible')
                try:
                    SVR_loss = self.svr(al, kernel)
                    print(f'SVR_{al}_{kernel}_loss: {SVR_loss}')
                    self.summary['SVR_loss'][f'{al}_{kernel}'] = SVR_loss
                except ValueError:
                    print(f'SVR_{al}_{kernel}_loss is not feasible')

    def kernel_ridge(self, alpha, kernel):
        reg = KernelRidge(alpha=alpha, kernel=kernel)
        return self.reg_eval(reg)

    def svr(self, c, kernel):
        reg = SVR(C=c, kernel=kernel)
        return self.reg_eval(reg)

    def decision_tree(self):
        reg = DecisionTreeRegressor(random_state=0)
        return self.reg_eval(reg)

    def prepare_data(self, path: str, select_path='', k_feature=100):
        data_df = pd.read_csv(path, sep='\t')
        labels = data_df['price'].values / 1e4

        select_feature_idx = None
        if select_path:
            select_df = pd.read_csv(select_path)
            select_feature_idx = select_df.iloc[1].values

        def concat_feature(row):
            feature = []
            for f in self.feature_list:
                feature += json.loads(row[f])
            if type(select_feature_idx) == np.ndarray:
                tmp_feature = []
                for i in select_feature_idx[1:k_feature]:
                    tmp_feature.append(feature[int(i)])
                feature = tmp_feature

            row.loc['feature_collection'] = feature
            return row

        feature = np.array(data_df.apply(concat_feature, axis=1)['feature_collection'].values.tolist())
        return feature, labels


if __name__ == "__main__":
    feature_list = FEATURE_LIST
    feature_list.remove('avg_reco_price')
    feature_list.remove('avg_nearby_price')
    feature_list.remove('geo_lat')
    feature_list.remove('geo_lng')
    # feature_list.remove('image_feature')
    hyperParameters = {
        'feature_list': feature_list
    }
    a = Train('../FeatureGeneration/data/feature_train.tsv', '../FeatureGeneration/data/feature_test.tsv',
              hyperParameters, 'storage/regression_result.json', './storage/selection_result_tree.tsv')
