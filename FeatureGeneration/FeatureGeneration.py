import pandas as pd
from sklearn.model_selection import train_test_split
import requests
import torch
import json
import math
import os
import re


class FeatureGeneration:
    def __init__(self, data_path):
        self.raw_data = self.collect_data(data_path)
        self.baidu_ak = 'EQ9YNKnRlrjeMWNEK8RGbX0HH1GMCbfo'

        self.init_columns = self.raw_data.columns
        self.raw_data = self.extract_geo_info(self.raw_data)

        self.raw_data = self.generate_feature(self.raw_data)

        self.train, self.test = train_test_split(self.raw_data, train_size=0.9)
        self.raw_data.to_csv('./data/feature_total.tsv', sep='\t')
        self.train.to_csv('./data/feature_train.tsv', sep='\t')
        self.test.to_csv('./data/feature_test.tsv', sep='\t')

    def extract_geo_info(self, raw_data):
        if 'geo_lat' in raw_data.columns and 'geo_lng' in raw_data.columns:
            print('geo_info is loaded')
            return raw_data
        ids = raw_data['house_id'].values
        geo_house_id = {}
        for i_list in self.raw_data['recommend_house_id_geo'].values:
            i_list = json.loads(i_list)
            for h_id, geo in i_list:
                if h_id in geo_house_id and geo_house_id[h_id] != geo:
                    raise ValueError('Geo info mismatch')
                if h_id not in ids:
                    # print(h_id)
                    continue
                geo_house_id[h_id] = geo

        left_ids = list(set(ids).difference(set(geo_house_id.keys())))

        for l_ids in left_ids:
            location = raw_data.loc[l_ids]['xiaoqu'].split('·')[-1]
            data = {}
            iter_time = 0
            while 'result' not in data:
                iter_time += 1
                request_url = f'http://api.map.baidu.com/geocoding/v3/?address={location}&output=json&ak={self.baidu_ak}&city=上海市'
                res = requests.get(request_url)
                data = res.json()
                if iter_time > 10:
                    raise ConnectionError
            location = data["result"]["location"]
            geo_info = f'{location["lat"]},{location["lng"]}'
            geo_house_id[l_ids] = geo_info

        left_ids = list(set(ids).difference(set(geo_house_id.keys())))
        if len(left_ids) != 0:
            raise ValueError('Unprocessed Geo info exist')

        raw_data['geo_lat'] = ['' for i in range(max(raw_data.count()))]
        raw_data['geo_lng'] = ['' for i in range(max(raw_data.count()))]

        def fill_geo_info(row):
            row.loc['geo_lat'] = float(geo_house_id[row['house_id']].split(',')[0])
            row.loc['geo_lng'] = float(geo_house_id[row['house_id']].split(',')[1])
            return row

        print('geo info append at **geo_info**')
        raw_data = raw_data.apply(fill_geo_info, axis=1)
        raw_data.to_csv('./data/total.tsv', sep='\t')
        print('update total.tsv file')
        return raw_data.apply(fill_geo_info, axis=1)

    def generate_feature(self, data):
        agent_num_house = {}
        list_agents = data['agent_name'].values.tolist()
        for name in data['agent_name'].unique():
            agent_num_house[name] = list_agents.count(name)

        min_lat = data['geo_lat'].min()
        min_lng = data['geo_lng'].min()

        with open('./ImageFeature/summary.json', 'r') as f:
            images_feature_dict = json.loads(f.read())

        images_feature_avg = []
        for i in images_feature_dict.values():
            images_feature_avg.append(i)
        images_feature_avg = torch.tensor(images_feature_avg)
        images_feature_avg = images_feature_avg.mean(axis=0).tolist()

        def gen_feature(row):
            room, hall = re.findall('\d+', row['layout'])
            row.loc['layout_feature'] = [int(room), int(hall)]
            row.loc['size_feature'] = [float(re.sub('平米', '', row['size']))]
            # [is_west, is_east, is_south, is_north, is_checkin_anytime, is_high,
            # is_medium, is_low, height, elevator, is_park_free, is_park_unknown, is_park_rental, is_water_commercial
            # is_elect_commercial, has_gas]
            row.loc['image_num_feature'] = [math.log(len(json.loads(row['relative_image_path'])) + 1, 1.5)]
            house_info_feature = []
            house_info_dict = json.loads(row['house_info_dict'])
            direction = house_info_dict['朝向']
            for di in list('东西南北'):
                house_info_feature.append(1 if di in direction else 0)
            house_info_feature.append(1 if house_info_dict['入住'] == '随时入住' else 0)
            building = house_info_dict['楼层']
            for he in list('高中低'):
                house_info_feature.append(1 if he in building else 0)
            house_info_feature.append(int(re.findall('\d+', building)[0]))
            house_info_feature.append(1 if house_info_dict['电梯'] == '有' else 0)
            for park in ['租用', '暂无', '免费']:
                house_info_feature.append(1 if park in house_info_dict['车位'] else 0)
            house_info_feature.append(1 if house_info_dict['用水'] == '商水' else 0)
            house_info_feature.append(1 if house_info_dict['用电'] == '商电' else 0)
            house_info_feature.append(1 if house_info_dict['采暖'] == '自采暖' else 0)

            row.loc['house_info_feature'] = house_info_feature

            # ['洗衣机', '空调', '衣柜', '电视', '冰箱', '热水器', '床', '暖气', '宽带', '天然气']
            facility_info_feature = []
            facility_info_dict = json.loads(row['facility_info_dict'])
            for val in facility_info_dict.values():
                facility_info_feature.append(1 if val else 0)

            row.loc['facility_info_feature'] = facility_info_feature

            # [how_detail_is_desc, has_market, has_hospital]
            house_description = re.sub(' ', '', row['house_description'])
            house_description_feature = [house_description.count('【'), 1 if '广场' in house_description else 0,
                                         1 if '医院' in house_description else 0]

            row.loc['house_description_feature'] = house_description_feature
            row.loc['agent_feature'] = [agent_num_house[row['agent_name']]]

            recommend_house_list = json.loads(row['recommend_house_id_geo'])
            avg_reco_price = 0
            counter = 0
            for h_id, _ in recommend_house_list:
                if h_id in data.index:
                    avg_reco_price += data.loc[h_id]['price']
                    counter += 1
            if counter:
                avg_reco_price /= counter
            row.loc['avg_reco_price'] = [avg_reco_price]

            nearby_house_list = json.loads(row['nearby_house_id'])
            avg_nearby_price = 0
            counter = 0
            for h_id in nearby_house_list:
                if h_id in data.index:
                    avg_nearby_price += data.loc[h_id]['price']
                    counter += 1
            if counter:
                avg_nearby_price /= counter
            row.loc['avg_nearby_price'] = [avg_nearby_price]

            tag_str = row['tag_list']
            tag_feature = []
            full_tags = ['必看好房', '随时看房', '业主自荐', '精装', '新上', '双卫生间', '近地铁', '押一付一', '官方核验', '月租']
            for t in full_tags:
                tag_feature.append(1 if t in tag_str else 0)

            row.loc['tag_feature'] = tag_feature
            row.loc['geo_feature'] = [(row['geo_lat'] - min_lat) * 111, (row['geo_lng'] - min_lng) * 111]

            sub_info = json.loads(row['subway_info'])
            min_dis = 3000
            subway_lines = 0
            for dis in sub_info.values():
                dis2subway = re.findall(r'\d+', dis)[0]
                min_dis = min(min_dis, int(dis2subway))
                subway_lines += 1
            row['subway_feature'] = [min_dis / 100, subway_lines]
            if row['house_id'] in images_feature_dict:
                row['image_feature'] = images_feature_dict[row['house_id']]
            else:
                row['image_feature'] = images_feature_avg
            return row

        return data.apply(gen_feature, axis=1)

    @staticmethod
    def data_cleaning(tar_df: pd.DataFrame) -> pd.DataFrame:
        ori_num = max(tar_df.count())
        tar_df.drop_duplicates(subset=['house_id'], inplace=True)
        rm_idx = []
        for idx, row in tar_df.iterrows():
            if not re.findall(r'\d+', row['size']):
                rm_idx.append(idx)
                continue
            subway_info = json.loads(row['subway_info'])
            for dis in subway_info.values():
                if not re.findall(r'\d+', dis):
                    rm_idx.append(idx)
                    break
        tar_df.drop(index=rm_idx, inplace=True)
        cur_num = max(tar_df.count())
        print(f'clean {ori_num - cur_num} data!')
        return tar_df

    def collect_data(self, data_path: str) -> pd.DataFrame:
        if os.path.exists('./data/total.tsv'):
            print('Load file from storage!')
            df = pd.read_csv('./data/total.tsv', sep='\t')
            df.index = df['house_id']
            return df

        tsv_dirs = os.listdir(data_path)
        total_df = pd.DataFrame()
        for f_name in tsv_dirs:
            if '.tsv' not in f_name:
                continue
            file_path = os.path.join(data_path, f_name)
            total_df = total_df.append(pd.read_csv(file_path, sep='\t'))

        total_df.index = total_df['house_id']

        total_df = self.data_cleaning(total_df)
        total_df.to_csv('./data/total.tsv', sep='\t')

        print('Data collect and saved!')

        return total_df


if __name__ == "__main__":
    f = FeatureGeneration('../FetchData/data/20210723')
