from pyecharts import options as opts
from pyecharts.charts import BMap, Geo
import pandas as pd
import json
import os


class Visualize:
    def __init__(self, raw_data_path: str,predict_data_path: str, store_path: str):
        self.predict_data_df = pd.read_csv(predict_data_path, sep='\t')
        self.raw_data_df = pd.read_csv(raw_data_path, sep='\t')
        self.coordinates_file_path, self.square_loss = self.prepare_coordinate_and_loss()
        self.rent_coordinates_file_path, self.rental_price = self.prepare_coordinate_and_rent()
        self.baidu_ak = 'lLBHT3P9THaeGUnni1l1eqdmfiVWHhwb'
        self.store_path = store_path
        if not os.path.exists(store_path):
            os.mkdir(store_path)

        self.draw_predict_loss()
        self.draw_rental_price()

        os.remove(self.coordinates_file_path)
        os.remove(self.rent_coordinates_file_path)

    def prepare_coordinate_and_loss(self):
        coordinate_json = {}
        values_list = []
        json_path = './data/coordinate.json'
        for idx, row in self.predict_data_df.iterrows():
            house_id = row['house_id']
            coordinate_json[house_id] = [row['geo_lng'], row['geo_lat']]
            values_list.append((house_id, row['square_loss']))
        with open(json_path, 'w') as f:
            f.write(json.dumps(coordinate_json))
        return json_path, values_list

    def prepare_coordinate_and_rent(self):
        coordinate_json = {}
        values_list = []
        json_path = './data/coordinate_rent.json'
        for idx, row in self.raw_data_df.iterrows():
            house_id = row['house_id']
            coordinate_json[house_id] = [row['geo_lng'], row['geo_lat']]
            values_list.append((house_id, row['price']))
        with open(json_path, 'w') as f:
            f.write(json.dumps(coordinate_json))
        return json_path, values_list

    def draw_rental_price(self):
        c = (
            BMap(
                init_opts=opts.InitOpts(
                    width="90rem",
                    height="50rem",
                )
            )
                .add_schema(baidu_ak=self.baidu_ak, center=[121.4906643769531, 31.213326336168205],zoom=15
                            )
                .add_coordinate_json(self.rent_coordinates_file_path)
                .add(
                "rental price",
                self.rental_price,
                type_="heatmap",
                label_opts=opts.LabelOpts(formatter="{b}"),
                blur_size=5
            )
                .set_global_opts(
                title_opts=opts.TitleOpts(title="Rental Price-HeatHap"),
                visualmap_opts=opts.VisualMapOpts(
                    pos_bottom='10%',
                    max_=4e4
                )
            )

                .render(os.path.join(self.store_path, "rental_price_heatmap.html"))
        )

    def draw_predict_loss(self):
        c = (
            BMap(
                init_opts=opts.InitOpts(
                    width="90rem",
                    height="50rem",
                )
            )
                .add_schema(baidu_ak=self.baidu_ak, center=[121.4906643769531, 31.213326336168205],zoom=15
                            )
                .add_coordinate_json(self.coordinates_file_path)
                .add(
                "square loss",
                self.square_loss,
                type_="heatmap",
                label_opts=opts.LabelOpts(formatter="{b}"),
                blur_size=5
            )
                .set_global_opts(
                title_opts=opts.TitleOpts(title="square loss between predict and label-HeatHap"),
                visualmap_opts=opts.VisualMapOpts(
                    type_='color',
                    range_color=['#CCEBFF', '#22DDDD', '#0099FF', '#003D66'],
                    is_piecewise=True,
                    pieces=[
                        {'min': 12000 ** 2, "label": '12k - infinity'},
                        {"min": 8000 ** 2, "max": 12000 ** 2, "label": '8k - 12k'},
                        {"min": 4000 ** 2, "max": 8000 ** 2, "label": '4k - 8k'},
                        {"min": 2000 ** 2, "max": 4000 ** 2, "label": '2l - 4k'},
                        {"min": 1000 ** 2, "max": 2000 ** 2, "label": '1k - 2k'},
                        {"min": 0, "max": 1000 ** 2, "label": '0 - 1k'},
                    ],
                    pos_bottom='10%'
                )
            )

                .render(os.path.join(self.store_path, "predict_loss_heatmap.html"))
        )


if __name__ == "__main__":
    a = Visualize('../FeatureGeneration/data/feature_total.tsv', '../Predict/storage/predict_result.tsv', './data/heatmaps')
