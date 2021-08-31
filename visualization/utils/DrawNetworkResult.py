import matplotlib.pyplot as plt
import pandas as pd
import math
import os


def visualize(df_list, base_path):
    file_name = os.path.join(base_path, '')
    if not os.path.exists(file_name):
        os.mkdir(file_name)
    for idx, df in enumerate(df_list):
        df.sort_values(by=['label'], inplace=True)

        x = [i for i in range(max(df.count()))]
        plt.plot(x, df['label'].values, label='Label Rental Price')
        plt.plot(x, df['pred'].values, label=f'Pred Rental Price')
        plt.xlabel('Index Ordered by Price')
        plt.ylabel('Price')
        plt.title("Prediction Results")
        plt.ylim([0, 2e4])
        plt.legend()
        plt.savefig(f'{file_name}/model_idx_{idx}.jpg', facecolor='grey', edgecolor='red')
        plt.clf()
    return plt
