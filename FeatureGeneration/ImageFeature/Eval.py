import torch
import FeatureGeneration.ImageFeature.clip as clip
from FeatureGeneration.ImageFeature.Data import FileDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import json


class EvalCLIP:
    def __init__(self, summary_file: str, data_path: str, model_name='ViT-B/32'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.transform = clip.load(model_name, device=self.device, jit=False)

        self.testDataset = DataLoader(
            FileDataset(data_path, transform=self.transform),
            batch_size=1)

        self.summary = None
        self.file_path = summary_file
        with open(summary_file, 'r') as f:
            self.summary = json.loads(f.read())

    def evaluation(self):
        test_set = self.testDataset
        model = self.model
        model.eval()
        score_list = []

        desc = '  - (Evaluating)   '
        descriptions = ['The room is pretty big and glorious', 'The room makes people feel comfortable and relaxed', 'The room is well-lighted and drafty', 'The room is clean and organized', 'The room is modern and well-designed',
                        'The room is small and unexceptional', 'The room makes people feel mournful and terrible', 'The room is dark and windowless.', 'The room is disordered and dusty', 'The room is bare and lifeless']
        score = torch.tensor([1,1,1,1,1,-1,-1,-1,-1,-1]).to(self.device)
        text_features = torch.cat([clip.tokenize(c) for c in descriptions]).to(self.device)
        with torch.no_grad():
            text_features = model.encode_text(text_features)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        idx = 0
        for batch in tqdm(test_set, mininterval=2, desc=desc, leave=False):
            images_features_list, file_name = batch
            file_name = file_name[0].split('/')[-1]
            if file_name in self.summary:
                print('Calculated!')
                continue
            result = torch.zeros(len(descriptions)).to(self.device)
            for images_features in images_features_list:
                images_features = images_features.to(self.device)

                with torch.no_grad():
                    images_features = model.encode_image(images_features)

                    images_features /= images_features.norm(dim=-1, keepdim=True)
                    similarity = (100.0 * images_features @ text_features.T).softmax(dim=-1).to(self.device)

                    result += similarity[0]

                    # # visualize prediction result
                    # values, indices = similarity[0].topk(5)
                    #
                    # print("\nTop predictions:\n")
                    # for value, index in zip(values, indices):
                    #     print(f"{descriptions[index]:>16s}: {100 * value.item():.2f}%")
            result /= len(images_features)
            result *= score
            self.summary[file_name.split('/')[-1]] = result.tolist()
            score_list.append([file_name.split('/')[-1], result.sum().item()])
            # print(score_list[-1])
            if idx % 200 == 0:
                with open(self.file_path, 'w') as f:
                    f.write(json.dumps(self.summary))
                    print('load to file system!')
            idx += 1
        return score_list


if __name__ == "__main__":
    a = EvalCLIP('./summary.json', '../../FetchData/data/20210723/')
    re_list = a.evaluation()
