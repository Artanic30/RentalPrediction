from torch.utils.data import Dataset
from PIL import Image
import os


class FileDataset(Dataset):
    def __init__(self, data_root_path, transform=None):
        self.img_dir = os.path.join(data_root_path, "images")
        self.transform = transform

        self.img_dirs = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.img_dirs)

    def __getitem__(self, idx):
        cur_file = os.path.join(self.img_dir, self.img_dirs[idx])
        images_file = os.listdir(cur_file)
        images = []
        for i in images_file:
            if 'æ·åå¾' in i or '户型图' in i or '其他图片' in i:
                continue
            try:
                images.append(self.transform(Image.open(os.path.join(cur_file, i)).convert("RGB")))
            except:
                continue

        return images, cur_file
