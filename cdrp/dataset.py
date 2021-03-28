from torch.utils import data
from PIL import Image
from misc import load_pickle
import random


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class SampleDataset(data.Dataset):
    def __init__(self, images_list_path, start_class, end_class, num_per_class, random_order=False, transform=None):
        self.images_list = load_pickle(images_list_path)
        self.transform = transform
        self.targets = []
        self.images = []

        classes = list(range(1000))[start_class: end_class]
        for i in classes:
            self.targets.extend([i] * num_per_class)
            imgs = self.images_list[i]
            if random_order:
                random.shuffle(imgs)
                self.images.extend(imgs[:num_per_class])
            else:
                self.images.extend(imgs[:num_per_class])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        path = self.images[item]
        target = self.targets[item]

        img = pil_loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target
