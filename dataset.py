import os
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
from sklearn.model_selection import train_test_split

Labels = {'Egg': 0, 'Fried Food': 1, 'Meat': 2, 'Rice': 3, 'Seafood': 4}


class MyData(data.Dataset):
    def __init__(self, root_dir, classes, transform = None):

        self.root_dir = root_dir
        self.classes = classes
        self.image_list = []

        for cls_index in range(len(self.classes)):
            class_files = [[os.path.join(self.root_dir, self.classes[cls_index], x), cls_index] for x in
                           os.listdir(os.path.join(self.root_dir, self.classes[cls_index]))]
            self.image_list += class_files

        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        path = self.image_list[idx][0]

        image = Image.open(path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = int(self.image_list[idx][1])

        data = {'img': image,
                'label': label,
                'paths': path}

        return data
