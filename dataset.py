import os.path

from torch.utils.data import Dataset
from PIL import Image

class TumorDataset(Dataset):
    def __init__(self, root, transform=None, train = True):
        super(TumorDataset, self).__init__()

        self.root = root
        self.transform = transform
        self.input_paths = []
        self.output_paths = []

        sub_paths_images = os.path.join('tumor-segmentation-datasets', 'images')
        sub_paths_labels = os.path.join('tumor-segmentation-datasets', 'images')

        for i in os.listdir(os.path.join(root, sub_paths_images)):
            self.input_paths.append(os.path.join(root, sub_paths_images, i))
            self.output_paths.append(os.path.join(root, sub_paths_labels, i))


    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, index):
        image_X = Image.open(self.input_paths[index])
        image_Y = Image.open(self.output_paths[index])

        if self.transform is not None:
            image_X = self.transform(image_X)
            image_Y = self.transform(image_Y)

        return image_X, image_Y


