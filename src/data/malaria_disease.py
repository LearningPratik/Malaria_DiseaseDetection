from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def get_preprocessor(size=(160, 160)):

        return transforms.Compose(
            [
                transforms.Resize(size=size),
                transforms.ToTensor(),
            ]
        )


class MalariaDisease(Dataset):
    def __init__(self, img_fps, labels, preprocess=transforms.ToTensor()):
        self.img_fps = img_fps
        self.labels = labels
        self.preprocess = preprocess

    def __len__(self):
        return len(self.img_fps)

    def __getitem__(self, idx):
        img_fp = self.img_fps[idx]

        img = Image.open(img_fp).convert("RGB")
        label = float(self.labels[idx])

        if self.preprocess:
            img = self.preprocess(img)

        return img, label, img_fp