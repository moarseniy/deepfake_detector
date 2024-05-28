import torch
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import v2


class SymReLU(torch.nn.Module):
    def __init__(self):
        super(SymReLU, self).__init__()

    def forward(self, x):
        return torch.max(torch.tensor([-1]), torch.min(torch.tensor([1]), x))


class ConvAct(torch.nn.Module):
    def __init__(self, in_ch, out_ch, f, s, p):
        super(ConvAct, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, f, stride=s, padding=p)  # padding_mode='reflect')
        self.act = nn.Softsign()

    def forward(self, x):
        return self.act(self.conv(x))


def prepare_augmentation():
    train_transform = v2.Compose(
        [
            v2.RandomHorizontalFlip(0.4),
            v2.RandomVerticalFlip(0.1),
            v2.RandomApply(transforms=[v2.RandomRotation(degrees=(0, 90))], p=0.5),
            v2.RandomApply(transforms=[v2.ColorJitter(brightness=0.3, hue=0.1)], p=0.3),
            v2.RandomApply(transforms=[v2.GaussianBlur(kernel_size=(5, 9))], p=0.3),
            # transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
        ]
    )

    transforms = v2.Compose(
        [
            v2.RandomApply(transforms=[v2.RandomRotation(degrees=(-5, 5), fill=1)], p=0.0),
            v2.RandomApply(transforms=[v2.Compose([
                v2.RandomResize(int(37 * 0.7), int(37 * 0.9)),
                v2.Resize(size=(37, 37))
            ])], p=0.0),
            v2.RandomApply(transforms=[v2.RandomPerspective(0.15, fill=1)], p=1.0)
            # v2.RandomApply(transforms=[v2.functional.perspective(startpoints=[[0, 0], [0, 37], [37, 37], [37, 0]],
            # 													 endpoints=[[0, 0], [0, 37], [uniRand(), 37], [uniRand(), 0]],
            # 													 fill=1)], p=1.0)
        ]
    )

    return transforms


class Meso4(nn.Module):
    def __init__(self, num_classes=2):
        super(Meso4, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.leakyrelu = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(8, 8, 5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(8, 16, 5, padding=2, bias=False)
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2, bias=False)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(4, 4))
        # flatten: x = x.view(x.size(0), -1)
        self.dropout = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(16 * 8 * 8, 16)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, input):
        x = self.conv1(input)  # (8, 256, 256)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling1(x)  # (8, 128, 128)

        x = self.conv2(x)  # (8, 128, 128)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.maxpooling1(x)  # (8, 64, 64)

        x = self.conv3(x)  # (16, 64, 64)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpooling1(x)  # (16, 32, 32)

        x = self.conv4(x)  # (16, 32, 32)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.maxpooling2(x)  # (16, 8, 8)

        x = x.view(x.size(0), -1)  # (Batch, 16*8*8)
        x = self.dropout(x)
        x = self.fc1(x)  # (Batch, 16)
        x = self.leakyrelu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
