# カスタムデータセットクラスの定義
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split
import os
import pycocotools
from PIL import Image

class DataSet(Dataset):
    def __init__(self, data, transform=False):
        self.X = data[0]
        self.y = data[1]
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        img = self.X[index].view(28, 28)
        label = self.y[index]
        if self.transform:
            img = transforms.ToPILImage()(img)
            img = self.transform(img)
        return img, label

def get_mnist_dataloaders(batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = datasets.MNIST(root='VQVAE/data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(root='VQVAE/data', train=False, transform=transforms.ToTensor())

    x_train = train_dataset.data.reshape(-1, 784).float() / 255
    y_train = F.one_hot(train_dataset.targets, 10).float()
    x_test = test_dataset.data.reshape(-1, 784).float() / 255
    y_test = F.one_hot(test_dataset.targets, 10).float()

    trainset = DataSet([x_train, y_train], transform=transform)
    testset = DataSet([x_test, y_test], transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=0)

    return trainloader, testloader

class DataSet(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        sample = [self.data[0][idx], self.data[1][idx]]
        if self.transform:
            # transformはPIL Imageオブジェクトを期待するので、必要に応じて変換します
            # MNISTデータはすでにTensorなので、ここでは変換しません
            pass
        return sample

class SimpleImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f)) and f.lower().endswith(('.jpeg', '.jpg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB') # RGBに変換
        except Exception as e:
            print(f"Error loading image at {img_path}: {e}")
            return None, None  # エラーが発生した場合はNoneを返す

        if self.transform:
            image = self.transform(image)

        # ファイル名からラベルを抽出する場合 (例: 'image_0.jpeg' -> label 0)
        try:
            label = int(img_name.split('_')[1].split('.')[0]) if '_' in img_name and '.' in img_name else -1
        except:
            label = -1 # ラベル抽出に失敗した場合

        return image, label

def get_image_dataloaders_from_folder(data_dir, batch_size, train_ratio=0.8):
    """
    指定されたディレクトリにあるjpeg画像を読み込み、256x256にリサイズして、訓練用とテスト用のデータローダーを作成します。
    データセットは指定された割合で分割されます。

    Args:
        data_dir (str): 画像データが格納されているディレクトリのパス
        batch_size (int): バッチサイズ
        train_ratio (float): 訓練データセットの割合 (0.0から1.0)

    Returns:
        tuple: trainloaderとtestloader
    """
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNetの平均と標準偏差で正規化 (一般的な初期値)
    ])

    full_dataset = SimpleImageDataset(root_dir=data_dir, transform=transform)
    dataset_size = len(full_dataset)
    train_size = int(train_ratio * dataset_size)
    test_size = dataset_size - train_size

    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return trainloader, testloader

# 使用例
if __name__ == '__main__':
    data_directory = '/workspace/inhouse-vqvae/vqvae2/data/coco-2017/validation/data'
    batch_size = 32
    train_loader, test_loader = get_image_dataloaders_from_folder(data_directory, batch_size, train_ratio=0.8)

    # train_loaderの動作確認
    for images, labels in train_loader:
        for img, _ in train_loader:
            print(img.shape)
        break

    # test_loaderの動作確認
    for images, labels in test_loader:
        print("Test batch shape:", images.shape)
        print("Test label shape:", labels.shape)
        break