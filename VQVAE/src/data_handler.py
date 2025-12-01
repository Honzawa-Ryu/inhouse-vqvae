# カスタムデータセットクラスの定義
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import numpy as np
from .utils import RandomGammaVolume, RandomGaussianNoise

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


def get_image_dataloaders(data_dir, batch_size, train_val_split=0.8, image_size=(256, 256)):
    """
    指定されたディレクトリから画像データを読み込み、訓練用と検証用のデータローダーを返す関数。

    Args:
        data_dir (str): 画像データが格納されている親ディレクトリのパス。
                        例: 'data/'
                        このディレクトリの直下にクラスごとのサブディレクトリがあることを想定。
                        （例: 'data/class_A', 'data/class_B', ...）
        batch_size (int): バッチサイズ。
        train_val_split (float): データセット全体のうち、訓練データとして使用する割合。
                                 残りは検証データとなる。
        image_size (tuple): リサイズ後の画像サイズ (高さ, 幅)。

    Returns:
        tuple: (train_loader, val_loader) のタプル。
    """
    # 1. 画像の前処理を定義
    # 画像をリサイズし、Tensorに変換後、[-1, 1]の範囲に正規化する
    transform = transforms.Compose([
        transforms.Resize(image_size),
        # transforms.CenterCrop(image_size),
        # 特定領域の切り出しにするなら後者、画像サイズ見て考える
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 3チャンネル画像の場合
        # グレースケール画像の場合はこちらを使用:
        # transforms.Grayscale(num_output_channels=1),
        # transforms.Normalize((0.5,), (0.5,))
    ])

    # 2. ImageFolderを使用してデータセットを読み込み
    # data_dir内のサブディレクトリ名が自動的にクラスラベルになる
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    print(f"クラス情報: {full_dataset.class_to_idx}")
    print(f"合計画像数: {len(full_dataset)}")


    # 3. データセットを訓練用と検証用に分割
    # 訓練用のデータ数を計算
    train_size = int(train_val_split * len(full_dataset))
    # 検証用のデータ数を計算
    val_size = len(full_dataset) - train_size

    # torch.manual_seedで乱数を固定し、再現性を確保することも可能
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"訓練データ数: {len(train_dataset)}")
    print(f"検証データ数: {len(val_dataset)}")


    # 4. データローダーを作成
    # 訓練用データローダーは、データをシャッフルして供給する
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,      # データをシャッフルする
        num_workers=32,     # データ読み込みを高速化するためのワーカー数
        pin_memory=True
    )

    # 検証用データローダーは、シャッフルは不要
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=32,
        pin_memory=True
    )

    return train_loader, val_loader

def idx_dataloaders(data_dir, batch_size, train_val_split=0.8, image_size=(256, 256), sampling_rate=None):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        # RandomGammaVolume(p=0.5),
        # RandomGaussianNoise(p=0.5)
        # transforms.Normalize((0.8259633779525757, 0.4840644896030426, 0.6278038620948792), (0.12393593788146973, 0.19072337448596954, 0.15796850621700287))
    ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    print(f"クラス情報: {full_dataset.class_to_idx}")
    print(f"元の合計画像数: {len(full_dataset)}")

    if sampling_rate is not None:
        if not (0.0 < sampling_rate <= 1.0):
            raise ValueError("Sampling_rateは0.0より大きく1.0以下の値でなければなりません。")
        
        num_samples = int(len(full_dataset) * sampling_rate)
        
        torch.manual_seed(42)
        indices = torch.randperm(len(full_dataset))[:num_samples]
        
        sampled_dataset = Subset(full_dataset, indices)
        
        print(f"サンプリング適用後 ({sampling_rate * 100}%)")
        print(f"  -> サンプリング後の合計画像数: {len(sampled_dataset)}")
        dataset_to_split = sampled_dataset
    else:
        dataset_to_split = full_dataset

    # 3. データセットを訓練用と検証用に分割
    # 訓練用のデータ数を計算
    train_size = int(train_val_split * len(dataset_to_split))
    # 検証用のデータ数を計算
    val_size = len(dataset_to_split) - train_size

    # torch.manual_seedで乱数を固定し、再現性を確保することも可能
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(dataset_to_split, [train_size, val_size])

    print(f"訓練データ数: {len(train_dataset)}")
    print(f"検証データ数: {len(val_dataset)}")

    data_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=32,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=32,
        pin_memory=True
    )
    return data_loader, val_loader

def get_class_specific_dataloaders(data_dir, batch_size, image_size=(256, 256)):
    """
    指定されたディレクトリから画像データを読み込み、クラスごとに分割された
    データローダーの辞書を返す関数。モデル解析向け。

    Args:
        data_dir (str): 画像データが格納されている親ディレクトリのパス。
                        （例: 'data/class_A', 'data/class_B', ...）
        batch_size (int): バッチサイズ。
        image_size (tuple): リサイズ後の画像サイズ (高さ, 幅)。

    Returns:
        dict: {class_name: DataLoader} の形式の辞書。
              各DataLoaderはそのクラスの画像データのみを供給する。
    """
    # 1. 画像の前処理を定義 (元のコードと同じ)
    # モデルの学習時と同じ前処理を適用することが重要
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        RandomGammaVolume(p=0.5),
        RandomGaussianNoise(p=0.5)
        # transforms.Normalize((0.8259633779525757, 0.4840644896030426, 0.6278038620948792), (0.12393593788146973, 0.19072337448596954, 0.15796850621700287)) # 3チャンネル画像,ここに入れる！
    ])

    # 2. ImageFolderを使用してデータセット全体を一度に読み込み
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    print(f"クラス情報: {full_dataset.class_to_idx}")
    print(f"合計画像数: {len(full_dataset)}")

    # 3. クラスごとのDataLoaderを格納する辞書を初期化
    class_loaders = {}
    torch.manual_seed(42)
    # 4. 各クラスについてループ処理
    # .classes属性でクラス名のリストを取得
    for class_name in full_dataset.classes:
        # クラス名からクラスのインデックス（ラベル）を取得
        class_idx = full_dataset.class_to_idx[class_name]

        # データセット全体から、現在のクラスに属するデータのインデックスをすべて見つける
        # full_dataset.targets には、全画像のラベルがリストとして格納されている
        indices = np.where(np.array(full_dataset.targets) == class_idx)[0]
        
        # Subsetクラスを使い、特定のインデックスのデータだけを持つサブデータセットを作成
        subset = Subset(full_dataset, indices)
        
        print(f"クラス '{class_name}' のデータ数: {len(subset)}")

        # このクラス専用のDataLoaderを作成
        # 解析が目的なので、シャッフル(shuffle=False)は不要
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,  # 一部抜け落ちる構成になりそうなのでシャッフルをします、データ数がバッチサイズの整数倍になってくれていないためです
            num_workers=32,  # 環境に合わせて調整してください
            pin_memory=True
        )

        # 辞書にクラス名と対応するDataLoaderを格納
        class_loaders[class_name] = loader

    return class_loaders

