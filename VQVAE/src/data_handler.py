# カスタムデータセットクラスの定義
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split

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
    # torch.manual_seed(42)
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

class ImageOnlyDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        # root_dir内のすべての画像ファイルのパスをリストに格納
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                for fname in os.listdir(subdir_path):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                        self.image_paths.append(os.path.join(subdir_path, fname))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 指定されたインデックスの画像パスを取得
        img_path = self.image_paths[idx]
        
        # 画像をPillowで読み込む
        image = Image.open(img_path).convert('RGB') # 適切なチャンネル数に変換
        
        # 前処理を適用
        if self.transform:
            image = self.transform(image)
            
        # ラベルは返さず、画像のみを返す
        return image


# --- 使用例 ---
if __name__ == '__main__':
    # 'data' ディレクトリに 'cat', 'dog' というサブディレクトリがあり、
    # それぞれにPNG画像が保存されていると仮定
    DATA_DIRECTORY = 'data'
    BATCH_SIZE = 32

    # (事前に 'data/cat' と 'data/dog' ディレクトリを作成し、画像を配置しておく必要があります)
    # 例:
    # data/
    # ├── cat/
    # │   ├── 001.png
    # │   └── 002.png
    # └── dog/
    #     ├── 001.png
    #     └── 002.png

    # ダミーのディレクトリとファイルを作成して動作確認
    import os
    if not os.path.exists('data/cat'): os.makedirs('data/cat')
    if not os.path.exists('data/dog'): os.makedirs('data/dog')
    from PIL import Image
    for i in range(10):
        Image.new('RGB', (100, 100)).save(f'data/cat/cat_{i}.png')
        Image.new('RGB', (100, 100)).save(f'data/dog/dog_{i}.png')


    try:
        train_loader, val_loader = get_image_dataloaders(
            data_dir=DATA_DIRECTORY,
            batch_size=BATCH_SIZE
        )

        # データローダーから1バッチ分のデータを取得してみる
        images, labels = next(iter(train_loader))

        print(f"\n取得したバッチの形状:")
        print(f"  画像のテンソル形状: {images.shape}") # -> [batch_size, channels, height, width]
        print(f"  ラベルのテンソル形状: {labels.shape}")   # -> [batch_size]
        print(f"  ラベルの例: {labels[:5]}")

    except FileNotFoundError:
        print(f"エラー: '{DATA_DIRECTORY}' ディレクトリが見つかりません。")
        print("使用例に記載のディレクトリ構造に従って画像データを配置してください。")