from torch.utils.data import Dataset
from PIL import Image
import os
import glob
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from .utils import RandomGammaVolume, RandomGaussianNoise
import numpy as np

class CustomBinaryClassDataset(Dataset):
    """
    サブディレクトリ構造から画像を読み込み、指定されたフォルダ名に基づいて
    ターゲットインデックス (0 または 1) を割り当てるカスタムデータセット。
    """
    def __init__(self, img_dir: str, class_1_folders: List[str], transform: Optional = None):
        """
        Args:
            img_dir (str): データセットのルートディレクトリのパス。
            class_1_folders (List[int]): クラス 1 (ターゲットインデックス 1) に割り当てる
                                         サブディレクトリ名のリスト（int型で指定）。
            transform (Optional): 画像に適用する変換処理。
        """
        self.img_dir = img_dir
        self.transform = transform
        
        # クラスの定義: クラス 0 と クラス 1 の 2 種類
        # class_to_idx は内部的なマッピングですが、ここでは固定で定義
        self.class_to_idx: Dict[str, int] = {'Class_0': 0, 'Class_1': 1}
        self.classes: List[str] = ['Class_0', 'Class_1']
        
        # ターゲットインデックス 1 に割り当てるフォルダ名のリスト（文字列に変換）
        # os.path.join()などで使用されることを考慮し、文字列型に変換して比較
        self.class_1_folder_names = class_1_folders
        
        # 全ての画像ファイルとそのターゲットインデックスを格納するリスト
        self.samples: List[Tuple[str, int]] = self._make_dataset()

    def _make_dataset(self) -> List[Tuple[str, int]]:
        """
        ルートディレクトリをスキャンし、画像パスとターゲットインデックスのリストを生成します。
        """
        samples: List[Tuple[str, int]] = []
        
        # ルートディレクトリ内のエントリをスキャン
        for entry in os.scandir(self.img_dir):
            if entry.is_dir():
                folder_name = entry.name
                folder_path = entry.path
                
                # フォルダ名が class_1_folder_names リストに含まれるかどうかでターゲットを決定
                if folder_name in self.class_1_folder_names:
                    target = 1  # クラス 1 に割り当て
                else:
                    target = 0  # クラス 0 に割り当て
                
                # フォルダ内の全ての画像ファイルを取得
                # 拡張子は一般的なものを想定 (例: *.jpg, *.png)
                # glob.globで再帰的に検索する場合は '**/*' を使用し、recursive=Trueが必要
                # ここでは、直下のファイルのみを検索
                # search_pattern = os.path.join(folder_path, "**", "*.jpeg")
                # image_paths = glob.glob(search_pattern, recursive=True)
                for file_path in glob.glob(os.path.join(folder_path, '**/*'), recursive=True):
                    if os.path.isfile(file_path):
                        # (画像パス, ターゲットインデックス) のタプルをリストに追加
                        samples.append((file_path, target))
                        
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        # self.samplesからファイルパスとターゲットインデックスを取得
        img_path, target = self.samples[idx]
        
        # 画像の読み込み (RGBに変換)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"警告: 画像ファイル {img_path} の読み込みに失敗しました。スキップします。エラー: {e}")
            # エラー処理: ダミーの画像を返すか、例外をスローするか、ここではスキップするために別の画像を返すロジックを考える必要があります。
            # 実際には、_make_datasetで不正なファイルをフィルタリングするのが望ましいです。
            # シンプルにするため、ここでは読み込みが失敗した場合に例外をスローします。
            raise IOError(f"Failed to load image at {img_path}") from e


        # 変換処理の適用
        if self.transform:
            image = self.transform(image)
            
        return image, target

# --- 使い方（仮想的な例） ---
"""
# 仮想的なディレクトリ構造:
# data_root/
# ├── 1/    <-- クラス 1 (ターゲット 1)
# ├── 4/    <-- クラス 0 (ターゲット 0)
# ├── 5/    <-- クラス 1 (ターゲット 1)
# └── 89/   <-- クラス 1 (ターゲット 1)

# クラス 1 にしたいフォルダ名のリスト (int型で指定)
target_class_1_folders = ['63431', '63433', '63435', '63438', '63442', '63446', '63450', '63466', '63473', '63477', '63481', '63485', '63489', '63494', '63497', '63501', '63505', '63509', '63513', '63517', '63522', '63526', '63530', '63534', '63538', '63542', '63546', '63550', '63553', '63558', '63562', '63566', '63571', '63575', '63579', '63583', '63587', '63591', '63595', '63598', '63602', '63605', '63609', '63612', '63616', '63620', '63623', '63665', '63669', '63672', '63676', '63680', '63748', '63752', '63756', '63831', '63937', '63684', '63687', '63691', '70929', '63698', '63760', '63763', '63768', '63772', '63777', '63855', '63860', '63865', '71285', '63875', '63941', '63944', '63948', '63952', '63955', '27622', '27624', '27626', '27628', '27630', '27632', '27634', '27636', '27638', '27640', '27642', '27644', '27646', '27651', '27655', '27657', '27660', '27662', '27663', '27665', '27668', '27670', '27672', '27674', '27676', '27678', '27680', '27682', '27684', '27686', '27689', '46720', '46727', '46736', '46740', '46744', '46747', '46750', '46753', '46756', '46758', '46759', '46763', '46765', '46768', '46769', '46772', '46774', '46778', '46781', '46783', '46787', '46789', '46792', '46795', '46797', '46800', '46803', '46806', '46809', '46811', '46815', '46817', '46820', '46823', '46826', '46829', '46831', '46833', '46835', '46839', '46841', '46844']

# # データセットの初期化
# custom_dataset = CustomBinaryClassDataset(
#     img_dir='/path/to/data_root',
#     class_1_folders=target_class_1_folders,
#     transform=your_transforms
# )

# # 確認
# print(f"合計画像数: {len(custom_dataset)}")
# print(f"クラス情報: {custom_dataset.class_to_idx}")
# print(f"最初の5つのサンプル (パス, ターゲット): {custom_dataset.samples[:5]}")
"""

def idx_dataloaders(data_dir, class_1_folders, batch_size, train_val_split=0.8, image_size=(256, 256), sampling_rate=None):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        RandomGammaVolume(p=0.5),
        RandomGaussianNoise(p=0.5)
        # transforms.Normalize((0.8259633779525757, 0.4840644896030426, 0.6278038620948792), (0.12393593788146973, 0.19072337448596954, 0.15796850621700287))
    ])

    full_dataset = CustomBinaryClassDataset(img_dir=data_dir, class_1_folders=class_1_folders, transform=transform)
    print(f"クラス情報: {full_dataset.class_to_idx}")
    print(f"元の合計画像数: {len(full_dataset)}")
    print(f"最初の5つのサンプル (パス, ターゲット): {full_dataset.samples[:5]}")

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