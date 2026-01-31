""" 
EmbLossによる異常検知ができるか試す
モデルを呼び出す
スライドの全パッチについてLossを計算、算出する
それをもとにヒートマップで可視化する
といった方針
"""

from glob import glob
import hydra
import torchvision.transforms as transforms
from omegaconf import DictConfig, OmegaConf
import torch
import wandb
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import os
from src.model import VQVAE
import torchvision
from tqdm import tqdm
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob



@hydra.main(config_name="config.yaml", version_base=None, config_path="/workspace/02_inhouse-vqvae/VQVAE/config")
def main(cfg: DictConfig):
    import os
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )
    # --- モデルの準備 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQVAE(**cfg.model).to(device)
    
    # --- 学習済みモデルのロード ---
    artifact = wandb.run.use_artifact(f"{cfg.wandb.artifact_name}:latest")
    model_path = artifact.get_entry("savetest.pth").download()
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()

    class ImageInferenceDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            # ディレクトリ内の画像ファイル一覧を取得
            self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, img_path
    
    def visualize_heatmap(results, output_path="anomaly_heatmap.png"):
        """
        results: { "patch_1024_2048.png": 0.012, ... } 形式の辞書
        """
        data = []

        # 1. ファイル名から x, y を抽出してリスト化
        for filename, score in results.items():
            # "patch_120_350.png" -> ['120', '350']
            nums = re.findall(r'\d+', filename)
            if len(nums) >= 2:
                y, x = int(nums[0]), int(nums[1])
                data.append({'x': x, 'y': y, 'score': score})

        if not data:
            print("座標を抽出できませんでした。ファイル名を確認してください。")
            return

        # 2. 座標のユニーク値を取得してソート（インデックス作成用）
        all_x = sorted(list(set(d['x'] for d in data)))
        all_y = sorted(list(set(d['y'] for d in data)))
        
        # 座標値をインデックス(0, 1, 2...)に変換するための辞書
        x_map = {val: i for i, val in enumerate(all_x)}
        y_map = {val: i for i, val in enumerate(all_y)}

        # 3. グリッド配列を作成 (yの数, xの数)
        # 値がない場所を区別するため、np.nan で初期化
        grid = np.full((len(all_y), len(all_x)), np.nan)

        # 4. データをグリッドに配置
        for d in data:
            grid[y_map[d['y']], x_map[d['x']]] = d['score']

        # 5. 可視化
        plt.figure(figsize=(20, 10))
        # cmapは 'jet' や 'viridis', 'magma' などが異常検知には見やすいです
        sns.heatmap(grid, cmap='magma', annot=False, square=True, cbar_kws={'label': 'Anomaly Score'},
                    xticklabels=False, yticklabels=False)
        
        plt.title(f"VQ-VAE Anomaly Score Heatmap from {output_path}")
        plt.xlabel(f"X axis ({len(all_x)} patches)")
        plt.ylabel(f"Y axis ({len(all_y)} patches)")
        plt.savefig(output_path)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        # RandomGammaVolume(p=0.5),
        # RandomGaussianNoise(p=0.5)
        transforms.Normalize((0.8259633779525757, 0.4840644896030426, 0.6278038620948792), (0.12393593788146973, 0.19072337448596954, 0.15796850621700287))
    ])
    
    root_dir = "/workspace/02_inhouse-vqvae/VQVAE/data/preprocessed_patches/benign"
    all_target_dirs = glob.glob(os.path.join(root_dir, "**/slice_*"), recursive=True)
    for target_dir in tqdm(all_target_dirs):
        dir_parts = target_dir.split(os.sep)
        case_id = f"{dir_parts[-2]}_{dir_parts[-1]}"
        print(f"Processing folder: {target_dir}")
        loader = DataLoader(ImageInferenceDataset(str(target_dir), transform=transform), 
                        batch_size=1, shuffle=False)
        results = {}
        
        with torch.no_grad():
            for images, paths in tqdm(loader):
                images = images.to(device)
                outputs, *_ = model(images)  # 出力は (batch_size, 1) などのテンソル
                
                # 値をリスト化して、パスと紐付け
                preds = outputs.cpu().numpy().flatten()
                for path, pred in zip(paths, preds):
                    results[os.path.basename(path)] = float(pred)


        # 推論結果の辞書 results を渡して実行
        visualize_heatmap(results, output_path=f"/workspace/02_inhouse-vqvae/VQVAE/data/anomaly_images/benign/anomaly_heatmap_{case_id}.png")

if __name__ == "__main__":
    main()
