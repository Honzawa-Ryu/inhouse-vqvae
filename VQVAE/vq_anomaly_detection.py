import os
import re
import glob
import numpy as np
import torch
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
import seaborn as sns
import matplotlib.pyplot as plt
from src.model import VQVAE


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
        nums = re.findall(r'\d+', filename)
        if len(nums) >= 2:
            # 元のコードに合わせて 0番目をy, 1番目をxとして取得
            y, x = int(nums[0]), int(nums[1])
            data.append({'x': x, 'y': y, 'score': score})

    if not data:
        print("座標を抽出できませんでした。ファイル名を確認してください。")
        return

    # 2. 座標のユニーク値を取得してソート
    all_x = sorted(list(set(d['x'] for d in data)))
    all_y = sorted(list(set(d['y'] for d in data)))
    
    x_map = {val: i for i, val in enumerate(all_x)}
    y_map = {val: i for i, val in enumerate(all_y)}

    # 3. グリッド配列を作成
    grid = np.full((len(all_y), len(all_x)), np.nan)

    # 4. データをグリッドに配置
    for d in data:
        grid[y_map[d['y']], x_map[d['x']]] = d['score']

    # # --- 追加：列単位の周期成分除去 (Median Subtracting) ---
    # # axis=0 で行方向を集計（つまり各列の統計量）を算出
    # # nanが含まれている可能性があるため np.nanmedian を使用
    # col_medians = np.nanmedian(grid, axis=0)
    
    # # 各列の中央値を差し引く
    # # ブロードキャストにより (y_size, x_size) - (x_size,) が行われます
    # grid = grid - col_medians
    # # --------------------------------------------------

    # 5. 可視化
    plt.figure(figsize=(20, 10))
    # 中央値を引いた後は値が負になることもあるため、
    # 異常値を目立たせる場合は vmin=0 などでクリップするか、中央が0のカラーマップ（'RdBu_r'など）も検討してください
    sns.heatmap(grid, cmap='magma', annot=False, square=True, 
                cbar_kws={'label': 'Corrected Anomaly Score'},
                xticklabels=False, yticklabels=False)
    
    plt.title(f"VQ-VAE Anomaly Score Heatmap (Column-wise Median Subtracted)")
    plt.xlabel(f"X axis ({len(all_x)} patches)")
    plt.ylabel(f"Y axis ({len(all_y)} patches)")
    plt.savefig(output_path)
    plt.close() # メモリ解放のためにclose推奨

# --- 既存のクラス・関数定義（変更なし） ---
# ImageInferenceDataset, visualize_heatmap などはそのまま利用

@hydra.main(config_name="config.yaml", version_base=None, config_path="/workspace/02_inhouse-vqvae/VQVAE/config")
def main(cfg: DictConfig):
    # 1. 共通の初期化（ループの外で1回だけ）
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQVAE(**cfg.model).to(device)
    
    # 学習済みモデルのロード
    artifact = run.use_artifact(f"{cfg.wandb.artifact_name}:latest")
    model_path = artifact.get_entry("savetest.pth").download()
    checkpoint = torch.load(model_path, weights_only=True, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.8259633779525757, 0.4840644896030426, 0.6278038620948792), 
                             (0.12393593788146973, 0.19072337448596954, 0.15796850621700287))
    ])

    # 2. ターゲットディレクトリの走査
    root_search_path = "/workspace/02_inhouse-vqvae/VQVAE/data/preprocessed_patches/malignant"
    target_dirs = []
    for filename in os.listdir(root_search_path):
        base_path = os.path.join(root_search_path, filename)
        if os.path.isdir(base_path):
            for subfolder in os.listdir(base_path):
                target_dirs.append(os.path.join(base_path, subfolder))

    # 3. ループ実行
    for target_dir in tqdm(target_dirs, desc="Total Slides"):
        dir_parts = target_dir.split(os.sep)
        case_id = f"{dir_parts[-2]}_{dir_parts[-1]}"
        
        loader = DataLoader(ImageInferenceDataset(str(target_dir), transform=transform), 
                            batch_size=1, shuffle=False)
        results = {}
        
        with torch.no_grad():
            for images, paths in loader:
                images = images.to(device)
                outputs, x_hat, *_ = model(images)
                preds = outputs.cpu().numpy().flatten()
                for path, pred in zip(paths, preds):
                    results[os.path.basename(path)] = float(pred)

        # ヒートマップ生成
        output_filename = f"clear_anomaly_heatmap_{case_id}.png"
        output_path = os.path.join("/workspace/02_inhouse-vqvae/VQVAE/data/anomaly_images/malignant_clear/", output_filename)
        
        visualize_heatmap(results, output_path=output_path)
        
        # WandBに画像をアップロード（1つのRunの中に画像が溜まっていく）
        wandb.log({f"heatmap/{case_id}": wandb.Image(output_path)})

    run.finish()

if __name__ == "__main__":
    main()