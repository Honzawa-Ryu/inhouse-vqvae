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
import torch.nn as nn
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
    artifact = wandb.run.use_artifact(f"{cfg.wandb.artifact_name}:v5")
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
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        # RandomGammaVolume(p=0.5),
        # RandomGaussianNoise(p=0.5)
        transforms.Normalize((0.8259633779525757, 0.4840644896030426, 0.6278038620948792), (0.12393593788146973, 0.19072337448596954, 0.15796850621700287))
    ])
    
    # root_dir = "/workspace/02_inhouse-vqvae/VQVAE/data/preprocessed_patches/benign"
    target_dir = "/workspace/02_inhouse-vqvae/VQVAE/data/preprocessed_patches/malignant/59195/slice_0"
    # all_target_dirs = glob.glob(os.path.join(root_dir, "**/slice_*"), recursive=True)
    # for target_dir in tqdm(all_target_dirs):
    dir_parts = target_dir.split(os.sep)
    case_id = f"{dir_parts[-2]}_{dir_parts[-1]}"
    print(f"Processing folder: {target_dir}")
    loader = DataLoader(ImageInferenceDataset(str(target_dir), transform=transform), 
                    batch_size=1, shuffle=False)
    results = {}
    
    with torch.no_grad():
        for images, paths in tqdm(loader):
            images = images.to(device)
            outputs, x_hat, *_ = model(images)  # 出力は (batch_size, 1) などのテンソル
            # outputs = nn.functional.mse_loss(x_hat, images, reduction='none')
            
            # 値をリスト化して、パスと紐付け
            preds = outputs.cpu().numpy().flatten()
            for path, pred in zip(paths, preds):
                results[os.path.basename(path)] = float(pred)
    print("完了しました。ヒートマップを生成します。")
    import numpy as np
    import matplotlib.pyplot as plt

    # 1. ファイル名でソートしてスコアの配列を作る（取得順を再現）
    sorted_keys = sorted(results.keys())
    scores = np.array([results[k] for k in sorted_keys])
    # 平均値を引いて中心を0にする（デトレンド）
    scores_detrended = scores - np.mean(scores)

    # もし全体的に右肩上がり/下がりなら、線形トレンドも引くとより綺麗になります
    # from scipy import signal
    # scores_detrended = signal.detrend(scores)

    # その後、再度FFTを実行
    fft_values_detrended = np.fft.fft(scores_detrended)
    frequencies = np.fft.fftfreq(len(scores_detrended))

    # 例：特定の周波数成分をカットする（目視で特定したピーク周辺を消す）
    # あるいは単純に高周波をカットする「ローパスフィルタ」でも縞模様は消えます
    fft_filtered = fft_values_detrended.copy()

    # もし「高周波の細かいガタつき」を消したいだけなら
    cutoff = 0.01  # 閾値はスペクトルを見て調整
    fft_filtered[np.abs(frequencies) < cutoff] = 0

        # スペクトルを表示（正の周波数のみ）
    plt.figure(figsize=(12, 4))
    plt.plot(frequencies[:len(scores)//2], np.abs(fft_filtered)[:len(scores)//2])
    plt.title("Frequency Spectrum of Results")
    plt.xlabel("Frequency")
    plt.savefig("/workspace/after.png")
    plt.close()
    print("フーリエ変換を実行します。")

        # 逆フーリエ変換
    cleaned_scores = np.fft.ifft(fft_filtered).real

    # 辞書に戻す
    filtered_results = {k: v for k, v in zip(sorted_keys, cleaned_scores)}

    # 推論結果の辞書 results を渡して実行
    visualize_heatmap(filtered_results, output_path=f"/workspace/02_inhouse-vqvae/VQVAE/data/anomaly_images/malignant/Blur_anomaly_heatmap_{case_id}.png")

if __name__ == "__main__":
    main()
