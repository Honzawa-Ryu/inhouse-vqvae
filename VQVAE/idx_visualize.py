import torch
import torch.nn.functional as F
from torch import optim, nn
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig

from src.model import VQVAE, VQVAE2
from src.trainer import Trainer
from src.data_handler import get_mnist_dataloaders, DataSet, get_image_dataloaders, get_class_specific_dataloaders  # DataSet もこちらで定義
from src.utils import plot_loss
import wandb

@hydra.main(config_name="config.yaml", version_base=None, config_path="/workspace/inhouse-vqvae/VQVAE/config")
def main(cfg: DictConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- パラメータ設定 ---
    DATA_DIRECTORY = '/workspace/inhouse-vqvae/VQVAE/data/hist_class'  # 画像データが入ったディレクトリ
    BATCH_SIZE = 64
    IMAGE_SIZE = (256, 256)

    # --- データハンドラを実行 ---
    class_specific_loaders = get_class_specific_dataloaders(
        data_dir=DATA_DIRECTORY,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE
    )

    # --- モデル解析のループ ---
    model = VQVAE(**cfg.model).to(device)
    
    model_path = "/workspace/inhouse-vqvae/VQVAE/model/vqvae/VQVAE_local_adam_25.pth"
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['param'])
    
    model.eval()

    # 辞書のitems()を使って、クラス名とデータローダーを順番に取り出す
    for class_name, data_loader in class_specific_loaders.items():
        print(f"\n--- \"{class_name}\" クラスの解析を開始 ---")
        codebook_usage_counter = torch.zeros(256, dtype=torch.long, device=device)
        
        # このループでは、指定したクラスの画像だけが順番に出てくる
        for i, (images, labels) in enumerate(data_loader):
            img = images.to(device, dtype=torch.float)
            *_, idx = model(img)
            
            idx = idx.flatten()
            counter = torch.bincount(idx, minlength=256)
            codebook_usage_counter += counter


            pass
            
        print(f"--- \"{class_name}\" クラスの解析が完了 ---")
        print(codebook_usage_counter)

if __name__ == "__main__":
    main()