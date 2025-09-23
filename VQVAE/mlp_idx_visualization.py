import torch
import torch.nn.functional as F
from torch import optim, nn
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig

from src.model import VQVAE, VQVAE2
from model.model import MLP
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
    model = MLP(**cfg.model).to(device)
    
    model_path = "/workspace/inhouse-vqvae/VQVAE/model/mlp/mlp_mnist_adam25.pth"
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    
    model.eval()

    # 辞書のitems()を使って、クラス名とデータローダーを順番に取り出す
    for class_name, data_loader in class_specific_loaders.items():
        print(f"\n--- \"{class_name}\" クラスの解析を開始 ---")
        codebook_usage_counter = torch.zeros(1, 512, dtype=torch.long, device=device)
        idx_bundle = torch.zeros(64, 256, device=device)
        
        # このループでは、指定したクラスの画像だけが順番に出てくる
        for i, (images, labels) in enumerate(data_loader):
            img = images.to(device, dtype=torch.float)
            *_, idx = model.vqvae(img)
            
            idx = idx.flatten()
            
            counter = torch.bincount(idx, minlength=512)
            counter = counter.view(1, -1)
            # codebook_usage_counter += counter
            if i == 0:
                codebook_usage_counter = counter
                idx_bundle = idx.view(64, 64, 64)
            elif i == 108:
                pass
            else:
                codebook_usage_counter = torch.cat([codebook_usage_counter, counter], 0)
                idx_bundle = torch.cat([idx_bundle, idx.view(64, 64, 64)], 0)
            
            

            pass
            
        print(f"--- \"{class_name}\" クラスの解析が完了 ---")
        torch.save(codebook_usage_counter, f'/workspace/non-shuffle/mlp/{class_name}.pt')
        torch.save(idx_bundle, f'/workspace/non-shuffle/mlp/{class_name}idx.pt')

if __name__ == "__main__":
    main()