import torch
import torch.nn.functional as F
from torch import optim, nn
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
import os
from src.model import VQVAE, VQVAE2
from src.trainer import Trainer
from src.data_handler import get_mnist_dataloaders, DataSet, get_image_dataloaders, get_class_specific_dataloaders  # DataSet もこちらで定義
from src.utils import plot_loss
import wandb

@hydra.main(config_name="config.yaml", version_base=None, config_path="/workspace/inhouse-vqvae/VQVAE/config")
def main(cfg: DictConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, name=cfg.wandb.name)

    # --- パラメータ設定 ---
    DATA_DIRECTORY = '/workspace/inhouse-vqvae/VQVAE/data/preprocessed'  # 画像データが入ったディレクトリ
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

    try:
        artifact = run.use_artifact('TGGATE-Recon:latest')
        model_path = artifact.get_entry("savetest.pth").download()
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        
        # モデルの状態辞書が 'param' キーの下に格納されていると仮定してロード
        # もし状態辞書が直接保存されている場合は model.load_state_dict(checkpoint) を使用してください
        model.load_state_dict(checkpoint)
        print(f"モデルを {model_path} から正常にロードしました。")
    except Exception as e:
        print(f"モデルのロード中にエラーが発生しました: {e}")
        return # ロードに失敗した場合は処理を中断

    model.eval()

    # 辞書のitems()を使って、クラス名とデータローダーを順番に取り出す
    with torch.no_grad():
        for class_name, data_loader in class_specific_loaders.items():
            print(f"\n--- \"{class_name}\" クラスの解析を開始 ---")
            codebook_usage_counter = torch.zeros(1, cfg.model.vqvae_n_embeddings, dtype=torch.long, device=device)
            idx_bundle = torch.zeros(64, 256, device=device)
            
            # このループでは、指定したクラスの画像だけが順番に出てくる
            for i, (images, labels) in enumerate(data_loader):
                img = images.to(device, dtype=torch.float)
                *_, idx = model(img)
                
                idx = idx.flatten()
                
                counter = torch.bincount(idx, minlength=cfg.model.vqvae_n_embeddings)
                counter = counter.view(1, -1)
                # codebook_usage_counter += counter
                if i == 0:
                    codebook_usage_counter = counter
                    idx_bundle = idx.view(64, 64, 64)
                elif idx.size()[0] / (64*64) != 64:
                    print(codebook_usage_counter)
                    pass
                else:
                    codebook_usage_counter = torch.cat([codebook_usage_counter, counter], 0)
                    idx_bundle = torch.cat([idx_bundle, idx.view(64, 64, 64)], 0)

                pass
            print(f"--- \"{class_name}\" クラスの解析が完了 ---")

            parent_dir = '/workspace/inhouse-vqvae/VQVAE/model/No-shuffle'
            save_dir = os.path.join(parent_dir, f'sence-loss_no-normal_{cfg.model.vqvae_n_embeddings}')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            torch.save(codebook_usage_counter, os.path.join(save_dir, f'{class_name}.pt'))
            torch.save(idx_bundle, os.path.join(save_dir, f'{class_name}idx.pt'))

if __name__ == "__main__":
    main()