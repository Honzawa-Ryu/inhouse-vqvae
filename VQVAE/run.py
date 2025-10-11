"""
VQVAE学習用のコード
Configから詳細引っ張ってこれるようにするべき
"""

import torch
import torch.nn.functional as F
from torch import optim, nn
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig, OmegaConf
import os

from src.model import VQVAE, VQVAE2, VQVAE16
from src.trainer import Trainer
from src.data_handler import get_mnist_dataloaders, DataSet, get_image_dataloaders  # DataSet もこちらで定義
from src.utils import plot_loss
import wandb

# Start a new wandb run to track this script.
@hydra.main(config_name="config.yaml", version_base=None, config_path="/workspace/inhouse-vqvae/VQVAE/config")
def main(cfg: DictConfig):
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )

    # 設定
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = cfg.train.batch_size
    max_epoch = cfg.train.epochs
    learning_rate = cfg.train.learning_rate

    # データローダーの取得
    # trainloader, testloader = get_mnist_dataloaders(batch_size)
    trainloader, testloader = get_image_dataloaders(cfg.data.data_root, batch_size,)

    # モデルの初期化
    if cfg.train.VQVAE:
        model = VQVAE(**cfg.model).to(device)
    else:
        model = VQVAE2(**cfg.model).to(device)
    opt = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.9))

    # Trainer の初期化
    trainer = Trainer(model, opt, device, trainloader, testloader, max_epoch, run)

    # 学習の実行
    trainer.train()

    save_directory = '/workspace/inhouse-vqvae/VQVAE/model/vqvae'
    save_path = os.path.join(save_directory, cfg.model.save_name)

    # 最終エポックでモデルを保存 (Trainer クラス内で行うことも可能です)
    if cfg.train.VQVAE:
        # torch.save({'param': model.to('cpu').state_dict(),
        #             'opt': opt.state_dict(),
        #             'epoch': trainer.epoch},
        #             save_path)
        torch.save(model.state_dict(), save_path)
        OmegaConf.save(config=cfg, f='/workspace/inhouse-vqvae/VQVAE/results/train_log/config_wandb.yaml')
        artifact = wandb.Artifact(name=cfg.wandb.artifact_name, metadata=dict(cfg.model), type='model')
        artifact.add_file(save_path)
        artifact.add_file('/workspace/inhouse-vqvae/VQVAE/results/train_log/config_wandb.yaml')
        wandb.log_artifact(artifact)
    else:
        torch.save({'param': model.to('cpu').state_dict(),
                    'opt': opt.state_dict(),
                    'epoch': trainer.epoch},
                    '/workspace/inhouse-vqvae/VQVAE/model/vqvae/VQVAE2_local.pth')
    

if __name__ == "__main__":
    main()