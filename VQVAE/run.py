"""
VQVAE学習用のコード
Configから詳細引っ張ってこれるようにするべき
"""

import torch
import torch.nn.functional as F
from torch import optim, nn
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig

from src.model import VQVAE, VQVAE2
from src.trainer import Trainer
from src.data_handler import get_mnist_dataloaders, DataSet, get_image_dataloaders, ImageOnlyDataset  # DataSet もこちらで定義
from src.utils import plot_loss
import wandb
from schedulefree import RAdamScheduleFree

# Start a new wandb run to track this script.
@hydra.main(config_name="config.yaml", version_base=None, config_path="/workspace/inhouse-vqvae/VQVAE/config")
def main(cfg: DictConfig):
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="benzelongji-the-university-of-tokyo",
        # Set the wandb project where this run will be logged.
        project="250610_VQVAE_Test",
        # Configからとってくるようにする、何なら全部
        config={
            "learning_rate": 3e-4,
            "architecture": "VQVAE",
            "dataset": "MNIST",
            "epochs": 6,
            "BSZ": 256,
        },
    )


    # 設定
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = cfg.train.batch_size
    max_epoch = cfg.train.epochs
    learning_rate = cfg.train.learning_rate

    # データローダーの取得
    # trainloader, testloader = get_mnist_dataloaders(batch_size)
    trainloader, testloader = get_image_dataloaders('/workspace/inhouse-vqvae/VQVAE/data/hist_class', batch_size)

    # モデルの初期化
    if cfg.train.VQVAE:
        model = VQVAE(**cfg.model).to(device)
    else:
        model = VQVAE2(**cfg.model).to(device)
    # opt = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.9))
    opt = RAdamScheduleFree(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)

    # Trainer の初期化
    trainer = Trainer(model, opt, device, trainloader, testloader, max_epoch, run)

    # 学習の実行
    train_loss_log, test_loss_log = trainer.train()

    # 結果のプロット
    plot_loss(train_loss_log, test_loss_log)

    # 最終エポックでモデルを保存 (Trainer クラス内で行うことも可能です)
    if cfg.train.VQVAE:
        torch.save({'param': model.to('cpu').state_dict(),
                    'opt': opt.state_dict(),
                    'epoch': trainer.epoch},
                    'VQVAE_local.pth')
    else:
        torch.save({'param': model.to('cpu').state_dict(),
                    'opt': opt.state_dict(),
                    'epoch': trainer.epoch},
                    'VQVAE2_local.pth')
    

if __name__ == "__main__":
    main()