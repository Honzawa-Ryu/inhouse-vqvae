import torch
import torch.nn.functional as F
from torch import optim, nn
import matplotlib.pyplot as plt
from schedulefree import RAdamScheduleFree

from src.model import VQVAE2
from src.trainer import Trainer
from src.data_handler import get_image_dataloaders_from_folder, DataSet  # DataSet もこちらで定義
# from src.utils import plot_loss
import wandb

# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="benzelongji-the-university-of-tokyo",
    # Set the wandb project where this run will be logged.
    project="250613_VQVAE2_enb-rate",
    # Track hyperparameters and run metadata.
    name="bizou-reerror",
    config={
        "learning_rate": 3e-4,
        "architecture": "VQVAE",
        "dataset": "Coco",
        "epochs": 0,
        "BSZ": 32,
    },
)


# 設定
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 32
max_epoch = 9
learning_rate = 3e-4

# データローダーの取得
data_directory = '/workspace/inhouse-vqvae/vqvae2/data/coco-2017/validation/data'
trainloader, testloader = get_image_dataloaders_from_folder(data_directory, batch_size, train_ratio=0.8)


# モデルの初期化
model = vqvae2 = VQVAE2.build_from_kwargs(
    in_dim=3, hidden_dim=128, codebook_dim=64, codebook_size=512, residual_dim=128, resample_factors=[4, 2, 2]
).to(device=device, dtype=torch.float32)

# opt = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.9))
opt = RAdamScheduleFree(model.parameters(), lr=learning_rate)

# Trainer の初期化
trainer = Trainer(model, opt, device, trainloader, testloader, max_epoch, run)

# 学習の実行
train_loss_log, test_loss_log = trainer.train()

# 結果のプロット
# plot_loss(train_loss_log, test_loss_log)

# 最終エポックでモデルを保存 (Trainer クラス内で行うことも可能です)
torch.save({'param': model.to('cpu').state_dict(),
            'opt': opt.state_dict(),
            'epoch': trainer.epoch},
           'VQVAE_local.pth')