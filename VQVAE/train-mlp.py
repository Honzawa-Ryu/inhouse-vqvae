"""
MLP学習用のコード
Configが長すぎるのでファイル分けするべき
"""

import torch
import torch.optim as optim
from torchvision import datasets, transforms
from model.model import MLP  # model.pyからMLPクラスをインポート
from model.modelvqvae2 import MLP2
import hydra
from omegaconf import DictConfig
from src.data_handler import get_mnist_dataloaders, DataSet, get_image_dataloaders  # DataSet もこちらで定義
from src.model import VQVAE
from src.utils import init_wandb
import wandb
from schedulefree import RAdamScheduleFree

@hydra.main(config_name="config.yaml", version_base=None, config_path="/workspace/inhouse-vqvae/VQVAE/config")
def main(cfg: DictConfig):
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Wandbの設定
    # wandb.init(config=dict(cfg.model),
    #            entity="benzelongji-the-university-of-tokyo",
    #            project="2025-9-2-vqvae2-mlp",
    #            name='dataset-test')
    init_wandb(cfg)
    
    train_loader, test_loader = get_image_dataloaders('/workspace/inhouse-vqvae/VQVAE/data/hist_class', 64)
    
    # モデル、損失関数、最適化手法の定義
    if cfg.train.VQVAE:
        model = MLP(**cfg.model).to(device)
    else:
        model = MLP2(**cfg.model).to(device)
    if cfg.train.frozen:
        for param in model.vqvae.parameters():
            param.requires_grad = False

    # パラメータ数の記録
    total_params = sum(
	param.numel() for param in model.parameters()
    )
    wandb.config.total_parameters = total_params

    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train.learning_rate)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    optimizer = RAdamScheduleFree(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
 
    # 学習ループ
    for epoch in range(cfg.train.epochs):
        running_loss = 0.0
        class_loss = 0.0
        recon_loss = 0.0
        model.train()
        optimizer.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 勾配をゼロにリセット
            optimizer.zero_grad()

            # 順伝播、誤差計算、逆伝播、パラメータ更新
            outputs, vq_loss = model(inputs)
            mlp_loss = criterion(outputs, labels)
            if cfg.train.multiheadgrad:
                loss = mlp_loss * cfg.train.gamma + vq_loss
            else:
                loss = mlp_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            class_loss += mlp_loss.item()
            recon_loss += vq_loss.item()
        # scheduler.step()

        print(f'Epoch [{epoch+1}/{cfg.train.epochs}], Loss: {running_loss/len(train_loader):.4f}')
        wandb.log({"loss": running_loss/len(train_loader), "mlp_loss": class_loss/len(train_loader), "vq_loss": recon_loss/len(train_loader)})

        model.eval()  # モデルを評価モードに設定
        optimizer.eval()
        correct = 0
        total = 0

        # 推論中は勾配計算を無効にする
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # 予測を行う
                outputs, _ = model(inputs)

                # 確率が最も高いクラスのインデックスを取得
                # `torch.max`は (最大値, 最大値のインデックス) のタプルを返す
                predicted = torch.argmax(outputs.data, 1)
                if labels.dim() == 2 and labels.size(1) > 1:
                    labels = torch.argmax(labels, dim=1)

                # 全サンプルの総数を更新
                total += labels.size(0)

                # 正しく予測できた数を更新
                correct += (predicted == labels).sum().item()

        # 最終的な精度を計算して出力
        accuracy = 100 * correct / total
        print(f"Accuracy: {accuracy:.2f}%")
        wandb.log({"accuracy": accuracy})

        # 学習済みモデルの保存
    if cfg.train.VQVAE:
        torch.save(model.state_dict(), 'mlp_mnist.pth')

    else:
        torch.save(model.state_dict(), 'mlp2_mnist.pth')
    print('Finished Training')
    # wandb.alert(title="Finished training", text="Finished training")


if __name__ == "__main__":
    main()