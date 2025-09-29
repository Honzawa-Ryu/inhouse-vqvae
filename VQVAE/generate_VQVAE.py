"""
画像再構成の実例出力用のコード
スライドに載せる用などの画像に使う
"""

import torch
import matplotlib.pyplot as plt
import random  # ランダムな整数を生成するためのライブラリのインポート
from src.model import VQVAE, VQVAE16
from src.data_handler import get_mnist_dataloaders, get_image_dataloaders, get_class_specific_dataloaders
from omegaconf import DictConfig
import hydra


@hydra.main(config_name="config.yaml", version_base=None, config_path="/workspace/inhouse-vqvae/VQVAE/config")
def main(cfg: DictConfig):
    device = 'cuda' if torch.cuda.is_available else 'cpu'

    # 保存されたモデルのファイルパス
    model_path = "/workspace/inhouse-vqvae/VQVAE/model/vqvae/VQVAE16_wd0.1.pth"
    # VQVAEモデルのインスタンスの作成

    model = VQVAE16(**cfg.model)
    # 保存されたモデルのパラメータをロード
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['param'])
    # モデルを適切なデバイス（GPUまたはCPU）に移動
    model = model.to(device)

    # trainloader, testloader = get_mnist_dataloaders(256)
    # trainloader, testloader = get_image_dataloaders('/workspace/inhouse-vqvae/VQVAE/data/hist_class', 1024)
    class_specific_loaders = get_class_specific_dataloaders(data_dir='/workspace/inhouse-vqvae/VQVAE/data/hist_class', batch_size=64, image_size=(256, 256))
    testloader = class_specific_loaders['mouse_brain']
    # テストデータローダーから最初のバッチを取得し、適切なデバイスに移動
    img_batch = next(iter(testloader))[0].to(device)
    print(img_batch.size())

    # バッチからランダムにインデックスを選ぶ
    random_index = random.randint(0, img_batch.size(0) - 1)

    # 選ばれた画像をバッチに変換（次元を追加）
    img = img_batch[random_index].unsqueeze(0)
    # モデルを通じて画像をエンコードし、デコード
    embedding_loss, x_hat, *_ = model(img)
    print(x_hat.size())
    print(img.size())

    # x_hat = x_hat.permute(2, 3, 1, 0)

    # # 出力画像をCPUに移動し、NumPy配列に変換
    # pred = x_hat[0].to('cpu').detach().numpy().reshape(256, 256, 3)
    # # 元の画像をCPUに移動し、NumPy配列に変換
    # origin = img[0].to('cpu').detach().numpy().reshape(256, 256, 3)

    # 出力画像をCPUに移動し、次元を[高さ, 幅, チャンネル]に変更してからNumPy配列に変換
    pred = x_hat[0].to('cpu').detach().permute(1, 2, 0).numpy()

    # 元の画像をCPUに移動し、次元を[高さ, 幅, チャンネル]に変更してからNumPy配列に変換
    origin = img[0].to('cpu').detach().permute(1, 2, 0).numpy()


    # 元の画像を表示
    plt.subplot(211)
    plt.imshow(origin)
    plt.xticks([])  # x軸の目盛りを非表示
    plt.yticks([])  # y軸の目盛りを非表示
    plt.text(x=3, y=2, s="original image", c="red")  # テキストラベルの追加

    # 出力画像を表示
    plt.subplot(212)
    plt.imshow(pred)
    plt.text(x=3, y=2, s="output image", c="red")  # テキストラベルの追加
    plt.xticks([])  # x軸の目盛りを非表示
    plt.yticks([])  # y軸の目盛りを非表示
    plt.savefig("/workspace/inhouse-vqvae/VQVAE/results/generate.png")


if __name__ == "__main__":
    main()