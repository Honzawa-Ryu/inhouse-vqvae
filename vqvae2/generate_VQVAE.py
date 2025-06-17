import torch
import matplotlib.pyplot as plt
import random  # ランダムな整数を生成するためのライブラリのインポート
from src.model import VQVAE2
from src.data_handler import get_mnist_dataloaders


device = 'cuda' if torch.cuda.is_available else 'cpu'

# 保存されたモデルのファイルパス
model_path = "VQVAE_local.pth"
# VQVAEモデルのインスタンスの作成

model = vqvae2 = VQVAE2.build_from_kwargs(
    in_dim=1, hidden_dim=128, codebook_dim=64, codebook_size=512, residual_dim=128, resample_factors=[2, 2]
).to(device=device, dtype=torch.float32)
# 保存されたモデルのパラメータをロード
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['param'])
# モデルを適切なデバイス（GPUまたはCPU）に移動
model = model.to(device)

trainloader, testloader = get_mnist_dataloaders(256)
# テストデータローダーから最初のバッチを取得し、適切なデバイスに移動
img_batch = next(iter(testloader))[0].to(device)

# バッチからランダムにインデックスを選ぶ
random_index = random.randint(0, img_batch.size(0) - 1)

# 選ばれた画像をバッチに変換（次元を追加）
img = img_batch[random_index].unsqueeze(0)
# モデルを通じて画像をエンコードし、デコード
x_hat, _, _, _ = model(img)
# 出力画像をCPUに移動し、NumPy配列に変換
pred = x_hat[0].to('cpu').detach().numpy().reshape(28, 28, 1)
# 元の画像をCPUに移動し、NumPy配列に変換
origin = img[0].to('cpu').detach().numpy().reshape(28, 28, 1)

# 元の画像を表示
plt.subplot(211)
plt.imshow(origin, cmap="gray")
plt.xticks([])  # x軸の目盛りを非表示
plt.yticks([])  # y軸の目盛りを非表示
plt.text(x=3, y=2, s="original image", c="red")  # テキストラベルの追加

# 出力画像を表示
plt.subplot(212)
plt.imshow(pred, cmap="gray")
plt.text(x=3, y=2, s="output image", c="red")  # テキストラベルの追加
plt.xticks([])  # x軸の目盛りを非表示
plt.yticks([])  # y軸の目盛りを非表示
plt.savefig("/workspace/inhouse-vqvae/vqvae2/result/generate.png")