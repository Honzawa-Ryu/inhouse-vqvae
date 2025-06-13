import torch
import torch.optim as optim
from src.model import VQVAE2
from src.data_handler import get_image_dataloaders_from_folder
from src.trainer import train_epoch, create_optimizer

# ハイパーパラメータ (config として渡すことを想定)
config = {
    'data_dir': './data',
    'batch_size': 64,
    'learning_rate': 1e-4,
    'num_epochs': 10,
    'in_dim': 1,         # MNIST はモノクロなので入力チャンネルは 1
    'hidden_dim': 64,
    'codebook_dim': 32,
    'codebook_size': 256,
    'residual_dim': 32,
    'resample_factor': 4
}

def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # データローダーの準備
    data_directory = '/workspace/inhouse-vqvae/vqvae2/data/coco-2017/validation/data'
    batch_size = 32
    train_loader = get_image_dataloaders_from_folder(data_directory, batch_size, train_ratio=0.8)

    # モデルのインスタンス化
    model = vqvae2 = VQVAE2.build_from_kwargs(
        in_dim=3, hidden_dim=128, codebook_dim=64, codebook_size=512, residual_dim=128, resample_factors=[4, 2, 2]
    ).to(device=device, dtype=torch.float32)

    # オプティマイザの作成
    optimizer = create_optimizer(model, learning_rate=config['learning_rate'])

    # 学習ループ
    for epoch in range(config['num_epochs']):
        print(f"Epoch [{epoch+1}/{config['num_epochs']}]")
        train_epoch(model, train_loader, optimizer, device)

    # 学習済みモデルの保存 (例)
    torch.save(model.state_dict(), 'vqvae_mnist.pth')
    print("Training finished. Model saved to vqvae_mnist.pth")

if __name__ == '__main__':
    main(config)