"""
PixelCNN (VQ-VAE Stage 2) 学習用のコード
"""

import torch
import torch.optim as optim
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import os

# 既存のプロジェクト構造に合わせるためのimport
# ※ src.model に GatedPixelCNN が定義されている前提です
from src.data_handler import idx_dataloaders
from src.model import VQVAE, GatedPixelCNN 

@hydra.main(config_name="config.yaml", version_base=None, config_path="/workspace/02_inhouse-vqvae/VQVAE/config")
def main(cfg: DictConfig):
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Wandbの設定
    run = wandb.init(
        entity=cfg.wandb.entity, 
        project=cfg.wandb.project, 
        name="pixelcnn-training", 
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )
    
    # データローダーの取得 (提示コードと同様)
    # 必要に応じて class_1_folders を設定してください
    train_loader, test_loader = idx_dataloaders(
        cfg.data.data_root, 
        batch_size=cfg.train.batch_size, 
        sampling_rate=cfg.data.sampling_rate
    )
    
    # -------------------------------------------------------
    # 1. 学習済みVQ-VAEモデルのロードと固定
    # -------------------------------------------------------
    print("Loading VQ-VAE model...")
    vqvae_model = VQVAE(**cfg.model).to(device)

    artifact = run.use_artifact('TGGATE-Recon:v1')
    model_path = artifact.get_entry("savetest.pth").download()
    checkpoint = torch.load(model_path, weights_only=True)
    vqvae_model.load_state_dict(checkpoint)
    
    # VQ-VAEを評価モードにし、全パラメータを凍結 (勾配計算させない)
    vqvae_model.eval()
    for param in vqvae_model.parameters():
        param.requires_grad = False

    # -------------------------------------------------------
    # 2. PixelCNNモデルの定義
    # -------------------------------------------------------
    print("Initializing PixelCNN model...")
    pixelcnn_model = GatedPixelCNN(
        num_embeddings=cfg.model.vqvae_n_embeddings,
        hidden_dim=64,
        n_layers=15
    ).to(device)

    # 損失関数と最適化手法
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(pixelcnn_model.parameters(), lr=cfg.train.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=cfg.train.gamma)

    # パラメータ数の記録
    total_params = sum(p.numel() for p in pixelcnn_model.parameters())
    wandb.config.total_parameters = total_params
    print(f"Total Parameters: {total_params}")

    # -------------------------------------------------------
    # 3. 学習ループ
    # -------------------------------------------------------
    for epoch in range(cfg.train.epochs):
        pixelcnn_model.train()
        running_loss = 0.0
        correct = 0
        total_tokens = 0

        for i, data in enumerate(train_loader):
            inputs, _ = data # ラベルは不要
            inputs = inputs.to(device)

            # --- Step A: VQ-VAEでインデックスを作成 (教師データ) ---
            with torch.no_grad():
                # Encoderを通す -> z
                z = vqvae_model.encoder(inputs) 
                # Quantizerを通す -> indices (実装に合わせて調整してください)
                # 一般的な実装では quantizer の戻り値の最後に indices があります
                # 例: loss, quantized, perplexity, encodings, encoding_indices
                z = vqvae_model.pre_quantization_conv(z)
                
                _, _, _, encoding_indices = vqvae_model.vector_quantization(z)
                
                # [Batch, H, W] の形に整える
                # ※ encoding_indicesがフラット([B*H*W, 1])な場合があるのでreshape
                h, w = z.shape[2], z.shape[3]
                target_indices = encoding_indices.view(inputs.size(0), h, w)

            # --- Step B: PixelCNNの学習 ---
            optimizer.zero_grad()

            # 入力: 正解のインデックス (Teacher Forcing)
            outputs = pixelcnn_model(target_indices) # -> [B, CodebookSize, H, W]

            # 損失計算 (CrossEntropy)
            loss = criterion(outputs, target_indices)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 精度計算 (Next Token Accuracy)
            pred_tokens = torch.argmax(outputs, dim=1)
            correct += (pred_tokens == target_indices).sum().item()
            total_tokens += target_indices.numel()

        # epochごとのログ
        avg_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total_tokens
        print(f'Epoch [{epoch+1}/{cfg.train.epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%')
        wandb.log({"loss": avg_loss, "train_accuracy": train_acc})
        
        scheduler.step()

        # -------------------------------------------------------
        # 4. 評価ループ (Test Set)
        # -------------------------------------------------------
        pixelcnn_model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for data in test_loader:
                inputs, _ = data
                inputs = inputs.to(device)

                # VQ-VAEでインデックス化
                z = vqvae_model.encoder(inputs)
                z = vqvae_model.pre_quantization_conv(z)
                _, _, _, encoding_indices = vqvae_model.vector_quantization(z)
                h, w = z.shape[2], z.shape[3]
                target_indices = encoding_indices.view(inputs.size(0), h, w)

                # 推論
                outputs = pixelcnn_model(target_indices)
                loss = criterion(outputs, target_indices)
                test_loss += loss.item()

                # 精度
                pred_tokens = torch.argmax(outputs, dim=1)
                test_correct += (pred_tokens == target_indices).sum().item()
                test_total += target_indices.numel()

        test_acc = 100 * test_correct / test_total
        avg_test_loss = test_loss / len(test_loader)
        
        print(f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        wandb.log({"test_loss": avg_test_loss, "test_accuracy": test_acc})

    # -------------------------------------------------------
    # 5. モデルの保存
    # -------------------------------------------------------
    save_directory = '/workspace/inhouse-vqvae/VQVAE/model/pixelcnn'
    os.makedirs(save_directory, exist_ok=True)
    save_path = os.path.join(save_directory, "pixelcnn_model.pth")

    torch.save(pixelcnn_model.state_dict(), save_path)
    
    # WandBへのアップロード
    OmegaConf.save(config=cfg, f='config_pixelcnn_wandb.yaml')
    artifact = wandb.Artifact(name='PixelCNN-Model', metadata=dict(cfg), type='model')
    artifact.add_file(save_path)
    artifact.add_file('config_pixelcnn_wandb.yaml')
    wandb.log_artifact(artifact)

    print('Finished PixelCNN Training')
    wandb.finish()

if __name__ == "__main__":
    main()