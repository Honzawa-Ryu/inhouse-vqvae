"""
VQ-VAE + PixelCNN による画像生成スクリプト
"""
import torch
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
import hydra
from omegaconf import DictConfig
import os
from tqdm import tqdm
import wandb
from omegaconf import OmegaConf, DictConfig

# 既存のモデル定義をインポート
from src.model import VQVAE, GatedPixelCNN

def sample_indices(pixelcnn_model, batch_size, latent_h, latent_w, num_embeddings, device, temperature=1.0):
    """
    PixelCNNを使ってインデックスマップを自己回帰的に生成する関数
    ここが生成の核心部分です。
    """
    pixelcnn_model.eval()
    
    # 空のキャンバスを用意（最初は全て0で埋める）
    # shape: [Batch, H, W]
    indices = torch.zeros((batch_size, latent_h, latent_w), dtype=torch.long).to(device)
    
    print(f"Sampling indices ({latent_h}x{latent_w})... This will take time.")
    with torch.no_grad():
        # ラスタスキャン順序（左上から右下へ）で1つずつループ
        for i in tqdm(range(latent_h), desc="Rows"):
            for j in range(latent_w):
                # 1. 現在のキャンバスをモデルに入力
                # output shape: [Batch, CodebookSize, H, W]
                out = pixelcnn_model(indices)
                
                # 2. 現在着目している位置 (i, j) のロジット（スコア）を取り出す
                # shape: [Batch, CodebookSize]
                logits = out[:, :, i, j]
                
                # 3. Temperature Scaling (多様性の調整)
                logits = logits / temperature
                
                # 4. Softmaxで確率分布に変換
                probs = F.softmax(logits, dim=1)
                
                # 5. 確率分布に従ってサンプリング (Multinomial)
                # argmaxだと毎回同じ画像になってしまうため、確率的に選ぶ
                # next_token shape: [Batch]
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                
                # 6. キャンバスの該当位置を更新
                indices[:, i, j] = next_token
                
    return indices

def decode_indices(vqvae_model, indices):
    """
    生成されたインデックスをVQ-VAEのデコーダーで画像に戻す関数
    """
    vqvae_model.eval()
    with torch.no_grad():
        # 1. インデックスをEmbeddingベクトルに変換
        # VQVAEの実装によりますが、一般的にQuantizerがこの機能を持っています。
        # 例: indices [B, H, W] -> quantized [B, EmbeddingDim, H, W]
        
        # ※ お使いのモデル実装に合わせて調整が必要です。
        # 一般的な実装例:
        # quantized = vqvae_model.quantizer.embedding(indices)
        # quantized = quantized.permute(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W]

        # もしquantizerに直接変換メソッドがない場合の手動実装例:
        embedding_weight = vqvae_model.vector_quantization.embedding.weight #[NumEmbed, Dim]
        quantized = F.embedding(indices, embedding_weight) # [B, H, W, Dim]
        quantized = quantized.permute(0, 3, 1, 2).contiguous() # [B, Dim, H, W]

        # 2. デコーダーに通して画像にする
        generated_images = vqvae_model.decoder(quantized)
        
    return generated_images


@hydra.main(config_name="config", version_base=None, config_path="./config")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Generating on {device}")

    run = wandb.init(
        entity=cfg.wandb.entity, 
        project=cfg.wandb.project, 
        name="pixelcnn-training", 
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )

    # ==========================================
    # 1. モデルのロード
    # ==========================================
    # VQ-VAE (Stage 1)
    print("Loading VQ-VAE...")
    vqvae = VQVAE(**cfg.model).to(device)
    artifact = run.use_artifact('TGGATE-Recon:v1')
    model_path = artifact.get_entry("savetest.pth").download()
    checkpoint = torch.load(model_path, weights_only=True)
    vqvae.load_state_dict(checkpoint)
    vqvae.eval()

    # PixelCNN (Stage 2)
    print("Loading PixelCNN...")
    pixelcnn = GatedPixelCNN(
        num_embeddings=cfg.model.vqvae_n_embeddings,
        hidden_dim=64,
        n_layers=15
    ).to(device)
    artifact = run.use_artifact('PixelCNN-Model:latest')
    model_path = artifact.get_entry("pixelcnn_model.pth").download()
    checkpoint = torch.load(model_path, weights_only=True)
    pixelcnn.load_state_dict(checkpoint)
    pixelcnn.eval()

    # ==========================================
    # 2. 生成プロセス (サンプリング)
    # ==========================================
    H, W = [64, 64]
    
    # Stage 2: PixelCNNでインデックスマップを生成
    sampled_indices = sample_indices(
        pixelcnn, 
        16, 
        H, W, 
        512, 
        device,
        temperature=1.0
    )
    print("Indices sampled successfully.")

    # Stage 1: VQ-VAEで画像にデコード
    print("Decoding indices to images...")
    generated_images = decode_indices(vqvae, sampled_indices)

    # ==========================================
    # 3. 保存と表示
    # ==========================================
    os.makedirs("./data/generated_images", exist_ok=True)

    # 画像の値域を [0, 1] に正規化 (モデル出力がtanhなら[-1,1]なので)
    # お使いのモデルの出力に合わせて調整してください
    # generated_images = (generated_images + 1.0) / 2.0 
    generated_images = torch.clamp(generated_images, 0.0, 1.0)

    # グリッド画像として保存
    grid = make_grid(generated_images, nrow=4, padding=2, normalize=False)
    save_path = os.path.join("./data/generated_images", "generated_grid.png")
    save_image(grid, save_path)
    
    print(f"Generated images saved to {save_path}")
    
    # 個別の画像としても保存
    for i in range(16):
        ind_path = os.path.join("./data/generated_images", f"gen_{i:03d}.png")
        save_image(generated_images[i], ind_path)

if __name__ == "__main__":
    main()