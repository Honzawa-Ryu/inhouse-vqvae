import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig
from torchvision import transforms, datasets

from src.model import VQVAE, VQVAE2, MLP
from src.trainer import Trainer
from src.data_handler import get_mnist_dataloaders, DataSet, get_image_dataloaders, get_class_specific_dataloaders  # DataSet もこちらで定義
from src.utils import plot_loss
import wandb

def idx_dataloaders(data_dir, batch_size, train_val_split=0.8, image_size=(256, 256), sampling_rate=None):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    print(f"クラス情報: {full_dataset.class_to_idx}")
    print(f"元の合計画像数: {len(full_dataset)}")

    if sampling_rate is not None:
        if not (0.0 < sampling_rate <= 1.0):
            raise ValueError("Sampling_rateは0.0より大きく1.0以下の値でなければなりません。")
        
        num_samples = int(len(full_dataset) * sampling_rate)
        
        torch.manual_seed(42)
        indices = torch.randperm(len(full_dataset))[:num_samples]
        
        sampled_dataset = Subset(full_dataset, indices)
        
        print(f"サンプリング適用後 ({sampling_rate * 100}%)")
        print(f"  -> サンプリング後の合計画像数: {len(sampled_dataset)}")
        dataset_to_split = sampled_dataset
    else:
        dataset_to_split = full_dataset
    data_loader = DataLoader(
        dataset_to_split,
        batch_size=batch_size,
        shuffle=True,
        num_workers=32,
        pin_memory=True
    )
    return data_loader

@hydra.main(config_name="config.yaml", version_base=None, config_path="/workspace/inhouse-vqvae/VQVAE/config")
def main(cfg: DictConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, name=cfg.wandb.name)

    # --- パラメータ設定 ---
    DATA_DIRECTORY = '/workspace/inhouse-vqvae/VQVAE/data/preprocessed'  # 画像データが入ったディレクトリ
    BATCH_SIZE = 64
    IMAGE_SIZE = (256, 256)

    # --- データハンドラを実行 ---
    dataloader = idx_dataloaders(
        data_dir=DATA_DIRECTORY,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        sampling_rate=None
    )

    # --- モデル解析のループ ---
    model = VQVAE(**cfg.model).to(device)

    try:
        artifact = run.use_artifact('TGGATE-Recon:v0')
        model_path = artifact.get_entry("savetest.pth").download()
        checkpoint = torch.load(model_path, map_location=device)
        
        # モデルの状態辞書が 'param' キーの下に格納されていると仮定してロード
        # もし状態辞書が直接保存されている場合は model.load_state_dict(checkpoint) を使用してください
        model.load_state_dict(checkpoint)
        print(f"モデルを {model_path} から正常にロードしました。")
    except Exception as e:
        print(f"モデルのロード中にエラーが発生しました: {e}")
        return # ロードに失敗した場合は処理を中断

    model.eval()

    # 辞書のitems()を使って、クラス名とデータローダーを順番に取り出す
    total_codebook_usage_counter = torch.zeros(1, cfg.model.vqvae_n_embeddings, dtype=torch.long, device=device)
    idx_bundle_list = []
    batch_count = 0

    # このループでは、指定したクラスの画像だけが順番に出てくる
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            img = images.to(device, dtype=torch.float)
            *_, idx = model(img)
            
            idx = idx.flatten()
            
            counter = torch.bincount(idx, minlength=cfg.model.vqvae_n_embeddings)
            total_codebook_usage_counter += counter.view(1, -1)
            
            if i < 10:
                idx_bundle_list.append(idx.view(1, -1))

            batch_count += 1
            if batch_count % 100 == 0:
                print(f"処理済みパッチ数：{batch_count}")
            

            pass
                
    print(f"--- 解析が完了 ---")
    
    torch.save(total_codebook_usage_counter, f'/workspace/inhouse-vqvae/VQVAE/model/shuffle/tg/tg.pt')
    print(f"コードブック使用頻度を tg.pt に保存しました。形状: {total_codebook_usage_counter}")
    
    # 2. 全てのインデックスを連結 (行方向に)
    if idx_bundle_list:
        # idx_bundleは (全バッチ数, BATCH_SIZE * H' * W') の形状になる
        idx_bundle = torch.cat(idx_bundle_list, 0)
        torch.save(idx_bundle, f'/workspace/inhouse-vqvae/VQVAE/model/shuffle/tg8/tg_idx.pt')
        print(f"全インデックスバンドルを tg_idx.pt に保存しました。形状: {idx_bundle.shape}")

if __name__ == "__main__":
    main()