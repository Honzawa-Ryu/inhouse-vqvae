import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from src.data_handler import get_mnist_dataloaders, DataSet, get_image_dataloaders, idx_dataloaders  # DataSet もこちらで定義
from tqdm import tqdm

print('A')
trainloader, testloader = idx_dataloaders('/workspace/inhouse-vqvae/VQVAE/data/preprocessed', 256, sampling_rate=0.025)
device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_mean_and_std(loader, device): # deviceを引数に追加
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    
    with torch.no_grad():
        for data, _ in tqdm(loader, desc='Calculating mean and std'):
            
            # 🌟 ここでデータをGPUに移動 🌟
            data = data.to(device)
            
            # チャンネル軸 (dim=1) 以外のすべての次元で平均を計算
            channels_sum += torch.mean(data, dim=[0, 2, 3])
            channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
            num_batches += 1

    # 計算結果をCPUに戻してからリストに変換 (tolist()はCPUテンソルにのみ適用可能)
    mean = (channels_sum / num_batches).cpu().tolist()
    
    # torch.sqrt(E[x^2] - (E[x])^2)
    std = torch.sqrt(channels_squared_sum / num_batches - (channels_sum / num_batches)**2).cpu().tolist()
    
    return mean, std

# ⚠️ 注意: ここで定義したデータセットとデータローダーを使って計算を実行
# train_loader = DataLoader(dataset, batch_size=64, shuffle=False)
print('A')
dataset_mean, dataset_std = get_mean_and_std(trainloader, device)

print(f"計算された平均 (Mean): {dataset_mean}")
print(f"計算された標準偏差 (Std): {dataset_std}")