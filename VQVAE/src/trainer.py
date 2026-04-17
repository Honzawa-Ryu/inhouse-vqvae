import torch
import torch.nn as nn
from tqdm import tqdm
# import lpips
# from .lpips_loss import lpips_loss
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import torchvision
import torchvision.transforms as T

class Trainer:
    def __init__(self, model, optimizer, device, train_loader, test_loader, max_epochs, wandb_run, save_path='./best_model.pth'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.max_epochs = max_epochs
        self.epoch = 0
        self.run = wandb_run
        
        # --- 改善点1: 損失関数をここで一度だけ定義 ---
        self.criterion = nn.MSELoss()
        # self.lpips_loss_fn = lpips.LPIPS(net='vgg16').to(self.device)
        
        # --- Wandb画像可視化対応: 可視化用のバッチを固定 ---
        self.vis_images, _ = next(iter(self.test_loader))
        self.vis_images = self.vis_images[:10].to(self.device, dtype=torch.float) # 最初の10枚をデバイスへ

        # --- 改善点4: モデル保存用の変数を追加 ---
        self.best_test_loss = float('inf')
        self.save_path = save_path

    def train(self):
        print("Training started...")
        for i in tqdm(range(self.epoch, self.max_epochs), desc="Epochs"):
            self.epoch = i
            
            # --- 改善点3: ループ内でカウンターをリセット ---
            codebook_usage_counter = torch.zeros(self.model.n_emb, dtype=torch.long, device=self.device)

            # --- 訓練フェーズ ---
            self.model.train()
            train_loss, train_emb_loss, train_recon_loss, train_lpips_loss = 0, 0, 0, 0
            
            for img, _ in tqdm(self.train_loader, desc=f"  Train {i}", leave=False):
                img = img.to(self.device, dtype=torch.float)
                self.optimizer.zero_grad()
                
                embedding_loss, x_hat, *_ = self.model(img)
                # x_hat_ave = torch.mean(x_hat, dim=[2, 3], keepdim=True)
                # img_ave = torch.mean(img, dim=[2, 3], keepdim=True)
                # recon_loss = self.criterion(x_hat-x_hat_ave, img-img_ave)


                recon_loss = self.criterion(x_hat, img)
                # d = self.lpips_loss_fn.forward(x_hat, img)
                loss =  recon_loss + embedding_loss # + d.mean()
                
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                train_emb_loss += embedding_loss.item()
                train_recon_loss += recon_loss.item()
                # train_lpips_loss += abs(d.mean().item())
                
            # --- 改善点2: 損失をバッチ数で正規化 ---
            train_loss /= len(self.train_loader)
            train_emb_loss /= len(self.train_loader)
            train_recon_loss /= len(self.train_loader)
            # train_lpips_loss /= len(self.train_loader)

            # --- 評価フェーズ ---
            self.model.eval()
            test_loss, test_emb_loss, test_recon_loss, test_lpips_loss = 0, 0, 0, 0
            
            with torch.no_grad():
                for img_t, _ in tqdm(self.test_loader, desc=f"  Test {i}", leave=False):
                    img = img_t.to(self.device, dtype=torch.float)
                    embedding_loss, x_hat, idx = self.model(img)
                    
                    # x_hat_ave = torch.mean(x_hat, dim=[2, 3], keepdim=True)
                    # img_ave = torch.mean(img, dim=[2, 3], keepdim=True)
                    # recon_loss = self.criterion(x_hat-x_hat_ave, img-img_ave)
                    
                    recon_loss = self.criterion(x_hat, img)
                    # d = self.lpips_loss_fn.forward(x_hat, img)
                    loss = recon_loss + embedding_loss #+ d.mean().item()

                    test_loss += loss.item()
                    test_emb_loss += embedding_loss.item()
                    test_recon_loss += recon_loss.item()
                    # test_lpips_loss += abs(d.mean().item())
                    
                    # コードブック使用率の計算
                    counter = torch.bincount(idx.flatten(), minlength=self.model.n_emb)
                    codebook_usage_counter += counter
            
            test_loss /= len(self.test_loader)
            test_emb_loss /= len(self.test_loader)
            test_recon_loss /= len(self.test_loader)
            # test_lpips_loss /= len(self.test_loader)
            
            print(f'Epoch {i}: Train Loss: {train_loss:.5f}, Test Loss: {test_loss:.5f}')

            # --- 改善点4: ベストモデルの保存 ---
            if test_loss < self.best_test_loss:
                self.best_test_loss = test_loss
                torch.save(self.model.state_dict(), self.save_path)
                print(f"  -> New best model saved to {self.save_path} (Loss: {self.best_test_loss:.5f})")

            # --- Wandbへのログ記録 ---
            log_data = {
                "train_loss": train_loss,
                "train_emb_loss": train_emb_loss,
                "train_recon_loss": train_recon_loss,
                # "train_lpips_loss": train_lpips_loss,
                "test_loss": test_loss,
                "test_emb_loss": test_emb_loss,
                "test_recon_loss": test_recon_loss,
                # "test_lpips_loss": test_lpips_loss,
            }
            # wandbログに追加
            log_data["codebook_dead_units"] = (codebook_usage_counter == 0).sum().item()    
            # --- Wandb画像可視化対応: エポックごとに画像をログ ---
        
            _, x_hat_vis, _ = self.model(self.vis_images)
            # 元画像と再構成画像を結合してグリッド表示

            inv_normalize = T.Normalize(
                mean=[-0.8259633779525757 / 0.12393593788146973, 
                    -0.4840644896030426 / 0.19072337448596954, 
                    -0.6278038620948792 / 0.15796850621700287],
                std=[1/0.12393593788146973, 
                    1/0.19072337448596954, 
                    1/0.15796850621700287]
            )
            x_hat_vis = inv_normalize(x_hat_vis)
            vis_images = inv_normalize(self.vis_images)

            # restored_img = inv_normalize(normalized_tensor)
            combined_images = torch.cat([vis_images, x_hat_vis], dim=0)
            grid = torchvision.utils.make_grid(combined_images, nrow=10, normalize=True, value_range=(0, 1))
            log_data["reconstructions"] = wandb.Image(grid, caption=f"Top: Original, Bottom: Reconstructed (Epoch {i})")
        
            self.run.log(log_data)

        print("Training finished.")