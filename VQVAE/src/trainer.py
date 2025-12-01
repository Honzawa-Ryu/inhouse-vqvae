import torch
import torch.nn as nn
from tqdm import tqdm
import lpips

import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import torchvision

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
        self.lpips_loss_fn = lpips.LPIPS(net='alex').to(self.device)
        
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
            
            for img, _ in self.train_loader:
                img = img.to(self.device, dtype=torch.float)
                self.optimizer.zero_grad()
                
                embedding_loss, x_hat, *_ = self.model(img)
                # x_hat_ave = torch.mean(x_hat, dim=1, keepdim=True)
                # img_ave = torch.mean(img, dim=1, keepdim=True)
                # recon_loss = self.criterion(x_hat-x_hat_ave, img-img_ave)

                recon_loss = self.criterion(x_hat, img)
                d = self.lpips_loss_fn.forward(x_hat, img)
                loss =  recon_loss + embedding_loss + d.mean()*0.01
                
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                train_emb_loss += embedding_loss.item()
                train_recon_loss += recon_loss.item()
                train_lpips_loss += d.mean().item()
                
            # --- 改善点2: 損失をバッチ数で正規化 ---
            train_loss /= len(self.train_loader)
            train_emb_loss /= len(self.train_loader)
            train_recon_loss /= len(self.train_loader)
            train_lpips_loss /= len(self.train_loader)

            # --- 評価フェーズ ---
            self.model.eval()
            test_loss, test_emb_loss, test_recon_loss, test_lpips_loss = 0, 0, 0, 0
            
            with torch.no_grad():
                for img_t, _ in self.test_loader:
                    img = img_t.to(self.device, dtype=torch.float)
                    embedding_loss, x_hat, idx = self.model(img)
                    
                    # x_hat_ave = torch.mean(x_hat, dim=1, keepdim=True)
                    # img_ave = torch.mean(img, dim=1, keepdim=True)
                    # recon_loss = self.criterion(x_hat-x_hat_ave, img-img_ave)
                    
                    recon_loss = self.criterion(x_hat, img)
                    d = self.lpips_loss_fn.forward(x_hat, img)
                    loss = recon_loss + embedding_loss + d.mean()*0.01

                    test_loss += loss.item()
                    test_emb_loss += embedding_loss.item()
                    test_recon_loss += recon_loss.item()
                    test_lpips_loss += d.mean().item()
                    
                    # コードブック使用率の計算
                    counter = torch.bincount(idx.flatten(), minlength=self.model.n_emb)
                    codebook_usage_counter += counter
            
            test_loss /= len(self.test_loader)
            test_emb_loss /= len(self.test_loader)
            test_recon_loss /= len(self.test_loader)
            test_lpips_loss /= len(self.test_loader)
            
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
                "train_lpips_loss": train_lpips_loss,
                "test_loss": test_loss,
                "test_emb_loss": test_emb_loss,
                "test_recon_loss": test_recon_loss,
                "test_lpips_loss": test_lpips_loss,
            }

            # --- Wandb画像可視化対応: エポックごとに画像をログ ---
        
            _, x_hat_vis, _ = self.model(self.vis_images)
            # 元画像と再構成画像を結合してグリッド表示
            combined_images = torch.cat([self.vis_images, x_hat_vis], dim=0)
            grid = torchvision.utils.make_grid(combined_images, nrow=10, normalize=True, value_range=(0, 1))
            log_data["reconstructions"] = wandb.Image(grid, caption=f"Top: Original, Bottom: Reconstructed (Epoch {i})")
        
            self.run.log(log_data)

        print("Training finished.")