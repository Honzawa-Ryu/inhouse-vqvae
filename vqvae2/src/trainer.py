import torch
import torch.nn as nn
from tqdm import tqdm

class Trainer:
    def __init__(self, model, optimizer, device, train_loader, test_loader, max_epochs, wandb_run):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.max_epochs = max_epochs
        self.epoch = 0
        self.train_loss_log = []
        self.test_loss_log = []
        self.run = wandb_run

    def train(self):
        for i in tqdm(range(self.epoch, self.max_epochs + 1)):
            train_loss = 0
            test_loss = 0
            self.epoch = i

            # 訓練
            self.model.train()
            self.optimizer.train()
            for img, _ in self.train_loader:
                img = img.to(self.device, dtype=torch.float)
                self.optimizer.zero_grad()
                x_hat, _, _, train_embedding_loss = self.model(img)
                train_recon_loss = nn.MSELoss()(x_hat, img)
                lossr = train_recon_loss + train_embedding_loss*(i/(self.max_epochs*100))*0
                train_loss += lossr.item()
                lossr.backward()
                max_norm = 1.0  # 勾配ノルムの最大値
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
                # 勾配クリッピングの実装
                self.optimizer.step()

            # 評価
            self.model.eval()
            self.optimizer.eval()
            with torch.no_grad():
                for img_t, _ in self.test_loader:
                    img = img_t.to(self.device, dtype=torch.float)
                    x_hat, _, _, embedding_loss = self.model(img)
                    recon_loss = nn.MSELoss()(x_hat, img)
                    loss = recon_loss + embedding_loss*(i/(self.max_epochs*100))*0
                    test_loss += loss.item()

            # 損失の記録と表示
            train_loss /= len(self.train_loader)
            test_loss /= len(self.test_loader)
            print(f'epoch {i} train_loss: {train_loss:.5f} test_loss: {test_loss:.5f}')
            self.run.log({"train_loss": train_loss, "test_loss": test_loss,"train_recon_loss": train_recon_loss.item(), "train_embedding_loss": train_embedding_loss.item(), "test_recon_loss": recon_loss.item(), "test_embedding_loss": embedding_loss.item()})
            self.train_loss_log.append(train_loss)
            self.test_loss_log.append(test_loss)

        return self.train_loss_log, self.test_loss_log