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
        codebook_usage_counter = torch.zeros(256, dtype=torch.long, device=self.device)
        for i in tqdm(range(self.epoch, self.max_epochs + 1)):
            train_loss = 0
            train_emb_loss = 0
            train_recon_loss = 0
            test_loss = 0
            test_emb_loss = 0
            test_recon_loss = 0
            self.epoch = i

            # 訓練
            self.model.train()
            # self.optimizer.train()
            for img, _ in self.train_loader:
                img = img.to(self.device, dtype=torch.float)
                self.optimizer.zero_grad()
                embedding_loss, x_hat, *_ = self.model(img)
                recon_loss = nn.MSELoss()(x_hat, img)
                loss = recon_loss + embedding_loss
                train_loss += loss.item()
                train_emb_loss += embedding_loss.item()
                train_recon_loss += recon_loss.item()
                loss.backward()
                self.optimizer.step()

            # 評価
            self.model.eval()
            # self.optimizer.eval()
            with torch.no_grad():
                for img_t, _ in self.test_loader:
                    img = img_t.to(self.device, dtype=torch.float)
                    embedding_loss, x_hat, idx = self.model(img)
                    recon_loss = nn.MSELoss()(x_hat, img)
                    loss = recon_loss + embedding_loss
                    test_loss += loss.item()
                    test_emb_loss += embedding_loss.item()
                    test_recon_loss += recon_loss.item()
                    
                    idx = idx.flatten()
                    counter = torch.bincount(idx, minlength=256)
                    codebook_usage_counter += counter

            # 損失の記録と表示
            train_loss /= len(self.train_loader.dataset)
            train_emb_loss /= len(self.train_loader.dataset)
            train_recon_loss /= len(self.train_loader.dataset)
            test_loss /= len(self.test_loader.dataset)
            test_emb_loss /= len(self.test_loader.dataset)
            test_recon_loss /= len(self.test_loader.dataset)
            print(f'epoch {i} train_loss: {train_loss:.5f} test_loss: {test_loss:.5f}')
            self.run.log({"train_loss": train_loss, "train_emb_loss": train_emb_loss, "train_recon_loss": train_recon_loss,
                          "test_emb_loss": test_emb_loss, "test_recon_loss": test_recon_loss, "test_loss": test_loss})
            self.train_loss_log.append(train_loss)
            self.test_loss_log.append(test_loss)
            print(codebook_usage_counter)

        return self.train_loss_log, self.test_loss_log