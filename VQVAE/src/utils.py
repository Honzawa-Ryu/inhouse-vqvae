import matplotlib.pyplot as plt
import wandb
import torch
import numpy as np

def plot_loss(train_loss_log, test_loss_log):
    plt.suptitle('Loss')
    plt.plot(train_loss_log, label='train_loss')
    plt.plot(test_loss_log, label='test_loss')
    plt.grid(axis='y')
    plt.legend()
    plt.savefig("/workspace/inhouse-vqvae/VQVAE/results/test.png")

def init_wandb(cfg):
    if "wandb" not in cfg:
        print("A")
        return wandb.init(mode="disabled")

    wandb_kwargs = {
        "entity": cfg.wandb.entity,
        "project": cfg.wandb.project,
        # "resume": "auto",
    }

    if hasattr(cfg.wandb, 'name'):
        wandb_kwargs["name"] = cfg.wandb.name

    return wandb.init(config=dict(cfg.model), **wandb_kwargs)

class RandomGaussianNoise(object):
    """
    Random Gaussian noise augmentation

    img_new = img + sigma
    """
    def __init__(self, p=0.5, sigmas=[0.01, 0.05]):
        self.p = p
        self.sigma_lb, self.sigma_ub = sigmas

    def __call__(self, img):
        prob = np.random.rand()
        img_new = img

        sigma = self.sigma_lb + torch.rand(1) * (self.sigma_ub - self.sigma_lb)

        if prob < self.p:
            img_new = img + sigma * torch.randn(img.shape)
            img_new = torch.clamp(img_new, min=0, max=1)

        return img_new
    
class RandomGammaVolume(object):
    """
    Random intensity (gamma) agumentation

    img_new = a + b * img^g
    a ~ Uniform[-alpha, alpha]
    b ~ Uniform[1-beta, 1+beta]
    g ~ Uniform[1-gamma, 1+gamma]
    """
    def __init__(self, p=0.5, alpha=0.05, beta=0.1, gamma=0.3):
        self.p = p
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def __call__(self, img):
        prob = np.random.rand()
        img_new = img

        if prob < self.p:
            alpha = torch.rand(1) * self.alpha * 2 - self.alpha
            beta = 1 + torch.rand(1) * self.beta * 2 - self.beta
            gamma = 1 + torch.rand(1) * self.gamma * 2 - self.gamma

            img_new = alpha + beta * torch.pow(img, gamma)
            img_new = torch.clamp(img_new, min=0, max=1)

        return img_new