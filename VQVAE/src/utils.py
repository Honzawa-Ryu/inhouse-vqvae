import matplotlib.pyplot as plt
import wandb

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
