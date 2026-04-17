"""
MLP学習用のコード
Configが長すぎるのでファイル分けするべき
"""

import torch
import torch.optim as optim
from torchvision import datasets, transforms
import hydra
from omegaconf import DictConfig, OmegaConf
from src.data_handler import get_mnist_dataloaders, DataSet, get_image_dataloaders  # DataSet もこちらで定義
from src.data_handler_kari import idx_dataloaders
from src.model import VQVAE, MLP, MLP2
from src.utils import init_wandb
import wandb
import os

@hydra.main(config_name="config.yaml", version_base=None, config_path="/workspace/inhouse-vqvae/VQVAE/config")
def main(cfg: DictConfig):
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Wandbの設定
    # wandb.init(config=dict(cfg.model),
    #            entity="benzelongji-the-university-of-tokyo",
    #            project="2025-9-2-vqvae2-mlp",
    #            name='dataset-test')
    # init_wandb(cfg)
    run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project, name="mlp-training", config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    
    target_class_1_folders = ['63431', '63433', '63435', '63438', '63442', '63446', '63450', '63466', '63473', '63477', '63481', '63485', '63489', '63494', '63497', '63501', '63505', '63509', '63513', '63517', '63522', '63526', '63530', '63534', '63538', '63542', '63546', '63550', '63553', '63558', '63562', '63566', '63571', '63575', '63579', '63583', '63587', '63591', '63595', '63598', '63602', '63605', '63609', '63612', '63616', '63620', '63623', '63665', '63669', '63672', '63676', '63680', '63748', '63752', '63756', '63831', '63937', '63684', '63687', '63691', '70929', '63698', '63760', '63763', '63768', '63772', '63777', '63855', '63860', '63865', '71285', '63875', '63941', '63944', '63948', '63952', '63955', '27622', '27624', '27626', '27628', '27630', '27632', '27634', '27636', '27638', '27640', '27642', '27644', '27646', '27651', '27655', '27657', '27660', '27662', '27663', '27665', '27668', '27670', '27672', '27674', '27676', '27678', '27680', '27682', '27684', '27686', '27689', '46720', '46727', '46736', '46740', '46744', '46747', '46750', '46753', '46756', '46758', '46759', '46763', '46765', '46768', '46769', '46772', '46774', '46778', '46781', '46783', '46787', '46789', '46792', '46795', '46797', '46800', '46803', '46806', '46809', '46811', '46815', '46817', '46820', '46823', '46826', '46829', '46831', '46833', '46835', '46839', '46841', '46844']
    train_loader, test_loader = idx_dataloaders(cfg.data.data_root, class_1_folders=target_class_1_folders, batch_size=cfg.train.batch_size, sampling_rate=cfg.data.sampling_rate)
    
    # モデル、損失関数、最適化手法の定義
    if cfg.train.VQVAE:
        model = MLP(run, **cfg.model).to(device)
    else:
        model = MLP2(**cfg.model).to(device)
    if cfg.train.frozen:
        for param in model.vqvae.parameters():
            param.requires_grad = False

    # パラメータ数の記録
    total_params = sum(
	param.numel() for param in model.parameters()
    )
    # wandb.config.total_parameters = total_params

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    # optimizer = RAdamScheduleFree(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)
 
    # 学習ループ
    for epoch in range(cfg.train.epochs):
        running_loss = 0.0
        class_loss = 0.0
        recon_loss = 0.0
        model.train()
        # optimizer.train()
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 勾配をゼロにリセット
            optimizer.zero_grad()

            # 順伝播、誤差計算、逆伝播、パラメータ更新
            outputs, vq_loss = model(inputs)
            mlp_loss = criterion(outputs, labels)
            if cfg.train.multiheadgrad:
                loss = mlp_loss * cfg.train.gamma + vq_loss
            else:
                loss = mlp_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            class_loss += mlp_loss.item()
            recon_loss += vq_loss.item()
        # scheduler.step()

        print(f'Epoch [{epoch+1}/{cfg.train.epochs}], Loss: {running_loss/len(train_loader):.4f}')
        wandb.log({"loss": running_loss/len(train_loader), "mlp_loss": class_loss/len(train_loader), "vq_loss": recon_loss/len(train_loader)})

        model.eval()  # モデルを評価モードに設定
        # optimizer.eval()
        correct = 0
        total = 0

        # 推論中は勾配計算を無効にする
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # 予測を行う
                outputs, _ = model(inputs)

                # 確率が最も高いクラスのインデックスを取得
                # `torch.max`は (最大値, 最大値のインデックス) のタプルを返す
                predicted = torch.argmax(outputs.data, 1)
                if labels.dim() == 2 and labels.size(1) > 1:
                    labels = torch.argmax(labels, dim=1)

                # 全サンプルの総数を更新
                total += labels.size(0)

                # 正しく予測できた数を更新
                correct += (predicted == labels).sum().item()

        # 最終的な精度を計算して出力
        accuracy = 100 * correct / total
        print(f"Accuracy: {accuracy:.2f}%")
        wandb.log({"accuracy": accuracy})

        # 学習済みモデルの保存
    save_directory = '/workspace/inhouse-vqvae/VQVAE/model/mlp'
    save_path = os.path.join(save_directory, cfg.model.save_name)

    if cfg.train.VQVAE:
        torch.save(model.state_dict(), save_path)
        OmegaConf.save(config=cfg, f='config_wandb.yaml')
        artifact = wandb.Artifact(name='VQVAE-MLP', metadata=dict(cfg.model), type='model')
        artifact.add_file(save_path)
        artifact.add_file('config_wandb.yaml')
        wandb.log_artifact(artifact)
    else:
        torch.save(model.state_dict(), 'mlp2_mnist.pth')

    

    print('Finished Training')
    # wandb.alert(title="Finished training", text="Finished training")
    wandb.finish()


if __name__ == "__main__":
    main()