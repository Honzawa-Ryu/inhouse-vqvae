# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model import VQVAE

class MLP(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # MLPのパラメータ
        self.number_of_layers = kwargs['number_of_layers']
        self.input_size = kwargs['input_size']
        self.top_hidden_size = kwargs['top_hidden_size']
        self.bottom_hidden_size = kwargs['bottom_hidden_size']
        self.output_size = kwargs['output_size']
        self.emb_dim = kwargs['embedding_dim']

        # VQVAEのパラメータ
        self.vq_h_dim = kwargs['vqvae_h_dim']
        self.vq_res_h_dim = kwargs['vqvae_res_h_dim']
        self.vq_n_res = kwargs['vqvae_n_res_layers']
        self.vq_n_emb = kwargs['vqvae_n_embeddings']
        self.vq_emb_dim = kwargs['vqvae_embedding_dim']
        self.vq_beta = kwargs['vqvae_beta']
        # self.vqvae_path = kwargs['vqvae_path']
        
        # Convのパラメータ
        self.conv_dim = kwargs['conv_dim']
        self.conv_kernel_size = kwargs['conv_kernel_size']
        self.conv_pad = kwargs['conv_padding']
        self.conv_str = kwargs['conv_stride']

        self.pool_kernel_size = kwargs['pool_kernel_size']
        self.pool_str = kwargs['pool_stride']

        # VQVAEの用意
        self.vqvae = VQVAE(self.vq_h_dim,
                           self.vq_res_h_dim,
                           self.vq_n_res,
                           self.vq_n_emb,
                           self.vq_emb_dim,
                           self.vq_beta,)
        
        # pathベタ打ちなの嫌、どうにかならないか
        checkpoint = torch.load('VQVAE_local.pth', weights_only=True)
        self.vqvae.load_state_dict(checkpoint['param'])

        self.codebook = self.vqvae.vector_quantization.embedding.weight

        self.conv = nn.Conv2d(self.vq_emb_dim, self.conv_dim, kernel_size=self.conv_kernel_size, padding=self.conv_pad, stride=self.conv_str)
        self.pool = nn.MaxPool2d(kernel_size=self.pool_kernel_size, stride=self.pool_str)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # その他MLPの用意
        self.layers = nn.Sequential()
        self.layers.append(nn.Linear(self.conv_dim, self.top_hidden_size))
        self.layers.append(nn.ReLU())
        for i in range(int((self.number_of_layers-2)/2)):
            self.layers.append(nn.Linear(self.top_hidden_size, self.top_hidden_size))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(self.top_hidden_size, self.bottom_hidden_size))
        self.layers.append(nn.ReLU())
        for i in range(int((self.number_of_layers-3)/2)):
            self.layers.append(nn.Linear(self.bottom_hidden_size, self.bottom_hidden_size))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(self.bottom_hidden_size, self.output_size))
    

    def forward(self, x):
        # VQVAEを通し潜在表現を取得
        _, _, x = self.vqvae(x)
    
        min_encodings = torch.zeros(x.shape[0], self.vq_n_emb, device=x.device)
        min_encodings.scatter_(1, x, 1)
        x = torch.matmul(min_encodings, self.codebook)
        # print(x.shape)

        x = x.view(-1, 64, 64, 64)
        # x = x.view(-1, self.input_size)
        # print(x.shape)

        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = self.pool(x)

        x = self.gap(x)

        # 次元のベクトルに変換
        x = x.permute(0, 2, 3, 1)
        x = x.view(x.size(0), -1)
        # print(x.shape)

        x = self.layers(x)
        # ソフトマックスは損失関数内で計算されるため、ここでは適用しない
        return x