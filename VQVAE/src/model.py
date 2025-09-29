import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import math

class ResidualLayer(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x

class ResidualStack(nn.Module):
    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim)] * n_res_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Encoder, self).__init__()
        kernel = 4
        stride = 2
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers)
        )

    def forward(self, x):
        return self.conv_stack(x)

class Encoder16(nn.Module):
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super().__init__()
        kernel = 4
        stride = 2
        
        self.conv_stack = nn.Sequential(
            # 1層目: stride=2 -> 辺の長さが 1/2 に
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            # 2層目: stride=2 -> 辺の長さが 1/4 に
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            # 3層目 (追加): stride=2 -> 辺の長さが 1/8 に
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            # 4層目 (追加): stride=2 -> 辺の長さが 1/16 に
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            # 元のコードにあった層。サイズは変えずに特徴を調整
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1),
            # Residualブロック
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers)
        )

    def forward(self, x):
        return self.conv_stack(x)

class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z.device)  # デバイスを明示的に指定
        min_encodings.scatter_(1, min_encoding_indices, 1)
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        loss = torch.mean((z.detach() - z_q) ** 2) + \
               self.beta * torch.mean((z - z_q.detach()) ** 2)
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return loss, z_q, min_encodings, min_encoding_indices

class Decoder(nn.Module):
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2
        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(in_dim, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim // 2, 3, kernel_size=kernel, stride=stride, padding=1)
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)

class Decoder16(nn.Module):
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super().__init__()
        kernel = 4
        stride = 2
        
        self.inverse_conv_stack = nn.Sequential(
            # 入力された潜在表現をまずResidualブロックに通す
            nn.ConvTranspose2d(in_dim, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            # 1層目: 辺の長さが 2倍 に
            nn.ConvTranspose2d(h_dim, h_dim, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            # 2層目: 辺の長さが 4倍 に
            nn.ConvTranspose2d(h_dim, h_dim, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            # 3層目: 辺の長さが 8倍 に
            nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            # 4層目: 辺の長さが 16倍 に
            nn.ConvTranspose2d(h_dim // 2, 3, kernel_size=kernel, stride=stride, padding=1)
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)
    

class FakeDecoder(nn.Module):
    # 二段階目以上のVQVAEモジュールのデコーダ、画像サイズに戻すのではなく一段階上の潜在表現に戻す
    # なんか適当にやったので三層以上にしたとき齟齬が出るかも、要確認
    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super().__init__()
        kernel = 4
        stride = 2
        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(in_dim, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim, kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=kernel, stride=stride, padding=1)
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)


class VQVAE(nn.Module):
    def __init__(self, **kwargs):
        super(VQVAE, self).__init__()
        self.h_dim = kwargs['vqvae_h_dim']
        self.res_h_dim = kwargs['vqvae_res_h_dim']
        self.n_res = kwargs['vqvae_n_res_layers']
        self.n_emb = kwargs['vqvae_n_embeddings']
        self.emb_dim = kwargs['vqvae_embedding_dim']
        self.beta = kwargs['vqvae_beta']

        self.encoder = Encoder(3, self.h_dim, self.n_res, self.res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(self.h_dim, self.emb_dim, kernel_size=1, stride=1)
        self.vector_quantization = VectorQuantizer(self.n_emb, self.emb_dim, self.beta)
        self.decoder = Decoder(self.emb_dim, self.h_dim, self.n_res, self.res_h_dim)

    def forward(self, x):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, _, idx = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)
        return embedding_loss, x_hat, idx
    
class VQVAE16(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.h_dim = kwargs['vqvae_h_dim']
        self.res_h_dim = kwargs['vqvae_res_h_dim']
        self.n_res = kwargs['vqvae_n_res_layers']
        self.n_emb = kwargs['vqvae_n_embeddings']
        self.emb_dim = kwargs['vqvae_embedding_dim']
        self.beta = kwargs['vqvae_beta']

        self.encoder = Encoder16(3, self.h_dim, self.n_res, self.res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(self.h_dim, self.emb_dim, kernel_size=1, stride=1)
        self.vector_quantization = VectorQuantizer(self.n_emb, self.emb_dim, self.beta)
        self.decoder = Decoder16(self.emb_dim, self.h_dim, self.n_res, self.res_h_dim)

    def forward(self, x):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        embedding_loss, z_q, _, idx = self.vector_quantization(z_e)
        x_hat = self.decoder(z_q)
        return embedding_loss, x_hat, idx

class VQVAE2(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.h_dim = kwargs['vqvae_h_dim']
        self.res_h_dim = kwargs['vqvae_res_h_dim']
        self.n_res = kwargs['vqvae_n_res_layers']
        self.n_emb = kwargs['vqvae_n_embeddings']
        self.emb_dim = kwargs['vqvae_embedding_dim']
        self.beta = kwargs['vqvae_beta']

        self.encoder_top = Encoder(3, self.h_dim, self.n_res, self.res_h_dim)
        self.pre_quantization_conv_top = nn.Conv2d(self.h_dim, self.emb_dim, kernel_size=1, stride=1)
        self.vector_quantization_top = VectorQuantizer(self.n_emb, self.emb_dim, self.beta)
        self.decoder_top = Decoder(self.emb_dim, self.h_dim, self.n_res, self.res_h_dim)
        self.encoder_bottom = Encoder(self.emb_dim, self.h_dim, self.n_res, self.res_h_dim)
        self.pre_quantization_conv_bottom = nn.Conv2d(self.h_dim, self.emb_dim, kernel_size=1, stride=1)
        self.vector_quantization_bottom = VectorQuantizer(self.n_emb, self.emb_dim, self.beta)
        self.decoder_bottom = FakeDecoder(self.emb_dim, self.h_dim, self.n_res, self.res_h_dim)

        self.summary = nn.Conv2d(self.emb_dim * 2, self.emb_dim, 3, 1, padding="same")


    def forward(self, x):
        z_e = self.encoder_top(x)

        z_e_top = self.pre_quantization_conv_top(z_e)
        embedding_loss_top, z_q_top, _, idx_top = self.vector_quantization_top(z_e_top)

        z_e_e = self.encoder_bottom(z_e_top)
        z_e_bottom = self.pre_quantization_conv_bottom(z_e_e)
        embedding_loss_bottom, z_q_bottom, _, idx_bottom = self.vector_quantization_bottom(z_e_bottom)

        x_hat_bottom = self.decoder_bottom(z_q_bottom)

        pre_x_hat = torch.cat([z_q_top, x_hat_bottom], dim=1)
        pre_x_hat = self.summary(pre_x_hat)

        x_hat = self.decoder_top(pre_x_hat)
        return embedding_loss_bottom + embedding_loss_top, x_hat, [idx_top, idx_bottom], pre_x_hat


# だいぶ雑な実装ですが動くには動きそう、データ確保したら動かす
# 今後やることとしてはまずエンコーダーの次元を調節可能にすること、三層以上に拡張可能にすること
