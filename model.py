import numpy as np
import torch
import torch.nn as nn


class GaussianDiffusion:
    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_steps = num_steps
        self.betas = torch.linspace(beta_start, beta_end, num_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

    def q_sample(self, x_start, t, noise=None):
        """Прямой процесс диффузии (добавление шума)"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t]).view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - self.alphas_cumprod[t]).view(-1, 1, 1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

class DiffusionTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, vocab_size=30522, max_seq_len=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model))

        # Мы кодируем шаг диффузии и условие диагноза в то же пространство эмбеддингов
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        self.cond_embedding = nn.Linear(d_model, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Мы прогнозируем шум в пространстве эмбеддингов
        self.output_layer = nn.Linear(d_model, d_model)

    def forward(self, x_noisy, t, cond_emb):
        """
        x_noisy: [batch_size, seq_len, d_model]
        t: [batch_size]
        cond_emb: [batch_size, d_model] (Эмбеддинг диагноза)
        """
        batch_size, seq_len, d_model = x_noisy.shape

        t_emb = self.time_mlp(t.float().view(-1, 1)).unsqueeze(1)

        c_emb = self.cond_embedding(cond_emb).unsqueeze(1)

        # Мы совмещаем позиционное кодирование с признаками шага и условия
        x = x_noisy + self.pos_encoding[:, :seq_len, :]

        # Мы добавляем специальные префиксы, чтобы трансформер видел время и диагноз перед последовательностью
        x = torch.cat([t_emb, c_emb, x], dim=1)

        out = self.transformer(x)

        out = out[:, 2:, :]

        return self.output_layer(out)
