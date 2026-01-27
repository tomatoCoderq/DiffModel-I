import torch
import torch.nn as nn
import numpy as np

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
        
        # Эмбеддинг времени для информирования модели о текущем шаге диффузии
        self.time_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Эмбеддинг условия (диагноза)
        self.cond_embedding = nn.Linear(d_model, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Linear(d_model, d_model) # Predicts the noise or x_start
        
    def forward(self, x_noisy, t, cond_emb):
        """
        x_noisy: [batch_size, seq_len, d_model]
        t: [batch_size]
        cond_emb: [batch_size, d_model] (Эмбеддинг диагноза)
        """
        batch_size, seq_len, d_model = x_noisy.shape
        
        # Эмбеддинг времени
        t_emb = self.time_mlp(t.float().view(-1, 1)).unsqueeze(1) # [batch_size, 1, d_model]
        
        # Эмбеддинг условия
        c_emb = self.cond_embedding(cond_emb).unsqueeze(1) # [batch_size, 1, d_model]
        
        # Добавляем позиционное кодирование и объединяем с временем/условием
        # Мы можем объединить их как специальные токены или сложить
        x = x_noisy + self.pos_encoding[:, :seq_len, :]
        
        # Конкатенируем информацию об условии и времени как префиксные токены
        x = torch.cat([t_emb, c_emb, x], dim=1) # [batch_size, seq_len + 2, d_model]
        
        # Проход через Transformer
        out = self.transformer(x)
        
        # Извлекаем денойзированную последовательность (пропуская префиксные токены)
        out = out[:, 2:, :]
        
        return self.output_layer(out)
