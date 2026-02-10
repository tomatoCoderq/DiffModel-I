import torch
import torch.nn as nn

from model import DiffusionTransformer, GaussianDiffusion


class MedicalDiffusionModel(nn.Module):
    def __init__(self, vocab_size=30522, d_model=512, max_seq_len=128):
        super().__init__()
        self.diffusion_pipeline = GaussianDiffusion()
        self.backbone = DiffusionTransformer(
            d_model=d_model,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len
        )
        self.vocab_size = vocab_size

    def get_noise_prediction(self, x_noisy, t, cond_emb):
        return self.backbone(x_noisy, t, cond_emb)

    @torch.no_grad()
    def generate(self, cond_emb, seq_len=64, device='cpu'):
        """Обратный процесс диффузии для генерации эмбеддингов текста"""
        self.eval()
        batch_size = cond_emb.shape[0]
        x = torch.randn(batch_size, seq_len, self.backbone.embedding.embedding_dim).to(device)

        for i in reversed(range(self.diffusion_pipeline.num_steps)):
            t = torch.full((batch_size,), i, dtype=torch.long).to(device)
            noise_pred = self.get_noise_prediction(x, t, cond_emb)

            # Мы используем базовый шаг обратной диффузии, оставляя пространство для альтернативных семплеров
            alpha = self.diffusion_pipeline.alphas[i]
            alpha_cumprod = self.diffusion_pipeline.alphas_cumprod[i]
            beta = self.diffusion_pipeline.betas[i]

            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0

            x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * noise_pred) + torch.sqrt(beta) * noise
            # Мы ограничиваем диапазон эмбеддингов, чтобы обратный процесс не взрывался численно
            x = torch.clamp(x, -5, 5)

        return x

    def decode_embeddings(self, embeddings):
        """Отображение непрерывных эмбеддингов обратно в ID токенов (Округление)"""
        W = self.backbone.embedding.weight

        logits = torch.matmul(embeddings, W.T)
        return torch.argmax(logits, dim=-1)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MedicalDiffusionModel().to(device)

    fake_cond = torch.randn(2, 512).to(device)

    print("Generating...")
    gen_embs = model.generate(fake_cond, seq_len=10, device=device)
    tokens = model.decode_embeddings(gen_embs)
    print("Generated tokens shape:", tokens.shape)
    print("Tokens:", tokens)
