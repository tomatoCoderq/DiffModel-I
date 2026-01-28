import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusion_wrapper import MedicalDiffusionModel
from tqdm import tqdm

from dataset import get_dataloader

def train():
    # Гиперпараметры
    batch_size = 32
    seq_len = 128
    d_model = 512
    epochs = 200
    learning_rate = 5e-5
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Инициализация модели и данных
    model = MedicalDiffusionModel(d_model=d_model, max_seq_len=seq_len).to(device)
    dataloader = get_dataloader(data_path="expanded_medical_dataset.csv", batch_size=batch_size, max_length=seq_len)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    mse_loss = nn.MSELoss()
    
    best_loss = float('inf')

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            optimizer.zero_grad()
            
            # Перенос данных на устройство
            input_ids = batch["input_ids"].to(device)
            cond_ids = batch["cond_ids"].to(device)

            # 1. Конвертация ID в эмбеддинги (x_start)
            # Мы используем собственную матрицу эмбеддингов трансформатора для клинических заметок
            with torch.no_grad():
                x_start = model.backbone.embedding(input_ids)
                # Мы также можем использовать фиксированный или обучаемый энкодер для условия
                # Здесь мы просто используем эмбеддинг модели и усредняем его
                cond_emb = model.backbone.embedding(cond_ids).mean(dim=1)
            
            # 2. Выбор случайных временных шагов
            t = torch.randint(0, model.diffusion_pipeline.num_steps, (input_ids.shape[0],)).to(device)
            
            # 3. Генерация шума
            noise = torch.randn_like(x_start)
            
            # 4. Добавление шума к x_start (Прямой процесс)
            x_noisy = model.diffusion_pipeline.q_sample(x_start, t, noise)
            
            # 5. Предсказание шума с помощью Transformer backbone
            noise_pred = model.get_noise_prediction(x_noisy, t, cond_emb)
            
            # 6. Loss: MSE между реальным шумом и предсказанным
            loss = mse_loss(noise_pred, noise)
            
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Сохранение лучшей версии
        avg_loss = loss.item()
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "medical_diffusion_best.pt")

    # Финальное сохранение
    torch.save(model.state_dict(), "medical_diffusion.pt")
    print(f"Training complete. Best loss: {best_loss:.4f}. Model saved.")

if __name__ == "__main__":
    train()
