import torch
from transformers import AutoTokenizer

from dataset import MedicalDataset
from diffusion_wrapper import MedicalDiffusionModel


def verify_pipeline():
    print("--- Проверка запущена ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 30522
    d_model = 512
    seq_len = 64

    # Мы создаем свежую модель и токенизатор, чтобы убедиться, что пайплайн стартует на выбранном устройстве
    print("1. Инициализация модели...")
    model = MedicalDiffusionModel(vocab_size=vocab_size, d_model=d_model, max_seq_len=seq_len).to(device)

    # Мы прогоняем прямой процесс, чтобы проверить корректность шумоподавления
    print("2. Тестирование прямого процесса...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    sample_text = "The patient has severe abdominal pain."
    tokens = tokenizer(sample_text, return_tensors="pt", padding="max_length", max_length=seq_len, truncation=True)["input_ids"].to(device)

    with torch.no_grad():
        x_start = model.backbone.embedding(tokens)
        # Мы берем средний шаг диффузии, чтобы увидеть типичный масштаб шума
        t = torch.tensor([500]).to(device)
        noise = torch.randn_like(x_start)
        x_noisy = model.diffusion_pipeline.q_sample(x_start, t, noise)

    print(f"   Shape x_start: {x_start.shape}")
    print(f"   Shape x_noisy: {x_noisy.shape}")

    print("3. Тестирование обратного процесса (предсказание шума)...")
    fake_cond = torch.randn(1, d_model).to(device)
    noise_pred = model.get_noise_prediction(x_noisy, t, fake_cond)
    print(f"   Shape noise_pred: {noise_pred.shape}")

    print("4. Тестирование генерации (обратная диффузия)...")
    # Мы уменьшаем число шагов, чтобы сделать тест обратной диффузии заметно быстрее
    model.diffusion_pipeline.num_steps = 10
    gen_embs = model.generate(fake_cond, seq_len=16, device=device)
    print(f"   Shape сгенерированных эмбеддингов: {gen_embs.shape}")

    print("5. Тестирование декодирования (округление)...")
    decoded_tokens = model.decode_embeddings(gen_embs)
    decoded_text = tokenizer.decode(decoded_tokens[0], skip_special_tokens=True)
    print(f"   Декодированный текст (начальный/необученный): '{decoded_text}'")

    print("--- Проверка завершена успешно ---")

if __name__ == "__main__":
    verify_pipeline()
