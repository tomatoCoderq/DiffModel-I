from dataclasses import dataclass

import torch
import torch.nn as nn
from tqdm import tqdm

from dataset import get_dataloader
from diffusion_wrapper import MedicalDiffusionModel

try:
    from training_ipc import fetch_stop_request, log_metric, update_status
except Exception:
    fetch_stop_request = None
    update_status = None
    log_metric = None


@dataclass
class TrainingConfig:
    batch_size: int = 32
    seq_len: int = 128
    d_model: int = 512
    epochs: int = 200
    learning_rate: float = 5e-5
    weight_decay: float = 1e-2
    data_path: str = "medical_dataset.csv"
    save_best_path: str = "medical_diffusion_best.pt"
    save_final_path: str = "medical_diffusion.pt"
    status_every_steps: int = 10
    device: str | None = None


def _resolve_config(config):
    cfg = TrainingConfig()
    if config is None:
        return cfg
    if isinstance(config, TrainingConfig):
        return config
    if isinstance(config, dict):
        for key, value in config.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
        return cfg
    raise TypeError("config must be dict, TrainingConfig, or None")


def _resolve_device(device):
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def train(config=None):
    cfg = _resolve_config(config)
    device = _resolve_device(cfg.device)

    model = MedicalDiffusionModel(d_model=cfg.d_model, max_seq_len=cfg.seq_len).to(device)
    dataloader = get_dataloader(data_path=cfg.data_path, batch_size=cfg.batch_size, max_length=cfg.seq_len)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    mse_loss = nn.MSELoss()

    best_loss = float('inf')

    print(f"Starting training on {device}...")

    stop_requested = False
    global_step = 0
    last_loss = None
    status_every = max(1, int(cfg.status_every_steps))

    if update_status:
        update_status(state="running", detail="training started", epoch=0, step=0, loss=None)

    for epoch in range(cfg.epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.epochs}")

        for batch in pbar:
            if fetch_stop_request and fetch_stop_request():
                stop_requested = True
                if update_status:
                    update_status(
                        state="stopping",
                        detail="stop requested",
                        epoch=epoch,
                        step=global_step,
                        loss=last_loss,
                    )
                break
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            cond_ids = batch["cond_ids"].to(device)

            # Мы переводим заметку и диагноз в пространство эмбеддингов самой модели, чтобы обучать денойзер без внешних энкодеров
            with torch.no_grad():
                x_start = model.backbone.embedding(input_ids)
                cond_emb = model.backbone.embedding(cond_ids).mean(dim=1)

            # Мы моделируем прямой шаг диффузии: выбираем время, добавляем шум и просим сеть восстановить его
            t = torch.randint(0, model.diffusion_pipeline.num_steps, (input_ids.shape[0],)).to(device)
            noise = torch.randn_like(x_start)
            x_noisy = model.diffusion_pipeline.q_sample(x_start, t, noise)
            noise_pred = model.get_noise_prediction(x_noisy, t, cond_emb)
            loss = mse_loss(noise_pred, noise)

            loss.backward()
            optimizer.step()

            last_loss = float(loss.item())
            global_step += 1
            pbar.set_postfix({"loss": f"{last_loss:.4f}"})
            if update_status and (global_step % status_every == 0):
                update_status(
                    state="running",
                    detail=f"epoch {epoch + 1}/{cfg.epochs}",
                    epoch=epoch + 1,
                    step=global_step,
                    loss=last_loss,
                )
            if log_metric and (global_step % status_every == 0):
                log_metric(epoch=epoch + 1, step=global_step, loss=last_loss)

        if stop_requested:
            break

        # Мы отслеживаем лучший лосс, чтобы сохранять устойчивые веса между эпохами
        avg_loss = loss.item()
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), cfg.save_best_path)

        if update_status:
            update_status(
                state="running",
                detail=f"epoch {epoch + 1}/{cfg.epochs} complete",
                epoch=epoch + 1,
                step=global_step,
                loss=last_loss,
            )
        if log_metric and last_loss is not None:
            log_metric(epoch=epoch + 1, step=global_step, loss=last_loss)

    # Мы сохраняем итоговые веса независимо от состояния раннего останова
    torch.save(model.state_dict(), cfg.save_final_path)
    if stop_requested:
        if update_status:
            update_status(
                state="stopped",
                detail="training stopped by request",
                epoch=epoch,
                step=global_step,
                loss=last_loss,
            )
        print(f"Training stopped early. Best loss: {best_loss:.4f}. Model saved.")
        return {"stopped": True, "best_loss": best_loss}

    if update_status:
        update_status(
            state="idle",
            detail="training finished",
            epoch=cfg.epochs,
            step=global_step,
            loss=last_loss,
        )
    print(f"Training complete. Best loss: {best_loss:.4f}. Model saved.")
    return {"stopped": False, "best_loss": best_loss}

if __name__ == "__main__":
    train()
