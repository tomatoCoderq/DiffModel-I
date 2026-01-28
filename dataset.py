import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class MedicalDataset(Dataset):
    def __init__(self, texts, diagnoses, tokenizer_name="bert-base-uncased", max_length=128):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.texts = texts
        self.diagnoses = diagnoses
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        diagnosis = self.diagnoses[idx]

        # Токенизация клинической заметки
        text_enc = self.tokenizer(
            text, 
            max_length=self.max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        # Токенизация диагноза (для условия)
        diag_enc = self.tokenizer(
            diagnosis,
            max_length=32, # Диагноз обычно короче
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": text_enc["input_ids"].squeeze(0),
            "cond_ids": diag_enc["input_ids"].squeeze(0),
        }

import pandas as pd

def get_dataloader(data_path="medical_dataset.csv", batch_size=32, max_length=128):
    # Загружаем данные из CSV файла
    try:
        df = pd.read_csv(data_path)
        texts = df['clinical_note'].tolist()
        diagnoses = df['diagnosis'].tolist()
    except Exception as e:
        print(f"Ошибка при загрузке {data_path}: {e}. Используем фиктивные данные.")
        # Фиктивные данные в случае ошибки
        texts = [
            "Patient presents with acute chest pain and shortness of breath.",
            "Follow-up for chronic hypertension and type 2 diabetes management.",
            "Post-operative report for appendectomy. Wound healing well.",
            "Patient complains of persistent headache and blurred vision for two weeks."
        ] * 25
        diagnoses = [
            "Acute Coronary Syndrome",
            "Hypertension and Diabetes",
            "Post-Appendectomy",
            "Migraine or Increased Intracranial Pressure"
        ] * 25

    dataset = MedicalDataset(texts, diagnoses, max_length=max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    dl = get_dataloader()
    batch = next(iter(dl))
    print("Clinical Note IDs shape:", batch["input_ids"].shape)
    print("Diagnosis IDs shape:", batch["cond_ids"].shape)
