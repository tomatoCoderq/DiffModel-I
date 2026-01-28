import pandas as pd
import random

# Базовые шаблоны для генерации расширенного датасета
diagnoses_templates = {
    "Acute Coronary Syndrome": [
        "Patient presents with {severity} chest pain radiations to the {radiation}, shortness of breath, and {symptom}.",
        "ECG shows {finding} in leads {leads}. Clinical picture suggests ACS.",
        "Emergency admission for {severity} retrosternal pain and diaphoresis. {finding} noted on monitor."
    ],
    "Hypertension": [
        "Blood pressure readings consistently high: {bp_systolic}/{bp_low}. Patient is {symptom}.",
        "Follow-up for chronic hypertension. Prescribed {medication} dosage adjusted to {dose}mg.",
        "Routine check-up. BP measured at {bp_systolic}/{bp_low} mmHg. Discussion on lifestyle and {medication}."
    ],
    "Type 2 Diabetes": [
        "Patient reports {symptom_dm} and weight {weight_change}. HbA1c level is {hba1c}%.",
        "Routine diabetes management. Current medication: {medication_dm}. Fasting glucose is {glucose} mg/dL.",
        "Follow-up for Type 2 DM. Patient complains of {symptom_dm}. Advised on diet and exercise."
    ],
    "Bacterial Pneumonia": [
        "Cough with {sputum} sputum and high fever ({temp}C). Lung auscultation reveals {breath_sounds}.",
        "Radiography shows infiltration in the {lung_lobe}. Symptoms: productive cough and pleuritic pain.",
        "Patient presents with {severity} respiratory distress, fever, and crackles in the {lung_lobe}."
    ],
    "Migraine": [
        "Patient reports {severity} unilateral headache with {migraine_symptom}. Duration: {duration} hours.",
        "Recurrent migraine attacks accompanied by photophobia and {migraine_symptom}. Relieved by {medication_pain}.",
        "Chronic migraine management. Frequency of attacks: {frequency} times per month."
    ]
}

data_options = {
    "severity": ["acute", "severe", "crushing", "mild", "persistent"],
    "radiation": ["left arm", "neck", "jaw", "back"],
    "symptom": ["diaphoresis", "nausea", "syncope", "fatigue"],
    "finding": ["ST-segment elevation", "T-wave inversion", "ST-depression", "arrhythmia"],
    "leads": ["V1-V4", "I, II, aVF", "V5-V6", "III, aVR"],
    "bp_systolic": [145, 150, 160, 175, 180],
    "bp_low": [90, 95, 100, 110],
    "medication": ["Lisinopril", "Amlodipine", "Losartan", "Metoprolol"],
    "dose": [5, 10, 20, 40, 50],
    "symptom_dm": ["polydipsia", "polyuria", "blurred vision", "slow healing"],
    "weight_change": ["loss", "gain", "stability"],
    "hba1c": [7.2, 8.5, 9.1, 6.8, 10.2],
    "medication_dm": ["Metformin", "Jardiance", "Ozempic", "Glipizide"],
    "glucose": [140, 180, 220, 250, 155],
    "sputum": ["yellow-green", "rusty", "purulent", "thick"],
    "temp": [38.2, 38.9, 39.4, 40.1],
    "breath_sounds": ["crackles", "rhonchi", "decreased breath sounds"],
    "lung_lobe": ["right lower lobe", "left upper lobe", "bilateral bases"],
    "migraine_symptom": ["photophobia", "phonophobia", "nausea", "aura"],
    "duration": [4, 12, 24, 48, 72],
    "medication_pain": ["Sumatriptan", "Ibuprofen", "Excedrin"],
    "frequency": [2, 4, 8, 12]
}

def generate_expanded_dataset(count=512):
    dataset = []
    
    # Сначала добавим оригинальные данные
    original_df = pd.read_csv("medical_dataset.csv")
    for _, row in original_df.iterrows():
        dataset.append({"diagnosis": row["diagnosis"], "clinical_note": row["clinical_note"]})
    
    # Генерируем новые записи
    for _ in range(count - len(dataset)):
        diag = random.choice(list(diagnoses_templates.keys()))
        template = random.choice(diagnoses_templates[diag])
        
        # Заполняем шаблон случайными значениями
        params = {k: random.choice(v) for k, v in data_options.items()}
        note = template.format(**params)
        
        dataset.append({"diagnosis": diag, "clinical_note": note})
        
    df = pd.DataFrame(dataset)
    df.to_csv("expanded_medical_dataset.csv", index=False)
    print(f"Generated {len(df)} records in expanded_medical_dataset.csv")

if __name__ == "__main__":
    generate_expanded_dataset()
