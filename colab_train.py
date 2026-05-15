# ============================================================
# ToS Summarization Model - Google Colab Training Script
# Project: Analisis Klausul Risiko dan Peringkasan ToS
# Course: COMP6885001 - NLP | Bina Nusantara University
# ============================================================
#
# URUTAN MENJALANKAN DI COLAB:
# Cell 1a → Cell 1b → Cell 2 → Cell 3 → Cell 4 → Cell 5
# → Cell 6 → Cell 7 (Training) → Cell 8 (Test) → Cell 9 (Download)
#
# Pastikan: Runtime > Change runtime type > GPU (T4)
# ============================================================


# ============================================================
# CELL 1a - Install Libraries
# ============================================================
# Jalankan cell ini paling pertama. Tunggu sampai selesai.
#
# !pip install -q --upgrade transformers datasets rouge-score sentencepiece accelerate evaluate


# ============================================================
# CELL 1b - Download NLTK Data
# ============================================================
# import nltk
# nltk.download('punkt', quiet=True)
# nltk.download('stopwords', quiet=True)
# nltk.download('punkt_tab', quiet=True)
# print("Library siap!")


# ============================================================
# CELL 2 - Import & Konfigurasi
# ============================================================

import os
import re
import glob
import random
import json
import warnings
import numpy as np
import torch
import nltk

warnings.filterwarnings('ignore')

from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from datasets import Dataset
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Konfigurasi ─────────────────────────────────────────────
CONFIG = {
    "model_name":   "facebook/bart-large-cnn",

    "max_input_length":  1024,
    "max_target_length": 256,

    "batch_size":                  2,
    "gradient_accumulation_steps": 8,
    "num_epochs":                  3,
    "learning_rate":               2e-5,
    "warmup_ratio":                0.1,
    "weight_decay":                0.01,

    "train_split": 0.80,
    "max_samples": 3000,
    "seed":        42,

    # Path sesuai hasil extract dataset di Colab
    "data_dir":     "/content/dataset/text",
    "save_dir":     "/content/model",
    "results_path": "/content/training_results.json",
}

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device  : {device}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
    print(f"VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print("CONFIG siap!")


# ============================================================
# CELL 3 - Verifikasi Dataset
# ============================================================

def verify_dataset():
    files = glob.glob(os.path.join(CONFIG["data_dir"], "*.txt"))
    print(f"Jumlah file .txt : {len(files)}")
    if files:
        print(f"Contoh file      : {[os.path.basename(f) for f in files[:3]]}")
    else:
        print("PERINGATAN: Tidak ada file .txt ditemukan!")
        print(f"Pastikan path benar: {CONFIG['data_dir']}")

verify_dataset()


# ============================================================
# CELL 4 - Fungsi Preprocessing & Pseudo-Label
# ============================================================

def clean_text(text: str) -> str:
    """Bersihkan teks dari encoding artifact dan whitespace berlebih."""
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\s+', ' ', text).strip()
    lines = [ln.strip() for ln in text.split('\n') if len(ln.strip()) > 30]
    return ' '.join(lines) if lines else text


def create_pseudo_summary(text: str, num_sentences: int = 5) -> str:
    """
    Buat extractive summary dengan TF-IDF scoring sebagai pseudo-label.
    Dataset ToSDR tidak punya ground-truth summary, jadi kita buat sendiri
    dengan memilih kalimat paling penting menggunakan TF-IDF.
    Pakai regex split (bukan nltk) untuk menghindari konflik versi.
    """
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    if not sentences:
        return text[:500]

    if len(sentences) <= num_sentences:
        return ' '.join(sentences)

    try:
        tfidf = TfidfVectorizer(stop_words='english', max_features=200)
        matrix = tfidf.fit_transform(sentences)
        scores = np.asarray(matrix.sum(axis=1)).flatten()
        top_idx = sorted(np.argsort(scores)[-num_sentences:].tolist())
        return ' '.join(sentences[i] for i in top_idx)
    except Exception:
        return ' '.join(sentences[:num_sentences])


def load_dataset_from_dir(data_dir: str, max_samples: int = None):
    """Muat semua .txt dan buat pasangan (input_text, target_text)."""
    txt_files = glob.glob(os.path.join(data_dir, '*.txt'))
    print(f"Total file ditemukan : {len(txt_files)}")

    if max_samples and len(txt_files) > max_samples:
        random.shuffle(txt_files)
        txt_files = txt_files[:max_samples]
        print(f"Menggunakan sample   : {max_samples}")

    pairs, skipped = [], 0

    for i, filepath in enumerate(txt_files):
        if i % 500 == 0:
            print(f"  Memproses {i}/{len(txt_files)}...")

        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                raw = f.read()

            text = clean_text(raw)

            if len(text.split()) < 80:
                skipped += 1
                continue

            # Potong input (~800 kata ≈ 1024 token untuk BART)
            input_text = ' '.join(text.split()[:800])
            summary    = create_pseudo_summary(text, num_sentences=5)

            if len(summary.split()) < 15:
                skipped += 1
                continue

            pairs.append({'input_text': input_text, 'target_text': summary})

        except Exception:
            skipped += 1

    print(f"\nBerhasil : {len(pairs)} pasangan")
    print(f"Dilewati : {skipped} file")
    return pairs

print("Fungsi preprocessing siap!")


# ============================================================
# CELL 5 - Fungsi Tokenisasi & ROUGE
# ============================================================

def tokenize_batch(examples, tokenizer):
    """
    Tokenize input dan target.
    Menggunakan parameter text_target (menggantikan as_target_tokenizer
    yang sudah deprecated di transformers versi baru).
    """
    model_inputs = tokenizer(
        examples['input_text'],
        max_length=CONFIG["max_input_length"],
        truncation=True,
        padding=False,
    )

    # text_target: cara baru tokenize label di transformers >= 4.41
    labels = tokenizer(
        text_target=examples['target_text'],
        max_length=CONFIG["max_target_length"],
        truncation=True,
        padding=False,
    )

    model_inputs['labels'] = labels['input_ids']
    return model_inputs


def compute_metrics(eval_pred, tokenizer):
    """Hitung ROUGE-1, ROUGE-2, ROUGE-L."""
    scorer_obj = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
    )

    predictions, labels = eval_pred

    # Decode prediksi
    decoded_preds = tokenizer.batch_decode(
        predictions, skip_special_tokens=True
    )

    # Ganti -100 (padding label) sebelum decode
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    r1, r2, rL = [], [], []
    for pred, ref in zip(decoded_preds, decoded_labels):
        pred = pred.strip()
        ref  = ref.strip()
        if not pred or not ref:
            continue
        s = scorer_obj.score(ref, pred)
        r1.append(s['rouge1'].fmeasure)
        r2.append(s['rouge2'].fmeasure)
        rL.append(s['rougeL'].fmeasure)

    return {
        'rouge1': round(float(np.mean(r1)) if r1 else 0.0, 4),
        'rouge2': round(float(np.mean(r2)) if r2 else 0.0, 4),
        'rougeL': round(float(np.mean(rL)) if rL else 0.0, 4),
    }

print("Fungsi tokenisasi & ROUGE siap!")


# ============================================================
# CELL 6 - Persiapan Training
# ============================================================

# Load tokenizer & model
print(f"Mengunduh model: {CONFIG['model_name']} ...")
tokenizer = BartTokenizer.from_pretrained(CONFIG["model_name"])
model     = BartForConditionalGeneration.from_pretrained(CONFIG["model_name"])
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters : {total_params:,}")

# Load & proses dataset
print("\nMemuat dataset...")
pairs = load_dataset_from_dir(CONFIG["data_dir"], CONFIG["max_samples"])

# Split 80:20
random.shuffle(pairs)
split_idx  = int(len(pairs) * CONFIG["train_split"])
train_data = pairs[:split_idx]
test_data  = pairs[split_idx:]

print(f"\nSplit dataset:")
print(f"  Train : {len(train_data)} sampel (80%)")
print(f"  Test  : {len(test_data)} sampel  (20%)")

# Buat HuggingFace Dataset & tokenize
train_ds = Dataset.from_list(train_data)
test_ds  = Dataset.from_list(test_data)

fn = lambda x: tokenize_batch(x, tokenizer)

print("\nTokenizing...")
train_tok = train_ds.map(
    fn, batched=True,
    remove_columns=train_ds.column_names,
    desc="Train"
)
test_tok = test_ds.map(
    fn, batched=True,
    remove_columns=test_ds.column_names,
    desc="Test"
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100,
)

print("Persiapan selesai!")


# ============================================================
# CELL 7 - Training (Estimasi 1-2 jam di Colab T4)
# ============================================================

os.makedirs(CONFIG["save_dir"], exist_ok=True)

training_args = Seq2SeqTrainingArguments(
    output_dir=CONFIG["save_dir"],

    num_train_epochs=CONFIG["num_epochs"],
    per_device_train_batch_size=CONFIG["batch_size"],
    per_device_eval_batch_size=CONFIG["batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],

    warmup_steps=100,
    weight_decay=CONFIG["weight_decay"],
    learning_rate=CONFIG["learning_rate"],

    eval_strategy="epoch",       # versi baru (evaluation_strategy deprecated)
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    greater_is_better=True,
    save_total_limit=2,

    predict_with_generate=True,
    generation_max_length=CONFIG["max_target_length"],
    generation_num_beams=4,

    fp16=torch.cuda.is_available(),
    logging_steps=50,
    report_to="none",
    seed=CONFIG["seed"],
    dataloader_num_workers=2,
)

def compute_metrics_wrapped(eval_pred):
    return compute_metrics(eval_pred, tokenizer)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=test_tok,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics_wrapped,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

print("Mulai training...")
print("=" * 50)
train_result = trainer.train()

# Evaluasi final
print("\nEvaluasi final...")
results = trainer.evaluate()
print(f"\n  ROUGE-1 : {results.get('eval_rouge1', 0):.4f}")
print(f"  ROUGE-2 : {results.get('eval_rouge2', 0):.4f}")
print(f"  ROUGE-L : {results.get('eval_rougeL', 0):.4f}")

# Simpan model
trainer.save_model(CONFIG["save_dir"])
tokenizer.save_pretrained(CONFIG["save_dir"])

results['train_runtime']  = train_result.metrics.get('train_runtime', 0)
results['train_samples']  = len(train_data)
results['test_samples']   = len(test_data)

with open(CONFIG["results_path"], 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nModel tersimpan di : {CONFIG['save_dir']}")
print("Training SELESAI!")



# ============================================================
# CELL 8 - Test Model (opsional, jalankan setelah training)
# ============================================================

def test_model(sample_text: str = None):
    """Test model dengan contoh teks ToS."""
    if sample_text is None:
        sample_text = (
            "By using our Service, you agree that we may collect and store your "
            "personal data including name, email, location, device identifiers, "
            "and browsing history. We may share this information with third-party "
            "advertising partners without your explicit consent. You agree to "
            "binding arbitration and waive your right to participate in class "
            "action lawsuits. We may terminate your account at any time without "
            "prior notice or liability."
        )

    print(f"Input ({len(sample_text.split())} kata):")
    print(sample_text)

    inputs = tokenizer(
        sample_text,
        max_length=1024,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    model.eval()
    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=256,
            min_length=30,
            num_beams=4,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\nSummary:")
    print(summary)
    return summary

# Uncomment untuk test:
# test_model()


# ============================================================
# CELL 9 - Download Model ke Komputer
# ============================================================
# Jalankan cell ini untuk download model setelah training selesai.
# File model akan ter-download otomatis ke komputer kamu.
#
# import shutil
# from google.colab import files
#
# # Zip folder model
# shutil.make_archive("/content/model_tos", 'zip', "/content/model")
# print("Model di-zip!")
#
# # Download
# files.download("/content/model_tos.zip")
# print("Download dimulai!")
#
# Setelah download:
# 1. Extract model_tos.zip
# 2. Rename folder hasil extract menjadi 'model'
# 3. Pindahkan ke D:\NLP prject\model\
# 4. Jalankan: streamlit run app.py
