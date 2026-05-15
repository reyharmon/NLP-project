# NLP Project Progress

**Judul:** Analisis Klausul Risiko dan Peringkasan Terms of Service Menggunakan NLP  
**Mata Kuliah:** COMP6885001 - Natural Language Processing  
**Tim:** Andrey Apriliady · Ezra Mayurga · Keanu Stadeva  
**Dosen:** Mohammad Faisal Riftiarrasyid, S.Kom., M.Kom.  
**Deadline:** Week 14

---

## Status Keseluruhan

| Komponen | Status | Keterangan |
|----------|--------|-----------|
| Proposal | Done | Dikumpulkan |
| Dataset | Ready | 9,491 file .txt |
| Pemilihan Model | Done | facebook/bart-large-cnn |
| Training Script | Done | colab_train.py |
| Streamlit App | Done | app.py |
| Fine-tuning Colab | Belum | Perlu upload dataset ke Drive |
| Download Model | Belum | Setelah training selesai |
| Test Lokal | Belum | Setelah download model |
| Deploy Streamlit | Belum | Setelah test lokal |
| Laporan | Belum | |
| Video Demo | Belum | |

---

## Task List

### Segera Dikerjakan
- [ ] Install dependencies: pip install -r requirements.txt
- [ ] Upload folder dataset/ ke Google Drive
- [ ] Buka Google Colab, set Runtime ke GPU (T4)
- [ ] Jalankan colab_train.py di Colab
- [ ] Download folder model/ dari Google Drive
- [ ] Letakkan model/ di project folder
- [ ] Test lokal: streamlit run app.py

### Pengembangan Lanjutan
- [ ] Push model ke HuggingFace Hub (untuk Streamlit Cloud)
- [ ] Deploy app ke Streamlit Cloud
- [ ] Tulis laporan (10-15 halaman)
- [ ] Rekam video demo (< 5 menit)

### Laporan (Required Structure)
- [ ] Title Page
- [ ] Abstract
- [ ] Introduction
- [ ] Related Work (min. 5 referensi)
- [ ] Methodology
- [ ] Implementation & Results (ROUGE scores, screenshots)
- [ ] Discussion & Limitations
- [ ] Conclusion & Future Work
- [ ] References (IEEE/APA)
- [ ] Appendix (kontribusi tim, code snippets, screenshots)

---

## Arsitektur Model

Model   : facebook/bart-large-cnn (406M parameters)
Task    : Abstractive Text Summarization
Input   : Raw ToS / Privacy Policy text (max 1024 tokens / ~700-800 kata)
Output  : Summary (max 256 tokens / ~150-200 kata)

Fine-tuning Strategy:
  - Dataset : 9,491 file .txt (ToSDR corpus)
  - Pseudo-labels : TF-IDF extractive summaries (top-5 kalimat)
  - Split  : 80% train / 20% test
  - Epochs : 3 (early stopping)
  - Batch  : 2 (effective 16 dengan gradient accumulation)
  - LR     : 2e-5

Evaluasi: ROUGE-1, ROUGE-2, ROUGE-L

---

## Struktur File

D:\NLP prject\
  dataset/           - 9,491 file .txt ToS (dari Kaggle ToSDR)
  materi/            - Slide kuliah NLP
  model/             - Model setelah training (download dari Colab)
  colab_train.py     - Script training di Google Colab
  app.py             - Streamlit web application
  requirements.txt   - Python dependencies
  progress.md        - File ini
  PROPOSAL NLP.md    - Proposal proyek

---

## Log Update

| Tanggal | Update |
|---------|--------|
| 2026-05-15 | Proposal dibuat, dataset ready, model dipilih, script training & Streamlit app selesai dibuat |
