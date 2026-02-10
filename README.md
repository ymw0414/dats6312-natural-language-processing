# Political Text Classification

Binary classification of U.S. Congressional speeches as **Democrat** or **Republican** using NLP, comparing a traditional TF-IDF baseline with a fine-tuned RoBERTa transformer.

[Live Demo](https://final-project-minwooyoo-bcbbuifbzcrqmmalx8z3gn.streamlit.app/)

## Results

| Model | Data Filter | Epochs | Accuracy | F1 |
|-------|------------|--------|----------|-----|
| TF-IDF + Logistic Regression | Yes | N/A | 0.694 | 0.692 |
| RoBERTa-base | Yes | 1 | 0.692 | 0.679 |
| RoBERTa-base | Yes | 2 | 0.715 | 0.699 |
| **RoBERTa-base** | **Yes** | **3** | **0.729** | **0.713** |

> Data filter: paragraph-level quality filtering (>=2 sentences, >=200 characters), resulting in ~0.39M speech samples from the 1980s (97th-100th Congress).

## Repository Structure

```
.
├── Code/
│   ├── utils/                        # Shared utility modules
│   │   ├── data_loader.py            #   CongressionalDataLoader class
│   │   ├── text_processing.py        #   TextPreprocessor class
│   │   ├── evaluation.py             #   ModelEvaluator class
│   │   └── device.py                 #   PyTorch device detection
│   ├── 01_load_speeches.py           # Parse raw speech text files
│   ├── 02_merge_speaker_map.py       # Parse speaker metadata
│   ├── 03_add_party_label.py         # Merge speeches with party labels
│   ├── 04_preprocess_text.py         # Clean text and filter to 1980s
│   ├── 05_train_baseline.py          # TF-IDF + Logistic Regression
│   ├── 06_tokenize_and_concat.py     # RoBERTa tokenization
│   ├── 07_train_roberta.py           # Fine-tune RoBERTa-base
│   └── 08_streamlit_app.py           # Local Streamlit demo
├── Streamlit_Demo_App/               # Cloud-deployable Streamlit app
├── scripts/
│   └── download_data.py              # Download raw data from Stanford
├── Presentation/                     # Slide deck (PDF)
├── Proposal/                         # Project proposal (PDF)
├── Report/                           # Final reports (PDF)
├── run_pipeline.py                   # Single entry point for full pipeline
├── requirements.txt                  # Python dependencies
└── .gitignore
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download data

The project uses the [Congressional Record dataset](https://data.stanford.edu/congress_text) by Gentzkow, Shapiro, and Taddy (Stanford).

**Option A** — Automated download:

```bash
python scripts/download_data.py --out_dir data/raw/congressional_speeches
```

**Option B** — Manual download:

1. Visit https://data.stanford.edu/congress_text
2. Download the **hein-daily** speech files and **SpeakerMap** files
3. Place all `.txt` files in `data/raw/congressional_speeches/`

Expected files:

```
data/raw/congressional_speeches/
├── speeches_043.txt
├── speeches_044.txt
├── ...
├── speeches_111.txt
├── SpeakerMap_043.txt
├── SpeakerMap_044.txt
├── ...
└── SpeakerMap_111.txt
```

### 3. Run the pipeline

**Full pipeline** (requires GPU for step 7):

```bash
python run_pipeline.py --raw_dir data/raw/congressional_speeches
```

**Baseline only** (no GPU needed):

```bash
python run_pipeline.py --raw_dir data/raw/congressional_speeches --skip_roberta
```

### 4. Launch the demo

```bash
streamlit run Code/08_streamlit_app.py
```

## Pipeline Overview

```
Raw speech text files ─── 01_load_speeches.py ───► speeches_merged.parquet
                                                        │
Raw SpeakerMap files ──── 02_merge_speaker_map.py ► speaker_map.parquet
                                                        │
                          03_add_party_label.py ◄───────┘
                                │
                                ▼
                    speeches_with_party.parquet
                                │
                          04_preprocess_text.py
                                │
                                ▼
                  speeches_clean_1980s_paragraph.parquet
                        │                    │
              05_train_baseline.py    06_tokenize_and_concat.py
                        │                    │
                        ▼                    ▼
               Logistic Regression    07_train_roberta.py
               (TF-IDF baseline)             │
                                             ▼
                                   Fine-tuned RoBERTa
                                             │
                                   08_streamlit_app.py
```

## Individual Script Usage

Each script accepts command-line arguments and can be run independently:

```bash
# Step 1: Load speeches
python Code/01_load_speeches.py --raw_dir data/raw/congressional_speeches

# Step 2: Load speaker metadata
python Code/02_merge_speaker_map.py --raw_dir data/raw/congressional_speeches

# Step 3: Attach party labels
python Code/03_add_party_label.py

# Step 4: Preprocess text (filter to 1980s, clean)
python Code/04_preprocess_text.py

# Step 5: Train baseline model
python Code/05_train_baseline.py

# Step 6: Tokenize for transformer
python Code/06_tokenize_and_concat.py

# Step 7: Fine-tune RoBERTa (GPU recommended)
python Code/07_train_roberta.py --epochs 3 --batch_size 32 --lr 1e-5
```

Run `python Code/<script>.py --help` for all available options.

## Tech Stack

- **Data**: pandas, pyarrow
- **Baseline ML**: scikit-learn (TF-IDF, Logistic Regression)
- **Deep Learning**: PyTorch, HuggingFace Transformers (RoBERTa-base)
- **Tokenization**: HuggingFace Datasets
- **Demo**: Streamlit
- **Evaluation**: scikit-learn, matplotlib

## Data Source

Gentzkow, M., Shapiro, J. M., and Taddy, M. (2018). "Congressional Record for the 43rd-114th Congresses: Parsed Speeches and Phrase Counts." Stanford Libraries. https://data.stanford.edu/congress_text

## License

This project was developed as part of the DATS 6312 (Natural Language Processing) course at George Washington University.
