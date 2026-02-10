# Code Directory

This directory contains the core implementation of the NLP pipeline.

Each script represents a single stage of the pipeline and is designed to be run
independently, with inputs produced by earlier stages.

## Execution Order

Scripts are numbered to reflect the recommended execution sequence:

```
01_load_speeches.py        → Load raw speech text
02_merge_speaker_map.py    → Load speaker metadata
03_add_party_label.py      → Merge speeches with party labels
04_preprocess_text.py      → Clean and filter text
05_train_baseline.py       → Train TF-IDF + Logistic Regression
06_tokenize_and_concat.py  → Tokenize for RoBERTa
07_train_roberta.py        → Fine-tune RoBERTa-base
08_streamlit_app.py        → Interactive Streamlit demo
```

## Utility Modules

Shared logic is organized in the `utils/` package:

| Module | Class / Function | Purpose |
|--------|-----------------|---------|
| `data_loader.py` | `CongressionalDataLoader` | Parse raw speech and SpeakerMap files |
| `text_processing.py` | `TextPreprocessor` | Unicode normalization, whitespace cleaning, paragraph filtering |
| `evaluation.py` | `ModelEvaluator` | Compute metrics, save reports, plot confusion matrices |
| `device.py` | `get_device()` | Detect best available PyTorch device (CUDA > MPS > CPU) |

## Data Paths

All scripts avoid hardcoded absolute paths.
Raw data locations and output directories are specified via command-line arguments.

Standard directory layout:

```
data/
├── raw/
│   └── congressional_speeches/
│       ├── speeches_043.txt
│       ├── SpeakerMap_043.txt
│       └── ...
└── processed/
    ├── speeches_merged.parquet
    ├── speaker_map.parquet
    ├── speeches_with_party.parquet
    └── speeches_clean_1980s_paragraph.parquet
```

Example usage:

```bash
python Code/01_load_speeches.py --raw_dir data/raw/congressional_speeches
```

Outputs from each stage are written to `data/processed/` by default.

## Notes

- Transformer fine-tuning assumes GPU availability.
- Experimental variants were removed to keep the codebase focused and reproducible.
