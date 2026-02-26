# Mini Transformer (Character-Level GPT) — Sherlock Holmes Text Generation

A small, self-contained **character-level Transformer (GPT-style)** implemented in **PyTorch**.  
It trains on `holmes.txt` and generates new text by predicting the **next character**.

## Files
- `transformer.py` — model + training loop + text generation
- `params.yaml` — hyperparameters / config
- `holmes.txt` — training corpus

## What this project does
- Reads `holmes.txt` and builds a **character vocabulary**
- Splits the dataset into **90% train / 10% validation**
- Trains a GPT-like Transformer with **causal self-attention** (cross-entropy next-character objective)
- Prints periodic train/val loss
- Generates a text sample at the end

> Note: the text is filtered to ASCII (`encode("ascii", "ignore")`) before building the vocabulary.

## Requirements
- Python 3.9+ recommended
- PyTorch
- PyYAML

Install:
```bash
pip install torch pyyaml
