# Holmes miniGPT — Character-Level Transformer from Scratch

A small, self-contained **character-level GPT-style Transformer** implemented from scratch in **PyTorch**.
It trains on the Sherlock Holmes stories (`holmes.txt`) and generates new text in the same style by predicting the **next character**.

The goal of the project was to understand the core building blocks behind modern large language models: token and positional embeddings, causal self-attention, multi-head attention, residual connections, layer normalization and autoregressive sampling.

## Files

- `transformer.py`: model, training loop and text generation
- `params.yaml`: hyperparameters and configuration
- `holmes.txt`: training corpus (~590 KB of text)

## How It Works

1. **Data preparation:** reads `holmes.txt`, filters it to ASCII, builds a character vocabulary and encodes the text as integer tokens.
2. **Train/validation split:** the first 90% of the text is used for training and the remaining 10% for validation.
3. **Model:** a decoder-only Transformer:
   - token and positional embeddings
   - `n_layer` Transformer blocks, each with **multi-head causal self-attention** and a feed-forward network, both using residual connections and pre-LayerNorm
   - final LayerNorm and a linear head over the vocabulary
4. **Training:** AdamW optimizer with cross-entropy loss on next-character prediction. Train and validation loss are reported periodically, and checkpoints are saved to `checkpoint.pt`.
5. **Generation:** autoregressive sampling of 500 new characters from the trained model.

## Configuration (`params.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `n_embd` | 192 | embedding dimension |
| `n_head` | 6 | attention heads per block |
| `n_layer` | 4 | number of Transformer blocks |
| `block_size` | 128 | context length (characters) |
| `batch_size` | 64 | sequences per batch |
| `dropout` | 0.2 | dropout rate |
| `learning_rate` | 3e-4 | AdamW learning rate |
| `max_iters` | 300 | training iterations |
| `eval_interval` / `eval_iters` | 100 / 40 | how often and over how many batches loss is estimated |
| `checkpoint_interval` | 500 | save a checkpoint every N iterations (0 disables it) |
| `device` | auto | `auto`, `cuda` or `cpu` |

With the default configuration the model has about **1.8M parameters**. Increase `max_iters` (for example to 5000) for noticeably better text quality.

## Running

Requirements: Python 3.9+, PyTorch and PyYAML.

```bash
pip install torch pyyaml
python transformer.py
```

The script prints the number of parameters and the train/validation loss during training, then a generated text sample.

## Tech Stack

Python · PyTorch · PyYAML
