# Embeddings and Sliding Windows

**Escuela Colombiana de Ingeniería Julio Garavito**
**Student:** Santiago Botero García

## Overview

The goal of this assignment is to reproduce the embedding pipeline described in the chapter and extend it with conceptual analysis and experimentation.

## Files Used (As Required)

Only the following two files were downloaded from the official repository:

- Notebook:
  [https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/ch02.ipynb](https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/ch02.ipynb)

- Text file:
  [https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt](https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt)

No additional files were used.

## What This Work Implements

The notebook reconstructs the **core embedding pipeline** for training an autoregressive language model:

1. Load raw text
2. Tokenize using GPT-style encoding
3. Create sliding window input-target pairs
4. Build a PyTorch Dataset and DataLoader
5. Convert token IDs into trainable embeddings
6. Analyze how embeddings encode meaning
7. Run a small experiment varying `max_length` and `stride`

This mirrors the data preparation stage required before training a GPT-style transformer.

## Conceptual Explanations Included

The notebook contains multiple Markdown cells with **original explanations** covering:

### Why Tokenization Matters

- Defines vocabulary size
- Determines embedding matrix shape
- Establishes atomic prediction units

### Why Sliding Windows Are Necessary

- Convert one long text into thousands of supervised samples
- Enable next-token prediction learning
- Control training signal density via stride

### Why Embeddings Encode Meaning

Embeddings encode meaning because they are optimized through gradient descent to minimize next-token prediction loss.

If two tokens appear in similar contexts:

- They receive similar gradient updates
- Their vectors move closer in embedding space

Mathematically:

- The embedding matrix is a trainable parameter
- It is equivalent to a linear layer applied to one-hot vectors
- It is updated through backpropagation

Meaning is therefore not symbolic -
it emerges geometrically in vector space.

### Connection to Neural Network Concepts

Embeddings are:

- Trainable parameters
- Part of the model architecture
- The first learned layer in a transformer

They transform discrete symbolic input into continuous representations that can be processed by linear algebra operations.

Without embeddings:

- Neural networks cannot operate on language.
- There is no differentiable structure.
- There is no similarity geometry.

### Relevance to Agentic Systems

Embeddings are foundational in modern agent architectures:

- Vector databases
- Retrieval-Augmented Generation (RAG)
- Semantic memory
- Similarity search
- Context grounding

Pipeline example:

Text &rarr; Embedding &rarr; Vector Store &rarr; Similarity Search &rarr; Retrieved Context &rarr; LLM

Embeddings enable semantic retrieval and memory indexing, which are essential in agentic AI systems.

# Experiment

The notebook includes an experiment modifying:

- `max_length`
- `stride`

### Observations

- Smaller stride &rarr; more overlap &rarr; more samples
- Larger stride &rarr; fewer samples
- Overlap improves contextual continuity
- Higher overlap increases compute cost

### Why Overlap Is Useful

Language dependencies often span across chunk boundaries.

Overlap ensures:

- Transitions are learned multiple times
- Gradients are smoother
- Context continuity is preserved

This demonstrates understanding of the trade-off between:

Training efficiency
and
Statistical learning quality

## Learning Outcomes

By completing this Work, we understand:

- How raw text becomes structured training data
- Why tokenization defines representational granularity
- How sliding windows create supervision
- How embeddings transform discrete tokens into geometry
- Why embeddings encode semantic structure
- How this pipeline connects to modern agentic systems

This notebook represents the **first concrete step toward building a transformer-based LLM from scratch.**
