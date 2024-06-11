---
layout: post
title:  "GPT2 from scratch: Concept, architecture, training"
date:   2024-06-11 12:54:15 +0800
categories: jekyll update
---

I aim to replicate the GPT2 architecture and perform training on a dataset. The purpose of this exercise is to better understand the transformer architecture, its code and training in pytorch. 

### State of GPT2
Parameters inside the GPT2 model which I got from huggingface GPT2 model. The main thing to notice from the table is that GPT2 model has a vocabulary size of 50257 tokens and positional encoding of 1024 (which is used for context size). The vector size used in this model is 768 which are used to learn and store these embeddings. 

```markdown
| Layer/Component                          | Parameter                      | Shape         | Explanation |
|------------------------------------------|--------------------------------|---------------|-------------|
| `transformer.wte.weight`                 | Embedding weights              | [50257, 768]  | 50257 tokens in the vocabulary, each represented by a 768-dimensional embedding vector |
| `transformer.wpe.weight`                 | Positional encoding weights    | [1024, 768]   | 1024 possible positions in the input sequence, each represented by a 768-dimensional positional encoding |
| `transformer.h.0.ln_1.weight`            | Layer normalization weights    | [768]         | 768-dimensional vector for scaling each input feature |
| `transformer.h.0.ln_1.bias`              | Layer normalization bias       | [768]         | 768-dimensional vector for shifting each input feature |
| `transformer.h.0.attn.c_attn.weight`     | Attention query/key/value weights | [768, 2304]   | 768 input features transformed to 2304 features (768 * 3) for query, key, and value vectors (3 * 768) |
| `transformer.h.0.attn.c_attn.bias`       | Attention query/key/value bias | [2304]        | 2304-dimensional bias vector for the combined query, key, and value vectors |
| `transformer.h.0.attn.c_proj.weight`     | Attention output projection weights | [768, 768]    | 768 input features transformed back to 768 output features |
| `transformer.h.0.attn.c_proj.bias`       | Attention output projection bias | [768]        | 768-dimensional bias vector for the output projection |
| `transformer.h.0.ln_2.weight`            | Layer normalization weights    | [768]         | 768-dimensional vector for scaling each input feature |
| `transformer.h.0.ln_2.bias`              | Layer normalization bias       | [768]         | 768-dimensional vector for shifting each input feature |
| `transformer.h.0.mlp.c_fc.weight`        | MLP fully connected weights    | [768, 3072]   | 768 input features expanded to 3072 features |
| `transformer.h.0.mlp.c_fc.bias`          | MLP fully connected bias       | [3072]        | 3072-dimensional bias vector for the fully connected layer |
| `transformer.h.0.mlp.c_proj.weight`      | MLP output projection weights  | [3072, 768]   | 3072 input features compressed back to 768 output features |
| `transformer.h.0.mlp.c_proj.bias`        | MLP output projection bias     | [768]         | 768-dimensional bias vector for the output projection |
| `transformer.h.1.ln_1.weight`            | Layer normalization weights    | [768]         | 768-dimensional vector for scaling each input feature |
| `transformer.h.1.ln_1.bias`              | Layer normalization bias       | [768]         | 768-dimensional vector for shifting each input feature |
| `transformer.h.1.attn.c_attn.weight`     | Attention query/key/value weights | [768, 2304]   | 768 input features transformed to 2304 features (768 * 3) for query, key, and value vectors (3 * 768) |
| `transformer.h.1.attn.c_attn.bias`       | Attention query/key/value bias | [2304]        | 2304-dimensional bias vector for the combined query, key, and value vectors |
| `transformer.h.1.attn.c_proj.weight`     | Attention output projection weights | [768, 768]    | 768 input features transformed back to 768 output features |
| `transformer.h.1.attn.c_proj.bias`       | Attention output projection bias | [768]        | 768-dimensional bias vector for the output projection |
| `transformer.h.1.ln_2.weight`            | Layer normalization weights    | [768]         | 768-dimensional vector for scaling each input feature |
| `transformer.h.1.ln_2.bias`              | Layer normalization bias       | [768]         | 768-dimensional vector for shifting each input feature |
| `transformer.h.1.mlp.c_fc.weight`        | MLP fully connected weights    | [768, 3072]   | 768 input features expanded to 3072 features |
| `transformer.h.1.mlp.c_fc.bias`          | MLP fully connected bias       | [3072]        | 3072-dimensional bias vector for the fully connected layer |
| `transformer.h.1.mlp.c_proj.weight`      | MLP output projection weights  | [3072, 768]   | 3072 input features compressed back to 768 output features |
| `transformer.h.1.mlp.c_proj.bias`        | MLP output projection bias     | [768]         | 768-dimensional bias vector for the output projection |
| ...                                      | ...                            | ...           | ... |
| `transformer.h.11.mlp.c_proj.bias`       | MLP output projection bias     | [768]         | 768-dimensional bias vector for the output projection in the 12th layer |
| `transformer.ln_f.weight`                | Final layer normalization weights | [768]      | 768-dimensional vector for scaling each final output feature |
| `transformer.ln_f.bias`                  | Final layer normalization bias | [768]         | 768-dimensional vector for shifting each final output feature |
| `lm_head.weight`                         | Language model head weights    | [50257, 768]  | 768-dimensional output features for each of the 50257 tokens in the vocabulary |
```

GPT2 architecture is based on the transformer model from the "Attention is all you need" paper.
![Attention](/my-blog/images/GPT2-arch.png)

GPT-2 is a decoder only transformer model. Which means the following things:
1. GPT-2 generates text in an autoregressive manner, meaning it generates one token at a time and uses previously generated tokens as context for generating the next token.
2. GPT-2 does not encode an entire input sequence at once before generating an output. Instead, it builds the output sequence step-by-step.
3. It employs self-attention within the decoder blocks to allow each token to attend to previous tokens in the sequence. This ensures that each token can only attend to previous tokens and not future ones
4. The model is unidirectional, meaning it processes tokens from left to right.

### Config and classes

First step is to define the config class. Which will set the initial parameters like block size, vocab size, number of layers, attention heads and embedding dimensionality.
```
@dataclass
class GPTConfig:
    block_size: int= 246
    vocab_size: int = 65
    n_layer: int=6
    n_head: int = 6
    n_embd: int = 384
```

Next we define the GPT class. This class will consist of Word Token Embeddings (wte), Word Positional Embeddings (wpe), Transformer blocks (h) and Layer Normalization (ln_f).
- wte: Embedding layer that converts each token in the vocabulary to a n_embd-dimensional embedding vector.
- wpe: Embedding layer that assigns a unique n_embd-dimensional vector to each position in the input sequence.
- h: List of transformer blocks, where each block contains layers for self-attention and feed-forward neural networks.
- ln_f: Applies layer normalization to the final hidden states of the transformer to stabilize and accelerate training.
```

class GPT(nn.modules):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocal_size, config.n_embd), #weights of token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd), # weights of the positional embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.ln_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)
```

