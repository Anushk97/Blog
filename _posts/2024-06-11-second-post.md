---
layout: post
title:  "GPT-2 from scratch: Concept, architecture, training 😎"
date:   2024-06-11 12:54:15 +0800
categories: jekyll update
---

I aim to replicate the GPT-2 architecture and perform training on a dataset. The purpose of this exercise is to better understand the transformer architecture, its code and training in pytorch. 
Moreover, GPT-2 came out in 2019 from this [paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) which laid the foundation to more complex models using multi headed attention. So understanding this architecture is crucial.

### State of GPT-2 for 124M model ⛵ 
Parameters inside the GPT-2 model which I got from huggingface GPT-2 model. The main thing to notice from the table is that GPT-2 model has a vocabulary size of 50257 tokens and positional encoding of 1024 (which is used for context size). The vector size used in this model is 768 which are used to learn and store these embeddings. 

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

GPT-2 architecture is based on the transformer model from the "Attention is all you need" paper.
![Attention](/my-blog/images/GPT2-arch.png)

GPT-2 is a decoder only transformer model. Which means the following things:
1. GPT-2 generates text in an autoregressive manner, meaning it generates one token at a time and uses previously generated tokens as context for generating the next token.
2. GPT-2 does not encode an entire input sequence at once before generating an output. Instead, it builds the output sequence step-by-step.
3. It employs self-attention within the decoder blocks to allow each token to attend to previous tokens in the sequence. This ensures that each token can only attend to previous tokens and not future ones
4. The model is unidirectional, meaning it processes tokens from left to right.

### Config and classes 🏮 
To better understand the classes, we will work work backwards. That is first define the GPT class, then go one layer deep into transformer block, then MLP and then the Attention operation.

First step is to define the config class. Which will set the initial parameters like block size, vocab size, number of layers, attention heads and embedding dimensionality.
- block_size: Size of the input sequence block that the model processes at once. 
- vocab_size: Unique token that the model can recognize
- n_layers: Number of transformer layers
- n_head: Number of attention heads in each transformer layer.
- n_embd: The dimensionality of the embeddings (the size of the hidden layers).

These parameters are taken from the table above.
```
@dataclass
class GPTConfig:
    block_size: int= 1024
    vocab_size: int = 50527
    n_layer: int= 12
    n_head: int = 12
    n_embd: int = 768
```

---
#### 👉 GPT Class

This class will consist of Word Token Embeddings (wte), Word Positional Embeddings (wpe), Transformer blocks (h) and Layer Normalization (ln_f).
- wte: Embedding layer that converts each token in the vocabulary to a n_embd-dimensional embedding vector. Word embeddings can be better understood with [tiktokenizer](https://tiktokenizer.vercel.app/)
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
**from the docs..**

- [nn.modules](https://pytorch.org/docs/stable/generated/torch.nn.Module.html): Base class for all neural network modules
- [nn.ModuleDict](https://pytorch.org/docs/stable/generated/torch.nn.ModuleDict.html): Can be indexed like a regular Python dictionary
- [nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html): Applies a linear transformation to the incoming data: 
y=xA 
T
 +b.
- [nn.Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html): A simple lookup table that stores embeddings of a fixed dictionary and size.
- [nn.ModuleList](https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html): Can be indexed like a regular Python list
- [nn.LayerNorm](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html): Applies Layer Normalization over a mini-batch of inputs.

---
#### 👉 Transformer block

Next a single transformer Block which will consist of self attention modules followed by Multi Layer Perceptron (MLP) and layer normalization applied before each component. 

- self.ln_1 (Layer Normalization 1): Normalizes the input to the attention layer, which helps stabilize and accelerate training.
- self.attn (Causal Self-Attention): A self-attention mechanism that allows the model to focus on different parts of the input sequence. The attention mechanism is masked to prevent attending to future tokens, ensuring the autoregressive property.
- self.ln_2 (Layer Normalization 2): Normalizes the input to the MLP layer.
- self.mlp (Multi-Layer Perceptron): A feed-forward neural network that further processes the output from the attention layer.

```
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd) # layer norm 1
        self.attn = CasualSelfAttention(self.config) # attention layer
        self.ln_2 = nn.LayerNorm(config.n_embd) # layer norm 2
        self.mlp = MLP(config) #multi layer perceptron
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) #attention is a reduce operation
        x = x + self.mlp(self.ln_2(x)) #Map operation
        return x
```
---
#### 👉 Multi Layer Perceptron (MLP)

Used within a transformer block to process output of the self attention layer. It consists of a fully connected layer, an activation function and a projection layer. 

- self.c_fc (Fully Connected Layer): A linear layer that expands the input dimensions. This allows the network to learn more complex representations.
- self.gelu (Gaussian Error Linear Unit): A non-linear activation function that introduces non-linearity into the model, helping it to learn complex patterns. The approximate='tanh' parameter is used for faster computation.
- self.c_proj (Projection Layer): Another linear layer that projects the expanded dimensions back to the original config.n_embd dimensions. This  helps in reducing the dimensionality back to the desired size.

```
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd) # Linear expansion layer 
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
```
**From the docs...**
- [nn.GELU](https://pytorch.org/docs/stable/generated/torch.nn.GELU.html): Applies the Gaussian Error Linear Units function. GELU(x)=x∗Φ(x)

---
#### 👉 Self Attention Operation

The attention operation allows the model to weigh the importance of each token in a sequence and capture dependencies in a sentence. It has a linear attention layer, a projection layer and a buffer. This [article](https://slds-lmu.github.io/seminar_nlp_ss20/attention-and-self-attention-for-nlp.html) explains this concept.

- self.c_attn: A linear layer that projects the input embeddings to three separate sets of embeddings for queries (Q), keys (K), and values (V). This projection increases the dimensionality by 3 times (3 * config.n_embd).
- self.c_proj: A linear layer that projects the concatenated output of all attention heads back to the original embedding dimensionality (config.n_embd).
- self.register_buffer("bias", ...): A lower triangular matrix used as a causal mask to prevent attending to future tokens. 

**Forward method**: Processes the input through layers applying scaled dot-product attention, and returns the transformed output.
- Input Dimensions: The input x has dimensions [B, T, C] where B is the batch size, T is the sequence length, and C is the embedding dimensionality. 
- self.c_attn(x): The input is projected to query, key, and value embeddings.
- q, k, v = qkv.split(self.n_embd, dim=2): Splits the concatenated projections into separate query, key, and value tensors, each of shape [B, T, C].
- Reshape and Transpose: Reshapes and transposes the tensors to [B, nh, T, hs] where nh is the number of heads and hs is the head size (C // nh).
- Scaled Dot-Product Attention: Applies scaled dot-product attention with the causal mask (is_causal=True), ensuring that each token only attends to previous tokens and itself.
- Reshape Back: Transposes and reshapes the output of the attention heads back to [B, T, C].
- Output Projection: The output of the attention layer is projected back to the original embedding dimensionality using self.c_proj.


```
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() 
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y
```
**From the docs...**
- [torch.ones](https://pytorch.org/docs/stable/generated/torch.ones.html): Returns a tensor filled with the scalar value 1, with the shape defined by the variable argument size.
- [torch.split](https://pytorch.org/docs/stable/generated/torch.split.html): Splits the tensor into chunks. Each chunk is a view of the original tensor.
- [torch.transpose](https://pytorch.org/docs/stable/generated/torch.transpose.html): Returns a tensor that is a transposed version of input. The given dimensions dim0 and dim1 are swapped.
- [torch.tril](https://pytorch.org/docs/stable/generated/torch.tril.html): Returns the lower triangular part of the matrix (2-D tensor) or batch of matrices input, the other elements of the result tensor out are set to 0.
- [F.scaled_dot_product_attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html): Computes scaled dot product attention on query, key and value tensors
- [torch.Tensor.contiguous](https://pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html): Returns a contiguous in memory tensor containing the same data as self tensor.

---
### Dataloaders and training 🎲 

Before loading the data, we need to complete the GPT class defined earlier and add the forward method, load pre-trained weights and optimizers to it for training. 

```
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)
```
#### ---------------------------------------------------------

🔥 **Weights initialization method**:  Initializes with random weights.
- nn.Linear: Linear layers are initialized with a normal distribution with a mean of 0 and a standard deviation of 0.02 (scaled for some layers).
- nn.Embedding: Embedding layers are initialized with a normal distribution with a mean of 0 and a standard deviation of 0.02. 

```
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```
**From the docs...**
- [torch.nn.init.normal](https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1init_1a105c2a8ef81c6faa82a01cf35ce9f3b1.html): Fills the given 2-dimensional matrix with values drawn from a normal distribution parameterized by mean and std
- [torch.nn.init.zeroes](https://pytorch.org/cppdocs/api/function_namespacetorch_1_1nn_1_1init_1af7e7736ba2d050adc0523d84285564e8.html): Fills the given tensor with zeros.

#### ---------------------------------------------------------

🔥 **Forward method**: Make a forward pass through the transformer architecture and calculate loss.
- Input: idx is a tensor of shape [B, T] where B is the batch size and T is the sequence length.
- Position Embeddings: pos_emb is computed for positions.
- Token Embeddings: tok_emb is computed for tokens.
- Summation: Token and position embeddings are added.
- Transformer Blocks: The combined embeddings are passed through each transformer block.
- Layer Normalization: The final hidden states are normalized.
- Logits: The final hidden states are projected to vocabulary logits using lm_head.
- Loss Calculation: If targets are provided, cross-entropy loss is calculated.

```
    def forward(self, idx, targets=None):
        
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        
        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        
        x = tok_emb + pos_emb
        
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss

```
**From the docs...**
- [torch.arange](https://pytorch.org/docs/stable/generated/torch.arange.html): Returns a 1-D tensor of size 
⌈(end−start)/step⌉ with values from the interval [start, end) taken with common difference step beginning from start.
- [F.cross_entropy](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html): Compute the cross entropy loss between input logits and target.

#### ---------------------------------------------------------

🔥 **Load from pretrained method**: Class method to load pre-trained weights from Hugging Face's GPT-2 models.
- Model Configuration: Sets the number of layers, heads, and embedding dimensions based on the model type.
- Initialize Model: Creates a new GPT model with the specified configuration.
- Load Pre-trained Weights: Loads pre-trained weights from a Hugging Face model.
- Weight Alignment: Ensures that the weights are correctly aligned and copies them to the new model, handling any necessary transpositions.

```
    @classmethod
        def from_pretrained(cls, model_type):
            """Loads pretrained GPT-2 model weights from huggingface"""

            assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
            
            from transformers import GPT2LMHeadModel
            print("loading weights from pretrained gpt: %s" % model_type)

            # n_layer, n_head and n_embd are determined from model_type
            config_args = {
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
            }[model_type]

            config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
            config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
            
            # create a from-scratch initialized minGPT model
            config = GPTConfig(**config_args)
            model = GPT(config)
            sd = model.state_dict()
            sd_keys = sd.keys()
            sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

            # init a huggingface/transformers model
            model_hf = GPT2LMHeadModel.from_pretrained(model_type)
            sd_hf = model_hf.state_dict()

            # copy while ensuring all of the parameters are aligned and match in names and shapes
            sd_keys_hf = sd_hf.keys()
            sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
            sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
            transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
            
            # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
            # this means that we have to transpose these weights when we import them
            
            assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
            for k in sd_keys_hf:
                if any(k.endswith(w) for w in transposed):
                    # special treatment for the Conv1D weights we need to transpose
                    assert sd_hf[k].shape[::-1] == sd[k].shape
                    with torch.no_grad():
                        sd[k].copy_(sd_hf[k].t())
                else:
                    # vanilla copy over the other parameters
                    assert sd_hf[k].shape == sd[k].shape
                    with torch.no_grad():
                        sd[k].copy_(sd_hf[k])

            return model
```
#### ---------------------------------------------------------


🔥 **Config optimizer method**: Configures the optimizer for training.
- Parameter Selection: Selects parameters that require gradients.
- Weight Decay: Groups parameters for weight decay (e.g., weights of linear layers and embeddings) and no weight decay (e.g., biases and layer norms).
- Optimizer Initialization: Creates an AdamW optimizer, optionally using a fused version if available and running on CUDA.

```
    def configure_optimizers(self, weight_decay, learning_rate, device):
            
            # start with all of the candidate parameters (that require grad)
            
            param_dict = {pn: p for pn, p in self.named_parameters()}
            param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
            
            # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
            # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
            
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            if master_process:
                print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
                print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
            
            # Create AdamW optimizer and use the fused 
            version if it is available
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and 'cuda' in device
            if master_process:
                print(f"using fused AdamW: {use_fused}")
            
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
            
            return optimizer
```
**From the docs...**
- [torch.numel](https://pytorch.org/docs/stable/generated/torch.numel.html): Returns the total number of elements in the input tensor.
- [torch.optim.AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html): Implements AdamW algorithm.
- [torch.requires_grad](https://pytorch.org/docs/stable/generated/torch.Tensor.requires_grad.html): Is True if gradients need to be computed for this Tensor, False otherwise.

