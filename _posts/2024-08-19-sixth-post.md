---
layout: post
title:  "Running notes on ML"
date:   2024-08-19 12:54:15 +0800
categories: jekyll update
---

## NLP Cookbook 
- This book summarize and examine the current state-of-the-art (SOTA) NLP models that have been employed for numerous NLP tasks for optimal performance and efficiency





## NLP with transformers
- ULMFiT showed that training long short-term memory (LSTM) networks on a very large and diverse corpus could produce state-of-the-art text classifiers with little labeled data
- Two well known transformers are GPT and BERT. GPT is an encoder only model whereas BERT is decoder only
- BERT uses the encoder part of the Transformer architecture, and a special form of language modeling called masked language modeling.
- The job of the encoder is to encode the information from the input sequence into a numerical representation that is often called the last hidden state. This state is then passed to the decoder, which generates the output sequence.
- The main idea behind attention is that instead of producing a single hidden state for the input sequence, the encoder outputs a hidden state at each step that the decoder can access. However, using all the states at the same time would create a huge input for the decoder, so some mechanism is needed to prioritize which states to use. This is where attention comes in: it lets the decoder assign a different amount of weight, or “attention,” to each of the encoder states at every decoding timestep.