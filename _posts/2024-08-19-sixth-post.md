---
layout: post
title:  "Running notes on ML"
date:   2024-08-19 12:54:15 +0800
categories: jekyll update
---

## Aug 2024

### NLP Cookbook 
- This book summarize and examine the current state-of-the-art (SOTA) NLP models that have been employed for numerous NLP tasks for optimal performance and efficiency


### NLP with transformers
- ULMFiT showed that training long short-term memory (LSTM) networks on a very large and diverse corpus could produce state-of-the-art text classifiers with little labeled data
- Two well known transformers are GPT and BERT. GPT is an encoder only model whereas BERT is decoder only
- BERT uses the encoder part of the Transformer architecture, and a special form of language modeling called masked language modeling.
- The job of the encoder is to encode the information from the input sequence into a numerical representation that is often called the last hidden state. This state is then passed to the decoder, which generates the output sequence.
- The main idea behind attention is that instead of producing a single hidden state for the input sequence, the encoder outputs a hidden state at each step that the decoder can access. However, using all the states at the same time would create a huge input for the decoder, so some mechanism is needed to prioritize which states to use. This is where attention comes in: it lets the decoder assign a different amount of weight, or “attention,” to each of the encoder states at every decoding timestep.

### Designing ML systems
#### Training data
- ML engineers should learn how to handle data well. Data in production is neither finite or stationary. Therefore, creating training data is an iterative process.
- Sampling is an integral part of ML workflow. It allows tou to accomplish a task faster and cheaper by taking samples of much larger dataset which would otherwise require more capital and compute.
- There are two families of sampling: non probability and random sampling. Non probability sampling happens when the selection of data is not based on any proability criteria. 
    - Convenience sampling: samples are selected based on availability
    - Snowball sampling: future samples are selected based on existing samples
    - Judgement sampling: experts decide what samples to use
    - Quota sampling: select samples based on quotas for certain slices of data without any randomization
- The samples selected by non-probability criteria are not representative of the real world data and can be riddled by selection bias.
##### Examples
1. One example of these cases is language modeling. Language models are often trained not with data that is representative of all possible texts but with data that can be easily collected — Wikipedia, CommonCrawl, Reddit.
2. Another example is data for sentiment analysis of general text. Much of this data is collected from sources with natural labels (ratings) — IMDB reviews, Amazon reviews — even for the tasks where you want to predict sentiments of texts that aren’t IMDB or Amazon reviews. These sources don’t include people who don’t have access to the Internet and aren’t willing to put reviews online.
3. The third example is data for training self-driving cars. Initially, data collected for self-driving cars came largely from two areas: Phoenix in Arizona (because of its lax regulations) and the Bay Area in California (because many companies that build self-driving cars are located here). Both areas have generally sunny weather. In 2016, Waymo expanded its operations to Kirkland, WA specially for Kirkland’s rainy weather, but there’s still a lot more self-driving car data for sunny weather than for rainy or snowy weather.

#### Simple Random Sampling
