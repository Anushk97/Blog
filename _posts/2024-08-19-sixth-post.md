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

#### Latency vs throughput
- research prioritizes high throughput whereas production prioritizes low latency. Throughput refers to how many queries are processed within a specific period of time. 
- If your system always processes one query at a time, higher latency means lower throughput. If the average latency is 10ms, which means it takes 10ms to process a query, the throughput is 100 queries/second. If the average latency is 100ms, the throughput is 10 queries/second. 
- most modern distributed systems batch queries to process them together, often concurrently, higher latency might also mean higher throughput.
- Batching requires your system to wait for enough queries to arrive in a batch before processing them, which further increases latency. 
- Reducing latency might reduce the number of queries you can process on the same hardware at a time. If your hardware is capable of processing much more than one sample at a time, using it to process only one sample means making processing one sample more expensive. 
- If you optimize your models for better accuracy or lower latency, you can show that your models beat state-of-the-art. But there’s no equivalent state-of-the-art for fairness metrics. You or someone in your life might already be a victim of biased mathematical algorithms without knowing it.   
    - Your loan application might be rejected because the ML algorithm picks on your zip code, which embodies biases about one’s socio-economic background. 
    - Your resume might be ranked lower because the ranking system employers use picks on the spelling of your name. 
    - Your mortgage might get a higher interest rate because it relies partially on credit scores, which reward the rich and punish the poor. 
- 

#### Feature Engineering


Learned and Engineered features

- n-gram is a contiguous sequence of n items from a given sample of text. The items can be phonemes, syllables, letters, or words. For example, given the post “I like food”, its word-level 1-grams are [“I”, “like”, “food”] and its word-level 2-grams are [“I like”, “like food”]. This sentence’s set of n-gram features, if we want n to be 1 and 2, is: [“I”, “like”, “food”, “I like”, “like food”]. 
- Once you’ve generated n-grams for your training data, you can create a vocabulary that matches each n-gram to an index. Then you can convert each post into a vector based on its n-grams’ indices. For example, if we have a vocabulary of 7 n-grams, each post can be a vector of 7 elements. Each element corresponds to the number of times the n-gram at that index appears in the post. “I like food” will be encoded as the vector [1, 1, 0, 1, 1, 0, 1].
- The process of choosing what to use and extracting the information you want to use is feature engineering. For important tasks such as recommending videos for users to watch next on Tiktok, the number of features used can go up to millions. For domainspecific tasks such as predicting whether a transaction is fraudulent, you might need subject matter expertise with banking and frauds to be able to extract useful features. 

##### Feature engineering options
1. Handling missing values
    - deletion
    - imputation
2. Scaling
3. Discretization
4. Encoding categorical features
5. Feature crossing
6. Discrete and continuous positional embeddings
