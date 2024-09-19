---
layout: post
title: Designing ML systems by Chip Huyen notes
date:   2024-08-24 12:54:15 +0800
categories: jekyll update
---

### ML in production 
- One corollary of this is that research prioritizes high throughput whereas production prioritizes low latency
- Batching requires your system to wait for enough queries to arrive in a batch before processing them, which further increases latency. 
- Reducing latency might reduce the number of queries you can process on the same hardware at a time. If your hardware is capable of processing much more than one sample at a time, using it to process only one sample means making processing one sample more expensive. 


### Data engineering fundamentals
- Understanding the sources your data comes from can help you use your data more efficiently. This section aims to give a quick overview of different data sources to those unfamiliar with data in production
- One source is user input data, data explicitly input by users, which is often the input on which ML models can make predictions.
- Logs can record the state of the system and significant events in the system, such as memory usage, number of instances, services called, packages used, etc. It can record the results of different jobs, including large batch jobs for data processing and model training. These types of logs provide visibility into how the system is doing, and the main purpose of this visibility is for debugging and possibly improving the application.
- Data formats range from JSON, row vs column formats, 


### Training data
- Data is full of potential biases. These biases have many possible causes. There are biases caused during collecting, sampling, or labeling. Historical data might be embedded with human biases and ML models, trained on this data, can perpetuate them. Use data but don’t trust it too much! 
- the data that you use to train a model are subsets of real-world data, created by one sampling method or another
- There are two families of sampling: non-probability sampling and random sampling.
- Convenience sampling: samples of data are selected based on their availability. This sampling method is popular because, well, it’s convenient. 
- Snowball sampling: future samples are selected based on existing samples. For example, to scrape legitimate Twitter accounts without having access to Twitter databases, you start with a small number of accounts then you scrape all the accounts in their following, and so on. 
- Judgment sampling: experts decide what samples to include. 
- Quota sampling: you select samples based on quotas for certain slices of data without any randomization. 
- In the simplest form of random sampling, you give all samples in the population equal probabilities of being selected. For example, you randomly select 10% of all samples, giving all samples an equal 10% chance of being selected. 
- To avoid the drawback of simple random sampling listed above, you can first divide your population into the groups that you care about and sample from each group separately. Each group is called a strata, and this method is called stratified sampling. One drawback of this sampling method is that it isn’t always possible, such as when it’s impossible to divide all samples into groups.


### Feature Engineering


### Model Development

