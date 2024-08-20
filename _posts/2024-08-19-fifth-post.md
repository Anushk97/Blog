---
layout: post
title:  "Training a Multiclass classification model for sentiment analysis (interview assessment)"
date:   2024-08-19 12:54:15 +0800
categories: jekyll update
---

I recently gave an interview for AI engineer role. The task was to train a Binary classification model given a certain number of features. 


### Dataset and preprocessing
The task contained a train.pkl and test.pkl file. 
train.pkl looked like this:

```markdown
| Timestamp       | Order ID | User ID | Question ID | Batch ID | Feature1 | Feature2  | Feature3 | Feature4 | Feature5 | Is Correct |
|-----------------|----------|---------|-------------|----------|----------|-----------|----------|----------|-----------|------------|
| 1515899961349   | 14       | 100136  | 6442        | 4974     | d        | 0.600515  | 81000    | 5        | 0.051326  | False      |
| 1516008013164   | 17       | 100136  | 920         | 920      | b        | 0.610570  | 22000    | 2        | 1.024035  | False      |
| 1531159801867   | 193      | 100308  | 1103        | 1103     | g        | 0.209624  | 14000    | 2        | -0.043669 | True       |
| 1532796112007   | 236      | 100308  | 3184        | 1995     | h        | 0.781299  | 26333    | 4        | 3.125838  | False      |
| 1530594014682   | 165      | 100308  | 4600        | 3132     | b        | 0.300066  | 9000     | 5        | 0.987286  | True       |
```

To convert the pkl to dataframe object, we just need to read the file and convert it.
```
train_data = pd.read_pickle('./train.pickle')
test_data = pd.read_pickle('./test.pickle')
```
Upon inspecting this dataset, I realized that some values in the Timestamp column contained strings that need to be removed. 

```
# Drop rows with specific user_id prefix
train_data = train_data[~train_data['user_id'].astype(str).str.startswith('drop_this_users_data')]
```
Some values specifically contained the value 'drop_this_user_data' which needed to be removed.

Now looking at the columns, we probably don't need OrderID, userID, Question ID and Batch ID to train our classifier as they are not features with relevant information about the data. They are more like identifiers. 

```
combined_data = pd.concat([train_data, test_data], ignore_index=True)
```
