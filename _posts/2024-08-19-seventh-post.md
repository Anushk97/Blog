---
layout: post
title:  "2 Leetcodes a day! (running journal)"
date:   2024-08-19 12:54:15 +0800
categories: jekyll update
---

### 19th Aug

#### 1. Group Anagrams

- Given an array of strings strs, group all anagrams together into sublists. You may return the output in any order.

- An anagram is a string that contains the exact same characters as another string, but the order of the characters can be different.

    Example 1:
    - Input: strs = ["act","pots","tops","cat","stop","hat"]
    - Output: [["hat"],["act", "cat"],["stop", "pots", "tops"]]

```
# approach: make an empty dictionary and start adding the keys as sorted anagrams. if two words are anagrams, it will be the same when they are sorted. return the values which will be groups of words which are anagrams

class Solution:
    def groupAnagrams(self, strs):
        table = {} #init dict
        for i in strs: 
            sorted_strs = ''.join(sorted(i))
            if sorted_strs not in table:
                table[sorted_strs]  = []
        
            table[sorted_strs].append(i)
        return table.values()
```

#### 2. Top K Elements in List

- Given an integer array nums and an integer k, return the k most frequent elements within the array.
- The test cases are generated such that the answer is always unique.
- You may return the output in any order.

    Example 1:

    - Input: nums = [1,2,2,3,3,3], k = 2
    - Output: [2,3]

```
#approach: make a dictionary count for each value in nums and sort it in descending order. return the top k keys 

class Solution:
    def topKFrequent(self, nums, k):
        nums_counter = Counter(nums)
        res = []
        nums_counter = dict(sorted(nums_counter.items()), key = lambda i:i[1], reverse = True)
        for i, v in nums_counter.items():
            if len(res) > k:
                break
            res.append(i)
        
        return res

```