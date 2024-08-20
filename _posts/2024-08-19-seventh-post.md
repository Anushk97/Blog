---
layout: post
title:  "2 Leetcodes a day! (running journal)"
date:   2024-08-19 12:54:15 +0800
categories: jekyll update
---

### 19th Aug

*“Don’t practice until you get it right. Practice until you can’t get it wrong.”*

#### 1. Group Anagrams

- Given an array of strings strs, group all anagrams together into sublists. You may return the output in any order.

- An anagram is a string that contains the exact same characters as another string, but the order of the characters can be different.

    Example 1:
    - Input: strs = ["act","pots","tops","cat","stop","hat"]
    - Output: [["hat"],["act", "cat"],["stop", "pots", "tops"]]

```
# approach: 
- make an empty dictionary and start adding the keys as sorted anagrams. 
- if two words are anagrams, it will be the same when they are sorted. 
- return the values which will be groups of words which are anagrams

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
#approach: 
- make a dictionary count for each value in nums and sort it in descending order. 
- return the top k keys 

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

-------------
### 20th Aug
*"It does not matter how slowly you go as long as you do not stop"*

#### 3. Product of array discluding self
- Given an integer array nums, return an array output where output[i] is the product of all the elements of nums except nums[i].
- Each product is guaranteed to fit in a 32-bit integer.
- Follow-up: Could you solve it in 
O(n)
O(n) time without using the division operation?


    Example 1:
    - Input: nums = [1,2,4,6]
    - Output: [48,24,12,8]

```
# approach:
- keep two pointers l and r. 
- iterate the list with l and r. 
- keep the product variable and times with each element using r

class Solution:
    def productExceptSelf(self, nums):
        l = 0
        res = []
        while l < len(nums):
            prod = 1
            for r in range(len(nums)):
                if l == r:
                    continue
                prod *= nums[r]
            
            res.append(prod)
            l += 1
        return res
```

#### 4. Longest Consecutive Sequence
- Given an array of integers nums, return the length of the longest consecutive sequence of elements.
- A consecutive sequence is a sequence of elements in which each element is exactly 1 greater than the previous element.
- You must write an algorithm that runs in O(n) time.

    Example 1:

    - Input: nums = [2,20,4,10,3,4,5]
    - Output: 4
    (Explanation: The longest consecutive sequence is [2, 3, 4, 5].)

```
#approach: 
- make a set of nums list. iterate over the set. 
- if previous number does not exist in set then the longest will start from there. 
- and while there is i + length element present in the set, the length will increase by 1. 
- then longest will simply be the max of length and longest variable. 

class Solution:
    def longestConsecutive(self, nums):
        numsSet = set(nums)
        longest = 0
        
        for i in numsSet:
            if (i-1) not in numsSet:
                length = 1
                while (i + length) in numsSet:
                    length += 1
                 longest = max(length, longest)
        
        return longest
```