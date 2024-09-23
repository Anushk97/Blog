---
layout: post
title:  "2 Leetcodes a day! (running journal) 🚩"
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

[visualize](https://memlayout.com?code=class%20Solution%3A%0A%20%20%20%20def%20longestConsecutive(self%2C%20nums)%3A%0A%20%20%20%20%20%20%20%20numsSet%20%3D%20set(nums)%0A%20%20%20%20%20%20%20%20longest%20%3D%200%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20for%20i%20in%20numsSet%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20if%20(i-1)%20not%20in%20numsSet%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20length%20%3D%201%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20while%20(i%20%2B%20length)%20in%20numsSet%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20length%20%2B%3D%201%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20longest%20%3D%20max(length%2C%20longest)%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20return%20longest%0A%0Asol%20%3D%20Solution()%0Aprint(sol.longestConsecutive(%5B2%2C20%2C4%2C10%2C3%2C4%2C5%5D)))

-----
### 21st Aug 
*“Continuous improvement is better than delayed perfection.”*

#### 5. Two integer sum II

- Given an array of integers numbers that is sorted in non-decreasing order.
- Return the indices (1-indexed) of two numbers, [index1, index2], such that they add up to a given target number target and index1 < index2. Note that index1 and index2 cannot be equal, therefore you may not use the same element twice.
- There will always be exactly one valid solution.
- Your solution must use O(1)
- O(1) additional space.

    Example 1:
    - Input: numbers = [1,2,3,4], target = 3
    - Output: [1,2]

```
# approach:
- usa two pointer approach where first iteration is through the array and second iteration (r) will be in for loop
- then check if element l + element r == target, if so append to new list
- return the new list

class Solution:
    def twoSum(self, nums, target):
        l = 0
        res = []
        while l < len(nums):
            for r in range(l+1, len(nums)):
                if nums[l] + nums[r] == target:
                    res.append(l+1)
                    res.append(r+1)
            l += 1
        
        return res
```

#### 6. Max water container

- You are given an integer array heights where heights[i] represents the height of the ith bar.
- You may choose any two bars to form a container. Return the maximum amount of water a container can store.

    Example 1:
    - Input: height = [1,7,2,5,4,7,3,6]
    - Output: 36

```
# approach:
- two points with l at starting and r at end. while l < r
- calculate the max area which is length * breadth
- length will be min of height[l] and height[r]. breadth will be (r - l)
- if l < r then increase l else decrease r

class Solution:
    def maxArea(self, heights):
        l = 0
        r = len(heights) - 1
        res = 0

        while l < r:
            res = max(res, min(heights[l], heights[r]) * (r - l))
            if heights[l] < heights[r]:
                l += 1
            elif heights[r] <= heights[l]:
                r -= 1
        
        return res
```

[visualize](https://memlayout.com?code=class%20Solution%3A%0A%20%20%20%20def%20maxArea(self%2C%20heights)%3A%0A%20%20%20%20%20%20%20%20l%20%3D%200%0A%20%20%20%20%20%20%20%20r%20%3D%20len(heights)%20-%201%0A%20%20%20%20%20%20%20%20res%20%3D%200%0A%0A%20%20%20%20%20%20%20%20while%20l%20%3C%20r%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20res%20%3D%20max(res%2C%20min(heights%5Bl%5D%2C%20heights%5Br%5D)%20*%20(r%20-%20l))%0A%20%20%20%20%20%20%20%20%20%20%20%20if%20heights%5Bl%5D%20%3C%20heights%5Br%5D%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20l%20%2B%3D%201%0A%20%20%20%20%20%20%20%20%20%20%20%20elif%20heights%5Br%5D%20%3C%3D%20heights%5Bl%5D%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20r%20-%3D%201%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20return%20res%0A%20%20%20%20%20%20%20%20%0Asol%20%3D%20Solution()%0Aprint(sol.maxArea(%5B1%2C7%2C2%2C5%2C4%2C7%2C3%2C6%5D)))

------
### 22nd Aug 
*"You’ll never change your life until you change something you do daily."*

#### 7. Is Palindrome
- Given a string s, return true if it is a palindrome, otherwise return false.
- A palindrome is a string that reads the same forward and backward. It is also case-insensitive and ignores all non-alphanumeric characters.

    Example 1:
    - Input: s = "Was it a car or a cat I saw?"
    - Output: true
    - Explanation: After considering only alphanumerical characters we have "wasitacaroracatisaw", which is a palindrome.

```
#Approach: 
- iterate over the string to see if the characters are alphabets 
- check if each character is a number or an alphabet
- if yes, then lower it and add to empty list with lower method
- Finally check if the list and reverse of the list is equal

class Solution:
    def isPalindrome(self, s):
        new_list = []
        for i in s:
            if i.isalnum():
                new_list.append(i.lower())
        
        if new_list == new_list[::-1]:
            ruturn True 
    
        return False
```

#### 8. Buy and sell crypto
- You are given an integer array prices where prices[i] is the price of NeetCoin on the ith day.
- You may choose a single day to buy one NeetCoin and choose a different day in the future to sell it.
- Return the maximum profit you can achieve. You may choose to not make any transactions, in which case the profit would be 0.

    Example 1:
    - Input: prices = [10,1,5,6,7,1]
    - Output: 6 (Explanation: Buy prices[1] and sell prices[4], profit = 7 - 1 = 6.)

```
approach: 
- have a variable x which is float('inf')
- iterate over the list and calulate the minimum of x with x and i 
- if i > x then calculate profit as the max of profit and i 
- 

class Solution:
    def maxProfit(self, prices):
        res = float('inf')
        profit = 0
        for i in range(len(prices)):
            res = min(res, prices[i])
            if prices[i] > res:
                profit = max(profit, prices[i]-res)
        
        return profit
```
----
### 23rd Aug 

#### 9. Longest substring without duplicates
- Given a string s, find the length of the longest substring without duplicate characters.
- A substring is a contiguous sequence of characters within a string.

    Example 1:
    - Input: s = "zxyzxyz"
    - Output: 3 (Explanation: The string "xyz" is the longest without duplicate characters.)

```
approach:
- use a set and a l pointer
- iterate over string and check if char is in set. if it is then remove lth character 
- add ith char to set
- calculate res as max of res and window which will be i - l + 1



class Solution:
    def lengthofLongestSubstring(self, s):
        s_set = set()
        l = 0
        res = 0
        for i in range(len(s)):
            while s[i] in s_set:
                s_set.remove(s[l])
                l += 1
            
            s_set.add(s[i])
            res = max(res, i - l + 1)

        return res

```

#### 10. Longest repeating substring with replacement
- You are given a string s consisting of only uppercase english characters and an integer k. You can choose up to k characters of the string and replace them with any other uppercase English character.
- After performing at most k replacements, return the length of the longest substring which contains only one distinct character.

    Example 1:
    - Input: s = "XYYX", k = 2
    - Output: 4 (Explanation: Either replace the 'X's with 'Y's, or replace the 'Y's with 'X's.)

```
approach: 
- use a dictionary, l pointer and max pointer 
- iterate over s to count the number of each character
- calculate max as max of max pointer and char s
- if the window size - max pointer > k then decrease the count of lth char in dictionary
- return window size

class Solution:
    def characterReplacement(self, s, k):
        count = {}
        l = 0
        maxD = 0
        for i in range(len(s)):
            count[s[i]] = 1 + count.get(s[i], 0)
            maxD = max(maxD, count[s[i]])
        
        if (i - l + 1) - maxD > k:
            count[s[l]] -= 1
            l += 1
        
        return (i - l + 1)
```

[visualize](https://memlayout.com?code=class%20Solution%3A%0A%20%20%20%20def%20characterReplacement(self%2C%20s%2C%20k)%3A%0A%20%20%20%20%20%20%20%20count%20%3D%20%7B%7D%0A%20%20%20%20%20%20%20%20l%20%3D%200%0A%20%20%20%20%20%20%20%20maxD%20%3D%200%0A%20%20%20%20%20%20%20%20for%20i%20in%20range(len(s))%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20count%5Bs%5Bi%5D%5D%20%3D%201%20%2B%20count.get(s%5Bi%5D%2C%200)%0A%20%20%20%20%20%20%20%20%20%20%20%20maxD%20%3D%20max(maxD%2C%20count%5Bs%5Bi%5D%5D)%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20if%20(i%20-%20l%20%2B%201)%20-%20maxD%20%3E%20k%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20count%5Bs%5Bl%5D%5D%20-%3D%201%0A%20%20%20%20%20%20%20%20%20%20%20%20l%20%2B%3D%201%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20return%20(i%20-%20l%20%2B%201)%0A%0Asol%20%3D%20Solution()%0Aprint(sol.characterReplacement(s%20%3D%20'XYYX'%2C%20k%20%3D%202)))

-----
### 24th Aug

#### 11. Permutation String
- You are given two strings s1 and s2.
- Return true if s2 contains a permutation of s1, or false otherwise. That means if a permutation of s1 exists as a substring of s2, then return true.
- Both strings only contain lowercase letters.

    Example 1:
    - Input: s1 = "abc", s2 = "lecabee"
    - Output: true (Explanation: The substring "cab" is a permutation of "abc" and is present in "lecabee".)

```
# approach
- create a hashmap with the count of every character in the string s1
- slide a window over string s2 and decrease the counter for chars occured in the window
- if all counters in hashmap get to zero, means we encountered the permutation

class Solution:
    def checkInclusion(self, s1, s2):
        cntr, w = Counter(s1), len(s1)

        for i in range(len(s2)):
            if s2[i] in cntr:
                cntr[s2[i]] -= 1
            if i >= w and s2[i-w] in cntr:
                cntr[s2[i-w]] += 1
            
            if all([cntr[i] == 0 for i in cntr]):
                return True
        
        return False
```

[visualize](https://memlayout.com?code=from%20collections%20import%20Counter%0A%0Aclass%20Solution%3A%0A%20%20%20%20def%20checkInclusion(self%2C%20s1%2C%20s2)%3A%0A%20%20%20%20%20%20%20%20cntr%2C%20w%20%3D%20Counter(s1)%2C%20len(s1)%0A%0A%20%20%20%20%20%20%20%20for%20i%20in%20range(len(s2))%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20if%20s2%5Bi%5D%20in%20cntr%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20cntr%5Bs2%5Bi%5D%5D%20-%3D%201%0A%20%20%20%20%20%20%20%20%20%20%20%20if%20i%20%3E%3D%20w%20and%20s2%5Bi-w%5D%20in%20cntr%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20cntr%5Bs2%5Bi-w%5D%5D%20%2B%3D%201%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20if%20all(%5Bcntr%5Bi%5D%20%3D%3D%200%20for%20i%20in%20cntr%5D)%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20return%20True%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20return%20False%0A%20%20%20%20%20%20%20%20%0Asol%20%3D%20Solution()%0Aprint(sol.checkInclusion(s1%20%3D%20'abc'%2C%20s2%20%3D%20'lecabee')))


#### 12. Minimum Stack
- Design a stack class that supports the push, pop, top, and getMin operations.
- MinStack() initializes the stack object.
- void push(int val) pushes the element val onto the stack.
- void pop() removes the element on the top of the stack.
- int top() gets the top element of the stack.
- int getMin() retrieves the minimum element in the stack.
- Each function should run in O (1) O(1) time.

    Example 1:
    - Input: ["MinStack", "push", 1, "push", 2, "push", 0, "getMin", "pop", "top", "getMin"]
    - Output: [null,null,null,null,0,null,2,1]

        Explanation:
        - MinStack minStack = new MinStack();
        - minStack.push(1);
        - minStack.push(2);
        - minStack.push(0);
        - minStack.getMin(); // return 0
        - minStack.pop();
        - minStack.top();    // return 2
        - minStack.getMin(); // return 1

```
#approach: 
- inititate two lists, one for stack and another for minstack
- for push, append to stack first, then calculate the min, then append to minstack
- for pop, just pop from both
- for top, just return last element from stack
- for getMin, just return last element from minStack

class MinStack:
    def __init__(self):
        self.stack = []
        self.minStack = []

    def push(self, val):
        self.stack.append(val)
        val = min(val, self.minStack[-1] if self.minStack else val)
        self.minStack.append(val)
    
    def pop(self):
        self.stack.pop()
        self.minStack.pop()
    
    def top(self):
        return self.stack[-1]
    
    def getMin(self):
        return self.minStack[-1]

```
------
### 25th Aug

#### 13. Evaluate reverse polish notation

- You are given an array of strings tokens that represents a valid arithmetic expression in Reverse Polish Notation.
- Return the integer that represents the evaluation of the expression.
-The operands may be integers or the results of other operations.
- The operators include '+', '-', '*', and '/'.
- Assume that division between integers always truncates toward zero.

    Example 1:
    - Input: tokens = ["1","2","+","3","*","4","-"]
    - Output: 5 (Explanation: ((1 + 2) * 3) - 4 = 5)

```
#approach: 
- have an empty stack
- make if conditions for all the operational character and append to stack accordingly
- pop the last two element from the stack first and then append a new element

class Solution:
    def evalRPN(self, tokens):
        stack = []
        for i in tokens:
            if i == '+':
                stack.append(stack.pop() + stack.pop())
            elif i == '-':
                a, b = stack.pop(), stack.pop()
                stack.append(b-a)
            elif i == '*':
                stack.append(stack.pop() * stack.pop())
            elif i == '/':
                c, d = stack.pop(), stack.pop()
                stack.append(int(float(d)/c))

            else:
                stack.append(int(i))
        
        return stack[0]
```

#### 14. Generate parenthesis
- You are given an integer n. Return all well-formed parentheses strings that you can generate with n pairs of parentheses.

    Example 1:
    - Input: n = 1
    - Output: ["()"]

    Example 2:
    - Input: n = 3
    - Output: ["((()))","(()())","(())()","()(())","()()()"]


```
#approach:
- use backtracking with two pointers
- 

class Solution:
    def generateParenthesis(self, n):
        stack = []
        res = []
        def backtrack(openN, closeN):
            if openN == closeN == n: # base case
                res.append(''.join(stack))
            
            if openN < n:
                stack.append("(")
                backtrack(openN+1, closeN)
                stack.pop()
            
            if closeN < openN:
                stack.append(")")
                backtrack(openN, closeN+1)
                stack.pop()
        
        backtrack(0,0)
        return res
```

[visualize](https://memlayout.com?code=class%20Solution%3A%0A%20%20%20%20def%20generateParenthesis(self%2C%20n)%3A%0A%20%20%20%20%20%20%20%20stack%20%3D%20%5B%5D%0A%20%20%20%20%20%20%20%20res%20%3D%20%5B%5D%0A%20%20%20%20%20%20%20%20def%20backtrack(openN%2C%20closeN)%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20if%20openN%20%3D%3D%20closeN%20%3D%3D%20n%3A%20%23%20base%20case%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20res.append(''.join(stack))%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20if%20openN%20%3C%20n%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20stack.append(%22(%22)%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20backtrack(openN%2B1%2C%20closeN)%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20stack.pop()%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20if%20closeN%20%3C%20openN%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20stack.append(%22)%22)%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20backtrack(openN%2C%20closeN%2B1)%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20stack.pop()%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20backtrack(0%2C0)%0A%20%20%20%20%20%20%20%20return%20res%0A%20%20%20%20%20%20%20%20%0Asol%20%3D%20Solution()%0Aprint(sol.generateParenthesis(2)))

----
### 26th Aug

#### 15. Daily temperatures 
- You are given an array of integers temperatures where temperatures[i] represents the daily temperatures on the ith day.
- Return an array result where result[i] is the number of days after the ith day before a warmer temperature appears on a future day. If there is no day in the future where a warmer temperature will appear for the ith day, set result[i] to 0 instead.

    Example 1:
    - Input: temperatures = [30,38,30,36,35,40,28]
    - Output: [1,4,1,2,1,0,0]

```
#approach:
- create an array of zeroes which will be equal to the len of temp
- create an empty stack
- iterate with index and value over temp list
- check if current value of element in list is greater than the last value in stack
- if so, then pop value and index from stack
- change the element in array with zeroes for that particular index
- append value and index in stack


class Solution:
    def dailyTemperatures(self, temp):
        res = [0] * len(temp)
        stack = []

        for i, v in enumerate(temp):
            while stack and v > stack[-1][0]:
                stackT, stackInd = stack.pop()
                res[stackInd] = i - stackInd
            stack.append((v,i))
        
        return res
```

[visualize](https://memlayout.com?code=class%20Solution%3A%0A%20%20%20%20def%20dailyTemperatures(self%2C%20temp)%3A%0A%20%20%20%20%20%20%20%20res%20%3D%20%5B0%5D%20*%20len(temp)%0A%20%20%20%20%20%20%20%20stack%20%3D%20%5B%5D%0A%0A%20%20%20%20%20%20%20%20for%20i%2C%20v%20in%20enumerate(temp)%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20while%20stack%20and%20v%20%3E%20stack%5B-1%5D%5B0%5D%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20stackT%2C%20stackInd%20%3D%20stack.pop()%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20res%5BstackInd%5D%20%3D%20i%20-%20stackInd%0A%20%20%20%20%20%20%20%20%20%20%20%20stack.append((v%2Ci))%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20return%20res%0A%20%20%20%20%20%20%20%20%0Asol%20%3D%20Solution()%0Aprint(sol.dailyTemperatures(%5B30%2C38%2C30%2C36%2C35%2C40%2C28%5D)))

#### 16. Binary search
- You are given an array of distinct integers nums, sorted in ascending order, and an integer target.
- Implement a function to search for target within nums. If it exists, then return its index, otherwise, return -1.
- Your solution must run in O(logn) and O(logn) time.

    Example 1:
    - Input: nums = [-1,0,2,4,6,8], target = 4
    - Output: 3

```
#approach:
- initiate a left and right pointer at 0 and end of list respectively
- check while l < r
- calculate middle element 
- check if middle < target, which means we can bring out l pointer forward upto middle
- check if middle > target, which means we can bring down r pointer upto middle
- else we return middle which means the target is found 

class Solution:
    def search(self, nums, target):
        l = 0
        r = len(nums) - 1
        while l < r:
            m = (l + r) // 2
            if nums[m] < target:
                l = m + 1
            elif nums[m] > target:
                r = m - 1
            else:
                return m
        
        return -1


```

----
### 27th Aug

#### 17. Balanced binary tree

[problem](https://leetcode.com/problems/balanced-binary-tree/description/)

```
# approach:
- do it recursively with DFS
- return a boolean value and the height
- calculate the balance as the diff between left and right subtree should be less than 1
- balance is only going to be true if left and right subtrees are balanced

class Solution:
    def isBalanced(self, root):
        def dfs(root):
            if not root:
                return [True, 0]
            
            left = dfs(root.left)
            right = dfs(root.right)
            balance = (left[0] and right[0] and 
                    abs(left[1]-right[1]) <= 1)

            return [balanced, 1 + max(left[1], right[1])]
        
        return dfs(root)[0]
```

[visualize](https://memlayout.com?code=class%20TreeNode%3A%0A%20%20%20%20def%20__init__(self%2C%20val%3D0%2C%20left%3DNone%2C%20right%3DNone)%3A%0A%20%20%20%20%20%20%20%20self.val%20%3D%20val%0A%20%20%20%20%20%20%20%20self.left%20%3D%20left%0A%20%20%20%20%20%20%20%20self.right%20%3D%20right%0A%0Aclass%20Solution%3A%0A%20%20%20%20def%20isBalanced(self%2C%20root)%3A%0A%20%20%20%20%20%20%20%20def%20dfs(root)%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20if%20not%20root%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20return%20%5BTrue%2C%200%5D%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20left%20%3D%20dfs(root.left)%0A%20%20%20%20%20%20%20%20%20%20%20%20right%20%3D%20dfs(root.right)%0A%20%20%20%20%20%20%20%20%20%20%20%20balanced%20%3D%20(left%5B0%5D%20and%20right%5B0%5D%20and%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20abs(left%5B1%5D%20-%20right%5B1%5D)%20%3C%3D%201)%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20return%20%5Bbalanced%2C%201%20%2B%20max(left%5B1%5D%2C%20right%5B1%5D)%5D%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20return%20dfs(root)%5B0%5D%0A%0A%23%20Create%20a%20binary%20tree%0A%23%20%20%20%20%20%20%201%0A%23%20%20%20%20%20%20%2F%20%5C%0A%23%20%20%20%20%202%20%20%203%0A%23%20%20%20%20%2F%20%5C%0A%23%20%20%204%20%20%205%0A%23%20%20%2F%0A%23%206%0A%0Aroot%20%3D%20TreeNode(1)%0Aroot.left%20%3D%20TreeNode(2)%0Aroot.right%20%3D%20TreeNode(3)%0Aroot.left.left%20%3D%20TreeNode(4)%0Aroot.left.right%20%3D%20TreeNode(5)%0Aroot.left.left.left%20%3D%20TreeNode(6)%0A%0A%23%20Create%20an%20instance%20of%20the%20Solution%20class%0Asolution%20%3D%20Solution()%0A%0A%23%20Check%20if%20the%20tree%20is%20balanced%0Ais_balanced%20%3D%20solution.isBalanced(root)%0A%0Aprint(f%22Is%20the%20binary%20tree%20balanced%3F%20%7B'Yes'%20if%20is_balanced%20else%20'No'%7D%22)%0A%0A%23%20Expected%20output%3A%20Is%20the%20binary%20tree%20balanced%3F%20Yes)


#### 18. Lowest common ancestor of binary search tree

[problem](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/solutions/)

```
- start with the root as it is always the common ancestor
- left subtree is always going to be less than parent and right subtree is going to be greater than parent
- time complexity will be height of the tree which is usually log(n)

class Solution:
    def lowestCommonAncestor(self, root):
        cur = root

        while cur:
            if p.val > cur.val and q.val > cur.val:
                cur = cur.right
            elif p.val < cur.val and q.val < cur.val:
                cur = cur.left
            else:
                return cur
```

[visualize](https://memlayout.com?code=class%20TreeNode%3A%0A%20%20%20%20def%20__init__(self%2C%20val)%3A%0A%20%20%20%20%20%20%20%20self.val%20%3D%20val%0A%20%20%20%20%20%20%20%20self.left%20%3D%20None%0A%20%20%20%20%20%20%20%20self.right%20%3D%20None%0A%0Aclass%20Solution%3A%0A%20%20%20%20def%20lowestCommonAncestor(self%2C%20root%2C%20p%2C%20q)%3A%0A%20%20%20%20%20%20%20%20cur%20%3D%20root%0A%0A%20%20%20%20%20%20%20%20while%20cur%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20if%20p.val%20%3E%20cur.val%20and%20q.val%20%3E%20cur.val%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20cur%20%3D%20cur.right%0A%20%20%20%20%20%20%20%20%20%20%20%20elif%20p.val%20%3C%20cur.val%20and%20q.val%20%3C%20cur.val%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20cur%20%3D%20cur.left%0A%20%20%20%20%20%20%20%20%20%20%20%20else%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20return%20cur%0A%0A%23%20Create%20a%20binary%20search%20tree%0A%23%20%20%20%20%20%20%20%206%0A%23%20%20%20%20%20%20%2F%20%20%20%5C%0A%23%20%20%20%20%202%20%20%20%20%208%0A%23%20%20%20%20%2F%20%5C%20%20%20%2F%20%5C%0A%23%20%20%200%20%20%204%207%20%20%209%0A%23%20%20%20%20%20%20%2F%20%5C%0A%23%20%20%20%20%203%20%20%205%0A%0Aroot%20%3D%20TreeNode(6)%0Aroot.left%20%3D%20TreeNode(2)%0Aroot.right%20%3D%20TreeNode(8)%0Aroot.left.left%20%3D%20TreeNode(0)%0Aroot.left.right%20%3D%20TreeNode(4)%0Aroot.right.left%20%3D%20TreeNode(7)%0Aroot.right.right%20%3D%20TreeNode(9)%0Aroot.left.right.left%20%3D%20TreeNode(3)%0Aroot.left.right.right%20%3D%20TreeNode(5)%0A%0A%23%20Create%20an%20instance%20of%20the%20Solution%20class%0Asolution%20%3D%20Solution()%0A%0A%23%20Define%20two%20nodes%20to%20find%20their%20lowest%20common%20ancestor%0Ap%20%3D%20root.left%20%20%23%20Node%20with%20value%202%0Aq%20%3D%20root.left.right.right%20%20%23%20Node%20with%20value%205%0A%0A%23%20Find%20the%20lowest%20common%20ancestor%0Alca%20%3D%20solution.lowestCommonAncestor(root%2C%20p%2C%20q)%0A%0Aprint(f%22The%20lowest%20common%20ancestor%20of%20%7Bp.val%7D%20and%20%7Bq.val%7D%20is%3A%20%7Blca.val%7D%22)%0A%0A%23%20Expected%20output%3A%20The%20lowest%20common%20ancestor%20of%202%20and%205%20is%3A%202)

-----
### 28th Aug

#### 19. Implement queue using stacks

queue is a FIFO data structure. main operations in queue are push and pop. peek is to return the element in the front. 
push will be O(1) and pop should be O(n), how to get pop in constant time?
peek is technically getting the last element


[problem](https://leetcode.com/problems/implement-queue-using-stacks/solutions/)

```
class Solution:
    def __init__(self):
        self.s1 = []
        self.s2 = []

    def push(self, x):
        self.s1.append(x)

    def pop(self):
        if not self.s2:
            while self.s1:
                self.s2.append(self.s1.pop())

        return self.s2.pop()

    def peek(self):
        if not self.s2:
                while self.s1:
                    self.s2.append(self.s1.pop())
        
        return self.s2[-1]
    
    def empty(self):
        return max(len(self.s1), len(self.s2)) == 0
```

#### 20. Climbing stairs

[problem](https://leetcode.com/problems/climbing-stairs/solutions/)

```
- two variables shifting n - 1 times. already accounting for the first step during initialization

class Solution:
    def climbStairs(self, n):
        one, two = 1,1

        for i in range(n-1):
            tmp = one
            one = one + two # number of ways to reach the current step
            two = tmp
        
        return one # this contains the number of ways to climb n steps
```

[visualize](https://memlayout.com?code=class%20Solution%3A%0A%20%20%20%20def%20climbStairs(self%2C%20n)%3A%0A%20%20%20%20%20%20%20%20one%2C%20two%20%3D%201%2C1%0A%0A%20%20%20%20%20%20%20%20for%20i%20in%20range(n-1)%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20tmp%20%3D%20one%0A%20%20%20%20%20%20%20%20%20%20%20%20one%20%3D%20one%20%2B%20two%20%23%20number%20of%20ways%20to%20reach%20the%20current%20step%0A%20%20%20%20%20%20%20%20%20%20%20%20two%20%3D%20tmp%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20return%20one%20%23%20this%20contains%20the%20number%20of%20ways%20to%20climb%20n%20steps%0A%20%20%20%20%20%20%20%20%0A%0Asol%20%3D%20Solution()%0Aprint(sol.climbStairs(8)))

------
### 29th Aug

#### 21. [Longest Palindrome](https://leetcode.com/problems/longest-palindrome/solutions/)

Palindome is a string which is the same when reversed. odd and even length string matters 
For odd length string, we need a pair of matching pairs and a single character which is not a pair
for even length, we need all pairs 
need to have a hasmap to count the number of chars in string.

```
class Solution:
    def longestPalindrom(self, s):
        count = defaultdict(int)
        res = 0

        for c in s:
            count[c] += 1
            if count[c] % 2 == 0: #if there is a pair then you increment the result by 2
                res += 2
        
        # if the count is odd, then increment the result by 1
        for cnt in count.values():
            if cnt % 2 == 1:
                res += 1
                break

        return res
```

[visualize](https://memlayout.com?code=from%20collections%20import%20defaultdict%0A%0Aclass%20Solution%3A%0A%20%20%20%20def%20longestPalindrom(self%2C%20s)%3A%0A%20%20%20%20%20%20%20%20count%20%3D%20defaultdict(int)%0A%20%20%20%20%20%20%20%20res%20%3D%200%0A%0A%20%20%20%20%20%20%20%20for%20c%20in%20s%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20count%5Bc%5D%20%2B%3D%201%0A%20%20%20%20%20%20%20%20%20%20%20%20if%20count%5Bc%5D%20%25%202%20%3D%3D%200%3A%20%23if%20there%20is%20a%20pair%20then%20you%20increment%20the%20result%20by%202%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20res%20%2B%3D%202%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%23%20if%20the%20count%20is%20odd%2C%20then%20increment%20the%20result%20by%201%0A%20%20%20%20%20%20%20%20for%20cnt%20in%20count.values()%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20if%20cnt%20%25%202%20%3D%3D%201%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20res%20%2B%3D%201%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20break%0A%0A%20%20%20%20%20%20%20%20return%20res%0A%0Asol%20%3D%20Solution()%0Aprint(sol.longestPalindrom(%22abccccdd%22)))


#### 22. [Reverse Linked list](https://leetcode.com/problems/reverse-linked-list/description/)

Have a prev pointer which will be none and then iterate over LL starting from head. point each next pointer to prev node

```
class Solution:
    def reverseList(self, head):
        prev = None
        while head:
            tmp = head.next
            head.next = prev
            prev = head
            head = tmp
        
        return prev
```

[visualize](https://memlayout.com?code=class%20ListNode%3A%0A%20%20%20%20def%20__init__(self%2C%20val%3D0%2C%20next%3DNone)%3A%0A%20%20%20%20%20%20%20%20self.val%20%3D%20val%0A%20%20%20%20%20%20%20%20self.next%20%3D%20next%0A%0Aclass%20Solution%3A%0A%20%20%20%20def%20reverseList(self%2C%20head)%3A%0A%20%20%20%20%20%20%20%20prev%20%3D%20None%0A%20%20%20%20%20%20%20%20while%20head%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20tmp%20%3D%20head.next%0A%20%20%20%20%20%20%20%20%20%20%20%20head.next%20%3D%20prev%0A%20%20%20%20%20%20%20%20%20%20%20%20prev%20%3D%20head%0A%20%20%20%20%20%20%20%20%20%20%20%20head%20%3D%20tmp%0A%20%20%20%20%20%20%20%20return%20prev%0A%0Adef%20test_reverse_list()%3A%0A%20%20%20%20head%20%3D%20ListNode(1)%0A%20%20%20%20head.next%20%3D%20ListNode(2)%0A%20%20%20%20head.next.next%20%3D%20ListNode(3)%0A%20%20%20%20head.next.next.next%20%3D%20ListNode(4)%0A%20%20%20%20head.next.next.next.next%20%3D%20ListNode(5)%0A%0A%20%20%20%20solution%20%3D%20Solution()%0A%20%20%20%20reversed_head%20%3D%20solution.reverseList(head)%0A%20%20%20%20%0A%20%20%20%20return%20reversed_head%0A%0Aprint(test_reverse_list()))

----
### 30th Aug

#### 23. [max depth of binary tree](https://leetcode.com/problems/maximum-depth-of-binary-tree/solutions/)

Recursively iterate over the tree 

```
class Solution:
    def maxDepth(self, root):
        if not root:
            return 0
        return 1 + (self.maxDepth(root.left), self.maxDepth(root.right))
```

#### 24. [middle of linked list](https://leetcode.com/problems/middle-of-the-linked-list/description/)

Have two pointers slow and fast. iterate over fast until the end of the list which slow incrementing as well. when fast will reach the end
slow will reach the middle. 

```
class Solution:
    def middleNode(self, head):
        slow = head
        fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        
        return slow
```
----
### 31st Aug

#### 25. [maximum subarray](https://leetcode.com/problems/maximum-subarray/solutions/)

can compute every single subarray
maintain a current sum variable which will add the element in the array
another variable which will maintain the max subarray. it will be initialized to the first array element

```
class Solution:
    def maxSubArray(self, nums):
        maxSub = nums[0]
        curSum = 0
        for i in nums:
            if curSum < 0:
                curSum = 0
            curSum += i
            maxSub = max(maxSub, curSum)
        
        return maxSub
```
[visualize](https://memlayout.com?code=class%20Solution%3A%0A%20%20%20%20def%20maxSubArray(self%2C%20nums)%3A%0A%20%20%20%20%20%20%20%20maxSub%20%3D%20nums%5B0%5D%0A%20%20%20%20%20%20%20%20curSum%20%3D%200%0A%20%20%20%20%20%20%20%20for%20i%20in%20nums%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20if%20curSum%20%3C%200%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20curSum%20%3D%200%0A%20%20%20%20%20%20%20%20%20%20%20%20curSum%20%2B%3D%20i%0A%20%20%20%20%20%20%20%20%20%20%20%20maxSub%20%3D%20max(maxSub%2C%20curSum)%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20return%20maxSub%0A%20%20%20%20%20%20%20%20%0Asol%20%3D%20Solution()%0Aprint(sol.maxSubArray(%5B-2%2C1%2C-3%2C4%2C-1%2C2%2C1%2C-5%2C4%5D)))


#### 26. [insert intervals](https://leetcode.com/problems/insert-interval/solutions/)

iterate over the lists of list to check whether there is an overlap between previous and current subarray
if there is then find the max 
also need to sort the interval list by start time first and set the prev to the start time of first interval
in the loop, append prev to new list (merged)

```
class Solution:
    def insert(self, intervals, newIntervals):
        intervals.append(newIntervals)
        merged = []

        intervals.sort(key=lambda x:x[0])
        prev = intervals[0]
        for i in intervals[1:]:
            if prev[1] >= i[0]: #there is an overlap if this is true
                prev[1] = max(prev[1], i[1])
            else:
                merged.append(prev)
                prev = i
            
        merged.append(prev)
    
        return merged
```
[visualize](https://memlayout.com?code=class%20Solution%3A%0A%20%20%20%20def%20insert(self%2C%20intervals%2C%20newInterval)%3A%0A%20%20%20%20%20%20%20%20intervals.append(newInterval)%0A%20%20%20%20%20%20%20%20merged%20%3D%20%5B%5D%0A%0A%20%20%20%20%20%20%20%20intervals.sort(key%3Dlambda%20x%3Ax%5B0%5D)%0A%20%20%20%20%20%20%20%20prev%20%3D%20intervals%5B0%5D%0A%20%20%20%20%20%20%20%20for%20i%20in%20intervals%5B1%3A%5D%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20if%20prev%5B1%5D%20%3E%3D%20i%5B0%5D%3A%20%23there%20is%20an%20overlap%20if%20this%20is%20true%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20prev%5B1%5D%20%3D%20max(prev%5B1%5D%2C%20i%5B1%5D)%0A%20%20%20%20%20%20%20%20%20%20%20%20else%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20merged.append(prev)%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20prev%20%3D%20i%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20merged.append(prev)%0A%20%20%20%20%0A%20%20%20%20%20%20%20%20return%20merged%0A%0Asol%20%3D%20Solution()%0Aprint(sol.insert(intervals%20%3D%20%5B%5B1%2C3%5D%2C%5B6%2C9%5D%5D%2C%20newInterval%20%3D%20%5B2%2C5%5D)))


------
### 1st Sept

#### 27. [binary tree level order traversal BFS](https://leetcode.com/problems/binary-tree-level-order-traversal/description/)

start with a queue with root in it. iterate over the queue to pop the first element and append to level list
then append left and right nodes of the tree to the queue


```
class Solution:
    def levelOrder(self, root):
        q = [root]
        res = []
        while q:
            level = []
            for i in range(len(q)):
                node = q.pop(0)
                if node:
                    level.append(node.val)
                    q.append(node.left)
                    q.append(node.right)
            
            if level:
                res.append(level)

        return res
```
[visualize](https://memlayout.com?code=%23%20Definition%20for%20a%20binary%20tree%20node.%0Aclass%20TreeNode%3A%0A%20%20%20%20def%20__init__(self%2C%20val%3D0%2C%20left%3DNone%2C%20right%3DNone)%3A%0A%20%20%20%20%20%20%20%20self.val%20%3D%20val%0A%20%20%20%20%20%20%20%20self.left%20%3D%20left%0A%20%20%20%20%20%20%20%20self.right%20%3D%20right%0A%0Aclass%20Solution%3A%0A%20%20%20%20def%20levelOrder(self%2C%20root)%3A%0A%20%20%20%20%20%20%20%20q%20%3D%20%5Broot%5D%0A%20%20%20%20%20%20%20%20res%20%3D%20%5B%5D%0A%20%20%20%20%20%20%20%20while%20q%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20level%20%3D%20%5B%5D%0A%20%20%20%20%20%20%20%20%20%20%20%20for%20i%20in%20range(len(q))%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20node%20%3D%20q.pop(0)%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20if%20node%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20level.append(node.val)%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20q.append(node.left)%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20q.append(node.right)%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20if%20level%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20res.append(level)%0A%0A%20%20%20%20%20%20%20%20return%20res%0A%0Adef%20test_level_order()%3A%0A%20%20%20%20%23%20Create%20a%20binary%20tree%0A%20%20%20%20%23%20%20%20%20%20%203%0A%20%20%20%20%23%20%20%20%20%2F%20%20%20%5C%0A%20%20%20%20%23%20%20%209%20%20%20%2020%0A%20%20%20%20%23%20%20%20%20%20%20%20%2F%20%20%5C%0A%20%20%20%20%23%20%20%20%20%20%2015%20%20%207%0A%20%20%20%20root%20%3D%20TreeNode(3)%0A%20%20%20%20root.left%20%3D%20TreeNode(9)%0A%20%20%20%20root.right%20%3D%20TreeNode(20)%0A%20%20%20%20root.right.left%20%3D%20TreeNode(15)%0A%20%20%20%20root.right.right%20%3D%20TreeNode(7)%0A%0A%20%20%20%20%23%20Create%20an%20instance%20of%20Solution%0A%20%20%20%20solution%20%3D%20Solution()%0A%0A%20%20%20%20%23%20Run%20the%20level%20order%20traversal%0A%20%20%20%20result%20%3D%20solution.levelOrder(root)%0A%0A%20%20%20%20%23%20Expected%20output%0A%20%20%20%20expected%20%3D%20%5B%5B3%5D%2C%20%5B9%2C%2020%5D%2C%20%5B15%2C%207%5D%5D%0A%0A%20%20%20%20%23%20Check%20if%20the%20result%20matches%20the%20expected%20output%0A%20%20%20%20assert%20result%20%3D%3D%20expected%2C%20f%22Expected%20%7Bexpected%7D%2C%20but%20got%20%7Bresult%7D%22%0A%0A%20%20%20%20print(%22Test%20case%20passed%20successfully!%22)%0A%0A%20%20%20%20%23%20Test%20with%20an%20empty%20tree%0A%20%20%20%20assert%20solution.levelOrder(None)%20%3D%3D%20%5B%5D%2C%20%22Empty%20tree%20should%20return%20an%20empty%20list%22%0A%0A%20%20%20%20print(%22Empty%20tree%20test%20case%20passed%20successfully!%22)%0A%0A%23%20Run%20the%20test%0Atest_level_order())


#### 28. [combination sum](https://leetcode.com/problems/combination-sum/description/)

each candidate, we're exploring two possibilities: either we include it in our combination (potentially multiple times) or we don't. 
This creates a decision tree that the DFS traverses, building up combinations and backtracking when necessary.

```
class Solution:
    def combinationSum(self, candidates, target):
        res = []
        def dfs(i, lst, total):
            if total == target:
                res.append(lst.copy())
                return
            if i >= len(candidates) or total > target:
                return 
            
            # Include the current candidate: We append it to lst, update the total, and recurse
            lst.append(candidates[i])
            dfs(i, lst, total+candidates[i])

            # Exclude the current candidate: We move to the next index without changing lst or total
            lst.pop()
            dfs(i+1, lst, total)
        
        dfs(0, [], 0)
        return res
```

[visualize](https://memlayout.com?code=class%20Solution%3A%0A%20%20%20%20def%20combinationSum(self%2C%20candidates%2C%20target)%3A%0A%20%20%20%20%20%20%20%20res%20%3D%20%5B%5D%0A%20%20%20%20%20%20%20%20def%20dfs(i%2C%20lst%2C%20total)%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20if%20total%20%3D%3D%20target%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20res.append(lst.copy())%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20return%0A%20%20%20%20%20%20%20%20%20%20%20%20if%20i%20%3E%3D%20len(candidates)%20or%20total%20%3E%20target%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20return%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%23%20Include%20the%20current%20candidate%3A%20We%20append%20it%20to%20lst%2C%20update%20the%20total%2C%20and%20recurse%0A%20%20%20%20%20%20%20%20%20%20%20%20lst.append(candidates%5Bi%5D)%0A%20%20%20%20%20%20%20%20%20%20%20%20dfs(i%2C%20lst%2C%20total%2Bcandidates%5Bi%5D)%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%23%20Exclude%20the%20current%20candidate%3A%20We%20move%20to%20the%20next%20index%20without%20changing%20lst%20or%20total%0A%20%20%20%20%20%20%20%20%20%20%20%20lst.pop()%0A%20%20%20%20%20%20%20%20%20%20%20%20dfs(i%2B1%2C%20lst%2C%20total)%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20dfs(0%2C%20%5B%5D%2C%200)%0A%20%20%20%20%20%20%20%20return%20res%0A%0Asol%20%3D%20Solution()%0Aprint(sol.combinationSum(candidates%20%3D%20%5B2%2C3%2C6%2C7%5D%2C%20target%20%3D%207)))

----
### 2nd Sept

#### 29. [permutations](https://leetcode.com/problems/permutations/description/)

 swapping elements to generate all possible arrangements.
 time complexity of this algorithm is O(n!)

```
class Solution:
    def permute(self, nums):
        res = []
        def backtracking(start, end):
            
            #base case
            if start == end:
                res.append(nums[:])
            
            for i in range(start, end):
                nums[i], nums[start] = nums[start], nums[i]
                backtracking(start+1, end)
                nums[i], nums[start] = nums[start], nums[i]
        
        backtracking(0, len(nums))
        return res 
```

[visualize](https://memlayout.com?code=class%20Solution%3A%0A%20%20%20%20def%20permute(self%2C%20nums)%3A%0A%20%20%20%20%20%20%20%20res%20%3D%20%5B%5D%0A%20%20%20%20%20%20%20%20def%20backtracking(start%2C%20end)%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%23base%20case%0A%20%20%20%20%20%20%20%20%20%20%20%20if%20start%20%3D%3D%20end%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20res.append(nums%5B%3A%5D)%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20for%20i%20in%20range(start%2C%20end)%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20nums%5Bi%5D%2C%20nums%5Bstart%5D%20%3D%20nums%5Bstart%5D%2C%20nums%5Bi%5D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20backtracking(start%2B1%2C%20end)%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20nums%5Bi%5D%2C%20nums%5Bstart%5D%20%3D%20nums%5Bstart%5D%2C%20nums%5Bi%5D%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20backtracking(0%2C%20len(nums))%0A%20%20%20%20%20%20%20%20return%20res%20%0A%0Asol%20%3D%20Solution()%0Aprint(sol.permute(%5B1%2C2%2C3%5D)))

#### 30. [subsets](https://leetcode.com/problems/subsets/description/)

making a binary choice (include/exclude) for each element and exploring all paths in the resulting decision tree, we naturally generate all possible subsets of the input list.

```
class Solution:
    def subsets(self, nums):
        res = []
        subset = []
        def dfs(i):
            if i == len(nums):
                res.append(subset.copy())
                return 
            
            subset.append(nums[i])
            dfs(i + 1)
            subset.pop()
            dfs(i + 1)
        
        dfs(0)
        return res
```

[visualize](https://memlayout.com?code=class%20Solution%3A%0A%20%20%20%20def%20subsets(self%2C%20nums)%3A%0A%20%20%20%20%20%20%20%20res%20%3D%20%5B%5D%0A%20%20%20%20%20%20%20%20subset%20%3D%20%5B%5D%0A%20%20%20%20%20%20%20%20def%20dfs(i)%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20if%20i%20%3D%3D%20len(nums)%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20res.append(subset.copy())%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20return%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20subset.append(nums%5Bi%5D)%0A%20%20%20%20%20%20%20%20%20%20%20%20dfs(i%20%2B%201)%0A%20%20%20%20%20%20%20%20%20%20%20%20subset.pop()%0A%20%20%20%20%20%20%20%20%20%20%20%20dfs(i%20%2B%201)%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20dfs(0)%0A%20%20%20%20%20%20%20%20return%20res%0A%20%20%20%20%20%20%20%20%0Asol%20%3D%20Solution()%0Aprint(sol.subsets(%5B1%2C2%2C3%5D)))

-------

### 3rd Sept 

#### 31. [diameter of binary tree](https://leetcode.com/problems/diameter-of-binary-tree/description/)

```
class Solution:
    def diameterofBinaryTree(self, root):
        res = [0]
        def dfs(root):
            if not root:
                return -1
            left = dfs(root.left)
            right = dfs(root.right)
            res[0] = max(res[0], left + right + 2) #updates the maximum diameter.
            return 1 + max(left, right) #height of the subtree
        
        dfs(root)
        return res[0]
```
[visualize](https://memlayout.com?code=class%20TreeNode%3A%0A%20%20%20%20def%20__init__(self%2C%20val%3D0%2C%20left%3DNone%2C%20right%3DNone)%3A%0A%20%20%20%20%20%20%20%20self.val%20%3D%20val%0A%20%20%20%20%20%20%20%20self.left%20%3D%20left%0A%20%20%20%20%20%20%20%20self.right%20%3D%20right%0A%0Aclass%20Solution%3A%0A%20%20%20%20def%20diameterOfBinaryTree(self%2C%20root)%3A%0A%20%20%20%20%20%20%20%20res%20%3D%20%5B0%5D%0A%20%20%20%20%20%20%20%20def%20dfs(root)%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20if%20not%20root%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20return%20-1%0A%20%20%20%20%20%20%20%20%20%20%20%20left%20%3D%20dfs(root.left)%0A%20%20%20%20%20%20%20%20%20%20%20%20right%20%3D%20dfs(root.right)%0A%20%20%20%20%20%20%20%20%20%20%20%20res%5B0%5D%20%3D%20max(res%5B0%5D%2C%20left%20%2B%20right%20%2B%202)%0A%20%20%20%20%20%20%20%20%20%20%20%20return%201%20%2B%20max(left%2C%20right)%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20dfs(root)%0A%20%20%20%20%20%20%20%20return%20res%5B0%5D%0A%0A%23%20Create%20a%20binary%20tree%0A%23%20%20%20%20%20%20%201%0A%23%20%20%20%20%20%20%2F%20%5C%0A%23%20%20%20%20%202%20%20%203%0A%23%20%20%20%20%2F%20%5C%0A%23%20%20%204%20%20%205%0A%23%20%20%2F%0A%23%206%0A%0Aroot%20%3D%20TreeNode(1)%0Aroot.left%20%3D%20TreeNode(2)%0Aroot.right%20%3D%20TreeNode(3)%0Aroot.left.left%20%3D%20TreeNode(4)%0Aroot.left.right%20%3D%20TreeNode(5)%0Aroot.left.left.left%20%3D%20TreeNode(6)%0A%0A%23%20Create%20an%20instance%20of%20the%20Solution%20class%0Asolution%20%3D%20Solution()%0A%0A%23%20Calculate%20the%20diameter%20of%20the%20binary%20tree%0Adiameter%20%3D%20solution.diameterOfBinaryTree(root)%0A%0Aprint(f%22The%20diameter%20of%20the%20binary%20tree%20is%3A%20%7Bdiameter%7D%22))

#### 32. [increasing triplet subsequence](https://leetcode.com/problems/increasing-triplet-subsequence/?envType=study-plan-v2&envId=leetcode-75)

```
class Solution:
    def increasingTriplet(self, nums):
        first = second = float('inf')
        for i in nums:
            if i <= first:
                first = i
            elif i <= second:
                second = i
            else:
                return True
        
        return False
```
[visualize](https://memlayout.com?code=class%20Solution%3A%0A%20%20%20%20def%20increasingTriplet(self%2C%20nums)%3A%0A%20%20%20%20%20%20%20%20first%20%3D%20second%20%3D%20float('inf')%0A%20%20%20%20%20%20%20%20for%20i%20in%20nums%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20if%20i%20%3C%3D%20first%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20first%20%3D%20i%0A%20%20%20%20%20%20%20%20%20%20%20%20elif%20i%20%3C%3D%20second%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20second%20%3D%20i%0A%20%20%20%20%20%20%20%20%20%20%20%20else%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20return%20True%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20return%20False%0A%20%20%20%20%20%20%20%20%0Asol%20%3D%20Solution()%0Aprint(sol.increasingTriplet(%5B2%2C1%2C5%2C0%2C4%2C6%5D)))

------
### 4th Sept

#### 33. [move zeroes](https://leetcode.com/problems/move-zeroes/description/?envType=study-plan-v2&envId=leetcode-75)
```
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        non_zero = 0

        for i in range(len(nums)):
            if nums[i] != 0:
                nums[non_zero], nums[i] = nums[i], nums[non_zero]
                non_zero += 1
```
[visualize](https://memlayout.com?code=class%20Solution%3A%0A%20%20%20%20def%20moveZeroes(self%2C%20nums)%3A%0A%20%20%20%20%20%20%20%20%22%22%22%0A%20%20%20%20%20%20%20%20Do%20not%20return%20anything%2C%20modify%20nums%20in-place%20instead.%0A%20%20%20%20%20%20%20%20%22%22%22%0A%20%20%20%20%20%20%20%20non_zero%20%3D%200%0A%0A%20%20%20%20%20%20%20%20for%20i%20in%20range(len(nums))%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20if%20nums%5Bi%5D%20!%3D%200%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20nums%5Bnon_zero%5D%2C%20nums%5Bi%5D%20%3D%20nums%5Bi%5D%2C%20nums%5Bnon_zero%5D%0A%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20%20non_zero%20%2B%3D%201%0A%0Asol%20%3D%20Solution()%0Anums%20%3D%20%5B0%2C1%2C0%2C3%2C12%5D%0Asol.moveZeroes(nums)%0Aprint(nums))

#### 34. [swap nodes in pairs](https://leetcode.com/problems/swap-nodes-in-pairs/description/)

```
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        dummy.next = head
        prev = dummy

        while prev.next and prev.next.next:
            first = prev.next
            second = first.next
            
            first.next = second.next
            second.next = first
            prev.next = second

            prev = prev.next.next
        
        return dummy.next
```
[visualize](https://memlayout.com?code=%23%20Definition%20for%20singly-linked%20list.%0Aclass%20ListNode%3A%0A%20%20%20%20def%20__init__(self%2C%20val%3D0%2C%20next%3DNone)%3A%0A%20%20%20%20%20%20%20%20self.val%20%3D%20val%0A%20%20%20%20%20%20%20%20self.next%20%3D%20next%0A%0Aclass%20Solution%3A%0A%20%20%20%20def%20swapPairs(self%2C%20head%3A%20ListNode)%20-%3E%20ListNode%3A%0A%20%20%20%20%20%20%20%20dummy%20%3D%20ListNode()%0A%20%20%20%20%20%20%20%20dummy.next%20%3D%20head%0A%20%20%20%20%20%20%20%20prev%20%3D%20dummy%0A%0A%20%20%20%20%20%20%20%20while%20prev.next%20and%20prev.next.next%3A%0A%20%20%20%20%20%20%20%20%20%20%20%20first%20%3D%20prev.next%0A%20%20%20%20%20%20%20%20%20%20%20%20second%20%3D%20first.next%0A%20%20%20%20%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20%20%20%20%20%23%20Swapping%20pairs%0A%20%20%20%20%20%20%20%20%20%20%20%20first.next%20%3D%20second.next%0A%20%20%20%20%20%20%20%20%20%20%20%20second.next%20%3D%20first%0A%20%20%20%20%20%20%20%20%20%20%20%20prev.next%20%3D%20second%0A%0A%20%20%20%20%20%20%20%20%20%20%20%20%23%20Move%20prev%20to%20the%20next%20pair%0A%20%20%20%20%20%20%20%20%20%20%20%20prev%20%3D%20prev.next.next%0A%20%20%20%20%20%20%20%20%0A%20%20%20%20%20%20%20%20return%20dummy.next%0A%0A%23%20Test%20case%0Adef%20test_swapPairs()%3A%0A%20%20%20%20%23%20Creating%20the%20linked%20list%3A%201%20-%3E%202%20-%3E%203%20-%3E%204%0A%20%20%20%20node1%20%3D%20ListNode(1)%0A%20%20%20%20node2%20%3D%20ListNode(2)%0A%20%20%20%20node3%20%3D%20ListNode(3)%0A%20%20%20%20node4%20%3D%20ListNode(4)%0A%0A%20%20%20%20node1.next%20%3D%20node2%0A%20%20%20%20node2.next%20%3D%20node3%0A%20%20%20%20node3.next%20%3D%20node4%0A%20%20%20%20%0A%20%20%20%20%23%20Initialize%20solution%0A%20%20%20%20solution%20%3D%20Solution()%0A%20%20%20%20%0A%20%20%20%20%23%20Apply%20the%20swapPairs%20function%0A%20%20%20%20new_head%20%3D%20solution.swapPairs(node1)%0A%20%20%20%20%0A%20%20%20%20%23%20Check%20the%20result%0A%20%20%20%20%23%20Expected%20linked%20list%20after%20swap%3A%202%20-%3E%201%20-%3E%204%20-%3E%203%0A%20%20%20%20assert%20new_head.val%20%3D%3D%202%2C%20f%22Expected%202%2C%20got%20%7Bnew_head.val%7D%22%0A%20%20%20%20assert%20new_head.next.val%20%3D%3D%201%2C%20f%22Expected%201%2C%20got%20%7Bnew_head.next.val%7D%22%0A%20%20%20%20assert%20new_head.next.next.val%20%3D%3D%204%2C%20f%22Expected%204%2C%20got%20%7Bnew_head.next.next.val%7D%22%0A%20%20%20%20assert%20new_head.next.next.next.val%20%3D%3D%203%2C%20f%22Expected%203%2C%20got%20%7Bnew_head.next.next.next.val%7D%22%0A%20%20%20%20assert%20new_head.next.next.next.next%20is%20None%2C%20%22Expected%20None%20at%20the%20end%20of%20the%20linked%20list%22%0A%0A%20%20%20%20print(%22Test%20passed!%22)%0A%0A%23%20Run%20the%20test%20case%0Atest_swapPairs()) | [explain](https://withmarble.io/learn?type=problem_breakdown&session_id=5124469d-48e8-46f4-9e8d-93d9f367aa01)

-----
### 5th Sept

#### 35. [Determine if two strings are close](https://leetcode.com/problems/determine-if-two-strings-are-close/description/?envType=study-plan-v2&envId=leetcode-75)

```
class Solution:
    def closeStrings(self, word1: str, word2: str) -> bool:
        if set(word1) == set(word2):

            word1_c = Counter(word1)
            word2_c = Counter(word2)

            return sorted(word1_c.values()) == sorted(word2_c.values())
```
[explain](https://withmarble.io/learn?type=problem_breakdown&session_id=650803e2-7bd9-4730-b2a7-abeb00b2fc51)

#### 36. [Equal rows and cols](https://leetcode.com/problems/equal-row-and-column-pairs/description/)
```
#approach
- Iterate through each row
- For each row, compare it with every column
- If a row and column are equal, increment a counter
- Return the final count

class Solution:
    def equalPairs(self, grid: List[List[int]]) -> int:
        count = 0
        l = 0
        for i in range(len(grid)):
            for j in range(len(grid)):
                if all(grid[i][k] == grid[k][j] for k in range(len(grid))):
                    count += 1
        return count
```
[explain](https://withmarble.io/learn?type=problem_breakdown&session_id=a2477313-1efd-49a3-927d-d1cc6f8d9839)

-----
### 6th Sept

#### 37. [Can place flowers](https://leetcode.com/problems/can-place-flowers/?envType=study-plan-v2&envId=leetcode-75)
```
# approach
- Iterate through the flowerbed
- Check if a flower can be planted
- If yes, plant and increment count
- If count equals n, return true immediately

class Solution:
    def canPlaceFlowers(self, flowerbed, n):
        if n == 0:
            return True
        count = 0
        for i in range(1, len(flowerbed)-1):
            if flowerbed[i-1] == flowerbed[i] == flowerbed[i+1] == 0:
                flowerbed[i] = 1
                count += 1
                if count == n:
                    return True
        return False
```
[explain](https://withmarble.io/learn?type=problem_breakdown&session_id=17f82e29-f9de-4277-8a9c-53ebee4875ad)

#### 38. [Reverse vowels in string](https://leetcode.com/problems/reverse-vowels-of-a-string/description/?envType=study-plan-v2&envId=leetcode-75)

- two pointer 

```
class Solution:
    def reverseVowels(self, s: str) -> str:
        s = list(s)
        vowels = set('aeiouAEIOU')
        l = 0
        r = len(s)-1
        while l < r:
            if s[l] not in vowels:
                l += 1
            elif s[r] not in vowels:
                r -= 1
            else:
                s[l], s[r] = s[r], s[l]
                l += 1
                r -= 1
        print(s)
        return ''.join(s)
```
[explain](https://withmarble.io/learn?type=problem_breakdown&session_id=3102059e-2c4c-4a6e-9c20-c3cd7136d96e)

----
### 7th Sept

#### 39. [Max number of K pairs](https://leetcode.com/problems/max-number-of-k-sum-pairs/description/?envType=study-plan-v2&envId=leetcode-75)

- hashmap

```
#approach
- Count occurrences of each number in a dictionary
- Iterate through the array once
- For each number, check if its complement (k - num) exists
- Update the count and increment the operations counter

class Solution:
    def maxOperations(self, nums, k):
        count = Counter(nums)
        operations = 0
        
        for num in count:
            complement = k - num
            if num == complement:
                operations += count[num] // 2
            elif complement in count:
                operations += min(count[num], count[complement])
                count[num] = 0
                count[complement] = 0
        
        return operations
```
[explain](https://withmarble.io/learn?type=problem_breakdown&session_id=4e5d7ae5-9a09-47b0-b703-9b30760cf116)

#### 40. [Maximum average subarray I](https://leetcode.com/problems/maximum-average-subarray-i/description/?envType=study-plan-v2&envId=leetcode-75)

- sliding window

```
class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        curr_sum = sum(nums[:k])
        max_sum = curr_sum

        for i in range(1, len(nums) - k + 1):
            curr_sum = curr_sum - nums[i-1] + nums[i+k-1]
            max_sum = max(max_sum, curr_sum)

        return max_sum / k
```
[explain](https://withmarble.io/learn?type=problem_breakdown&session_id=3c9584fa-5333-4191-93e2-2f8e46b9601e)

-----

### 8th Sept

#### 41. [Max number of vowels in a substring of given length](https://leetcode.com/problems/maximum-number-of-vowels-in-a-substring-of-given-length/description/?envType=study-plan-v2&envId=leetcode-75)

- sliding window

```
def maxVowels(self, s, k):
        vowels = set('aeiou')
        count = sum(1 for char in s[:k] if char in vowels)
        # print('count', count)
        max_count = count
        
        for i in range(k, len(s)):
            if s[i-k] in vowels:
                count -= 1
            if s[i] in vowels:
                count += 1
            max_count = max(max_count, count)
        
        return max_count
```
[explain](https://withmarble.io/learn?type=problem_breakdown&session_id=5a2fc79e-ebdd-4228-863c-121d515dce60)

#### 42. [max consecutive ones III](https://leetcode.com/problems/max-consecutive-ones-iii/description/?envType=study-plan-v2&envId=leetcode-75)

- sliding window

```
class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        left = right = zeros_count = max_length = 0
    
        for right in range(len(nums)):
            if nums[right] == 0:
                zeros_count += 1
            
            while zeros_count > k:
                if nums[left] == 0:
                    zeros_count -= 1
                left += 1
            
            max_length = max(max_length, right - left + 1)
        
        return max_length
        
```
[explain](https://withmarble.io/learn?type=problem_breakdown&session_id=3d7018a7-9fa0-4050-bff4-8bb490c647f7)

----

### 9th Sept

#### 43. [Longest Subarray of 1's After Deleting One Element](https://leetcode.com/problems/longest-subarray-of-1s-after-deleting-one-element/description/?envType=study-plan-v2&envId=leetcode-75)

- sliding window

```
class Solution:
    def longestSubarray(self, nums):
        n = len(nums)
        left = zeros = ans = 0
        
        for right in range(n):
            if nums[right] == 0:
                zeros += 1
            
            while zeros > 1:
                if nums[left] == 0:
                    zeros -= 1
                left += 1
            
            ans = max(ans, right - left + 1 - zeros)
        
        return ans - 1 if ans == n else ans
```
[explain](https://withmarble.io/learn?type=problem_breakdown&session_id=50d6302c-0b42-4281-87c6-634d9249a31d)

#### 44. [find pivot index](https://leetcode.com/problems/find-pivot-index/description/?envType=study-plan-v2&envId=leetcode-75)

```
class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        
        left_sum = 0
        total_sum = sum(nums)
        for r in range(len(nums)):
            right_sum = total_sum - left_sum - nums[r]

            if left_sum == right_sum:
                return r
            left_sum += nums[r]
        
        return -1

```
[explain](https://withmarble.io/learn?type=problem_breakdown&session_id=e8c54769-5138-4646-82a5-0ed72dfc8294)

----
### 10th sept

