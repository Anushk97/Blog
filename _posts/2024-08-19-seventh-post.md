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
            maxD = max(maxD, count[s[r]])
        
        if (i - l + 1) - maxD > k:
            count[s[l]] -= 1
            l += 1
        
        return (i - l + 1)
```
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

