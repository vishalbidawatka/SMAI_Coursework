#!/usr/bin/env python
# coding: utf-8

# ## Funtion to handle missing values in test data
# >- I implemented what was taught in the tutorial, what i did is that if a missing value is found in test row, it makes a recursive call on each of it's children and all the children returns the majority of ones or zeroes they have traversing throughout the leaf node.
# 
# >- if there are other features that are present in that row, the recursion will follow that part only 
# 
# >- Also there many other methods like, you can ignore that missing value and return the decision on that node only by the probablity.
# 
# >- Also there was increase of 0.5 % in accuracy by implementing the recursive way.
# 
# >- Also, we can fill all missing values with -1 that will tell oue model that this value is missing
