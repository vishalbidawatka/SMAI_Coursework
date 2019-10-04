#!/usr/bin/env python
# coding: utf-8

# House price prediction is a regression problem whereas predicting labels in the Apparal dataset was a classification problem.
# 
# The difference between a classification and regression is that a classification outputs a prediction probability for class/classes and regression provides a value. 
# 
# We can make a neural network to output a value by simply changing the activation function in the final layer to output the values.
# 
# By changing the activation function such as sigmoid,relu,tanh,etc. we can use a function (f(x)=x). So while back propagation we can simply derive f(x).
# 
# Artificial neural networks are often (demeneangly) called "glorified regressions". The main difference between ANNs and multiple / multivariate linear regression is of course, that the ANN models nonlinear relationships.
# 
# In multi-class classification, the assumption we typically make about our target data is that they are distributed according to the categorical distribution. In this case, we would choose the output of our 
# neural network to be the softmax function.
# 
# This way we turn our regressor, which smoothly approximates the categorical distribution, into a “hard” classifier by choosing to interpret its output as prediction through hard assignment (via argmax).
