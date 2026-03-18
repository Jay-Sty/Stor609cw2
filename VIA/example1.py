#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:36:15 2026

@author: stylesj
"""

from VIA import value_iteration

##example 1 
#define MDP components
states = [0, 1, 2]  
actions = [0, 1]     

def transition_model(s, a, s_next):
    if (s == 0 and a == 0 and s_next == 1) or (s == 1 and a == 0 and s_next == 0):
        return 1
    elif (s == 0 and a == 1 and s_next == 2) or (s == 2 and a == 1 and s_next == 1):
        return 1
    return 0

def reward_function(s, a, s_next):
    if s == 0 and a == 0 and s_next == 1:
        return 10
    elif s == 0 and a == 1 and s_next == 2:
        return 5
    return 0
gamma = 0.9  
epsilon = 0.001

#run algorithm
policy, value_function = value_iteration(states, actions, transition_model, reward_function, gamma, epsilon)

print("Optimal Policy:", policy)
print("Value Function:", value_function)