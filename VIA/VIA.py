#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 09:58:26 2026

@author: stylesj
"""

#609 cw 2
#Value Iteration Algorithms

import numpy as np





#define value iteration process
def value_iteration(S, A, P, R, gamma = 0.9, epsilon = 0.01):
    """
    Perform Value Iteration for a given MDP.

    Parameters:
    S: list of state names
    A: list of action names
    P: function
       State transition probability function P(s_next | s, a). Takes three arguments:
       current state `s`, action `a`, and next state `s_next`, and returns the probability of transitioning.
    R: function
       Reward function R(s, a, s_next). Takes three arguments:
       current state `s`, action `a`, and next state `s_next`, and returns the immediate reward for that transition.
    gamma: float, optional (default=0.9)
        Discount factor for future rewards, in the range [0,1).
    epsilon: float, optional (default=0.01)
        Small threshold for convergence. Iteration stops when the maximum change in value across states is less than epsilon.

    Returns:
    V: dictionary of optimal state values
    policy: dictionary of optimal action for each state
    """
    V = {s: 0 for s in S}
    
    while True:
        delta = 0
        for s in S:
            v = V[s]
            V[s] = max(sum(P(s, a, s_next) * (R(s, a, s_next) + gamma * V[s_next]) for s_next in S) for a in A)
            delta = max(delta, abs(v - V[s]))
        if delta < epsilon:
            break
    policy = {}
    for s in S:
        policy[s] = max(A, 
                        key=lambda a: sum(
                          P(s, a, s_next) *(R(s, a, s_next) + gamma * V[s_next]) for s_next in S))
    return policy, V



################################################
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
epsilon = 0.01

#run algorithm
policy, value_function = value_iteration(states, actions, transition_model, reward_function, gamma, epsilon)

print("Optimal Policy:", policy)
print("Value Function:", value_function)


###############################################
##example 2 (ex9.27 from of Artificial Intelligence: Foundations and Computational Agents 2nd edition)


states = ["healthy", "sick"]
actions = ["relax", "party"]

def transition_model(s, a, s_next):
    if s == 'healthy' and a == 'relax':
        return 0.95 if s_next == 'healthy' else 0.05
    elif s == 'healthy' and a == 'party':
        return 0.7 if s_next == 'healthy' else 0.3
    elif s == 'sick' and a == 'relax':
        return 0.5 if s_next == 'healthy' else 0.5
    elif s == 'sick' and a == 'party':
        return 0.1 if s_next == 'healthy' else 0.9
    return 0

def reward_function(s, a, s_next):
    if s == 'healthy' and a == 'relax':
        return 7
    elif s == 'healthy' and a == 'party':
        return 10
    elif s == 'sick' and a == 'relax':
        return 0
    elif s == 'sick' and a == 'party':
        return 2
    return 0

#run algorithm
policy, value_function = value_iteration(states, actions, transition_model, reward_function, gamma, epsilon)

print("Optimal Policy:", policy)
print("Value Function:", value_function)


######################################################
##







