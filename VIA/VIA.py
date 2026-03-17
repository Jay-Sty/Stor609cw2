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
            #Check if the state is effectively terminal
            has_outgoing = any(P(s, a, s_next) > 0 for a in A for s_next in S)
            if not has_outgoing:
                continue  # skip terminal state
                
            v = V[s]
            V[s] = max(sum(P(s, a, s_next) * (R(s, a, s_next) + gamma * V[s_next]) for s_next in S) for a in A)
            delta = max(delta, abs(v - V[s]))
        if delta < epsilon:
            break
    policy = {}
    for s in S:
        has_outgoing = any(P(s, a, s_next) > 0 for a in A for s_next in S)
        if not has_outgoing:
            policy[s] = None
            continue
        policy[s] = max(A, 
                        key=lambda a: sum(
                          P(s, a, s_next) *(R(s, a, s_next) + gamma * V[s_next]) for s_next in S))
    return policy, V



################################################
##example 1 
#define MDP components
# states = [0, 1, 2]  
# actions = [0, 1]     

# def transition_model(s, a, s_next):
#     if (s == 0 and a == 0 and s_next == 1) or (s == 1 and a == 0 and s_next == 0):
#         return 1
#     elif (s == 0 and a == 1 and s_next == 2) or (s == 2 and a == 1 and s_next == 1):
#         return 1
#     return 0

# def reward_function(s, a, s_next):
#     if s == 0 and a == 0 and s_next == 1:
#         return 10
#     elif s == 0 and a == 1 and s_next == 2:
#         return 5
#     return 0
# gamma = 0.9  
# epsilon = 0.01

# #run algorithm
# policy, value_function = value_iteration(states, actions, transition_model, reward_function, gamma, epsilon)

# print("Optimal Policy:", policy)
# print("Value Function:", value_function)


###############################################
##example 2 (ex9.27 from of Artificial Intelligence: Foundations and Computational Agents 2nd edition)


def sam_weekend_mdp():
    states = ['healthy', 'sick']
    actions = ['relax', 'party']

    #Transition and reward dictionaries
    P_dict = {
        'healthy': {'relax': {'healthy': 0.95, 'sick': 0.05},
                    'party': {'healthy': 0.7, 'sick': 0.3}},
        'sick': {'relax': {'healthy': 0.5, 'sick': 0.5},
                 'party': {'healthy': 0.1, 'sick': 0.9}}
    }

    R_dict = {
        'healthy': {'relax': 7, 'party': 10},
        'sick': {'relax': 0, 'party': 2}
    }

    #Wrapper functions
    def P(s, a, s_next):
        return P_dict[s][a].get(s_next, 0)

    def R(s, a, s_next):
        return R_dict[s][a]

    gamma = 0.9
    epsilon = 0.01

    policy, value_function = value_iteration(states, actions, P, R, gamma, epsilon)
    return policy, value_function

if __name__ == "__main__":
    #only run if script run directly (not imported)
    policy, value_function = sam_weekend_mdp()
    print("Optimal Policy:", policy)
    print("Value Function:", value_function)


######################################################
## Grid World Problem

def grid_world_mdp():
    #Define states and actions
    states = ['TL', 'TR', 'BL', 'BR']
    actions = ['R', 'L', 'U', 'D']
    
    P_dict = {
        'TL': {'R': {'TR': 0.9, 'BL': 0.1},
               'D': {'BL': 0.9, 'TR': 0.1}},
        'TR': {'L': {'TL': 0.9, 'BR': 0.1},
               'D': {'BR': 0.8, 'TL': 0.2}},
        'BL': {'R': {'BR': 0.9, 'TL': 0.1},
               'U': {'TL': 0.8, 'BR': 0.2}},
        'BR': {}
        }
    
    # R[(s, a, s_next)] = reward for that transition
    R_dict = {
        ('TL', 'R', 'TR'): -1,
        ('TL', 'R', 'BL'): -2,
        ('TL', 'D', 'BL'): -2,
        ('TL', 'D', 'TR'): -1,
        ('TR', 'L', 'TL'): -3/2,
        ('TR', 'L', 'BR'): 10,
        ('TR', 'D', 'BR'): 15,
        ('TR', 'D', 'TL'): -1,
        ('BL', 'R', 'BR'): 20,
        ('BL', 'R', 'TL'): -5/2,
        ('BL', 'U', 'TL'): -1/2,
        ('BL', 'U', 'BR'): 5,
        }
    
    def P(s, a, s_next):
        #return P_dict[s][a].get(s_next, 0)
        #return P_dict.get((s, a), {}).get(s_next, 0)
        return P_dict.get(s, {}).get(a, {}).get(s_next, 0)

    def R(s, a, s_next):
        return R_dict.get((s, a, s_next), 0)
        
    
    gamma = 0.9
    epsilon = 0.01

    policy, value_function = value_iteration(states, actions, P, R, gamma, epsilon)
    return policy, value_function

if __name__ == "__main__":
    #only run if script run directly (not imported)
    policy, value_function = grid_world_mdp()
    print("Optimal Policy:", policy)
    print("Value Function:", value_function)










