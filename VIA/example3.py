#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:41:46 2026

@author: stylesj
"""
from VIA import value_iteration


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
    epsilon = 1e-6

    policy, value_function = value_iteration(states, actions, P, R, gamma, epsilon)
    return policy, value_function

if __name__ == "__main__":
    #only run if script run directly (not imported)
    policy, value_function = grid_world_mdp()
    print("Optimal Policy:", policy)
    print("Value Function:", value_function)