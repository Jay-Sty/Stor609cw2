#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:41:17 2026

@author: stylesj
"""
from VIA import value_iteration


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
    epsilon = 0.05
    max_iterations = 10000

    policy, value_function = value_iteration(states, actions, P, R, gamma, epsilon, max_iterations)
    return policy, value_function

if __name__ == "__main__":
    #only run if script run directly (not imported)
    policy, value_function = sam_weekend_mdp()
    print("Optimal Policy:", policy)
    print("Value Function:", value_function)