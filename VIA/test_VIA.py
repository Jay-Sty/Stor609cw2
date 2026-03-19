#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:50:42 2026

@author: stylesj
"""

import pytest
from VIA.VIA import value_iteration
from VIA.example2 import sam_weekend_mdp


#check that returned outputs are dictionaries 
#check match set of states and action
def test_output_types():
    states = ['healthy', 'sick']
    actions = ['relax', 'party']

    policy, V = sam_weekend_mdp()

    # dictionaries
    assert isinstance(policy, dict)
    assert isinstance(V, dict)

    #match
    assert set(policy.keys()) == set(states)
    assert set(V.keys()) == set(states)

    #valid actions or None
    for s in policy:
        assert policy[s] in actions or policy[s] is None

    #Value function should be numeric
    for s in V:
        assert isinstance(V[s], (int, float))


#returns the states inputed in output
def test_expected_states():
    policy, V = sam_weekend_mdp()

    expected_states = {'healthy', 'sick'}

    assert set(policy.keys()) == expected_states
    assert set(V.keys()) == expected_states


def test_value_ordering():
    policy, V = sam_weekend_mdp()

    #being healthy should be better than being sick
    assert V['healthy'] > V['sick']

def test_convergence_and_policy():
    #Rebuild MDP locally to access delta_list
    states = ['healthy', 'sick']
    actions = ['relax', 'party']

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

    def P(s, a, s_next):
        return P_dict[s][a].get(s_next, 0)

    def R(s, a, s_next):
        return R_dict[s][a]

    policy, V, delta_list = value_iteration(states, actions, P, R, 0.9, 0.001, 10000)

    #check that delta decreases and final delta is small
    assert delta_list[-1] < 0.01
    assert all(d >= 0 for d in delta_list)
    
    #test the opt policy is correct (use set gamma = 0.9)
    assert policy['healthy'] == 'party'
    assert policy['sick'] == 'relax'
    
    
    




