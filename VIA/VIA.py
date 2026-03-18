#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 09:58:26 2026

@author: stylesj
"""

#609 cw 2
#Value Iteration Algorithms

import numpy as np
from typing import Callable, List, Any




#define value iteration process
def value_iteration(S: List[Any], A: List[Any], P: Callable[ [Any,Any,Any], float],
                    R: Callable[ [Any,Any,Any], float], gamma: float = 0.9,
                    epsilon: float = 0.001, max_iterations: int = 10000) -> tuple[dict, dict, list]:
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
    iteration = 0
    delta_list = []
    
    while True:
        delta = 0
        iteration += 1
        for s in S:
            #Check if the state is effectively terminal
            has_outgoing = any(P(s, a, s_next) > 0 for a in A for s_next in S)
            if not has_outgoing:
                continue  # skip terminal state
                
            v = V[s]
            V[s] = max(sum(P(s, a, s_next) * (R(s, a, s_next) + gamma * V[s_next]) for s_next in S) for a in A)
            delta = max(delta, abs(v - V[s]))
        delta_list.append(delta)
        if delta < epsilon:
            break
        if iteration >= max_iterations:
            print(f"Warning: reached maximum iterations ({max_iterations}) without full convergence.")
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
    return policy, V, delta_list















