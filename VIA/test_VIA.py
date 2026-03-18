#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:50:42 2026

@author: stylesj
"""

# test_via.py
import unittest
from VIA import value_iteration

class TestVIA(unittest.TestCase):

    def setUp(self):
        self.states = ['healthy','sick']
        self.actions = ['relax','party']

        self.P = lambda s,a,s_next: {
            ('healthy','relax','healthy'):0.95,
            ('healthy','relax','sick'):0.05,
            ('healthy','party','healthy'):0.7,
            ('healthy','party','sick'):0.3,
            ('sick','relax','healthy'):0.5,
            ('sick','relax','sick'):0.5,
            ('sick','party','healthy'):0.1,
            ('sick','party','sick'):0.9
        }.get((s,a,s_next),0)

        self.R = lambda s,a,s_next: {
            ('healthy','relax','healthy'):7,
            ('healthy','relax','sick'):7,
            ('healthy','party','healthy'):10,
            ('healthy','party','sick'):10,
            ('sick','relax','healthy'):0,
            ('sick','relax','sick'):0,
            ('sick','party','healthy'):2,
            ('sick','party','sick'):2
        }.get((s,a,s_next),0)

    def test_values_and_policy(self):
        policy, V = value_iteration(self.states, self.actions, self.P, self.R, gamma = 0.9, epsilon = 0.001, max_iterations = 10000)
        # Healthy state should be more valuable than sick
        self.assertGreater(V['healthy'], V['sick'])
        # Healthy should prefer party
        self.assertEqual(policy['healthy'],'party')
        # Sick should prefer relax
        self.assertEqual(policy['sick'],'relax')

if __name__ == "__main__":
    unittest.main()