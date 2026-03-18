#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 10:50:42 2026

@author: stylesj
"""


import unittest
from VIA import value_iteration

class TestValueIteration(unittest.TestCase):

    def setUp(self):
        # Sam's weekend MDP
        self.states = ['healthy', 'sick']
        self.actions = ['relax', 'party']

        # Transition function
        self.P = lambda s,a,s_next: {
            ('healthy','relax','healthy'): 0.95,
            ('healthy','relax','sick'): 0.05,
            ('healthy','party','healthy'): 0.7,
            ('healthy','party','sick'): 0.3,
            ('sick','relax','healthy'): 0.5,
            ('sick','relax','sick'): 0.5,
            ('sick','party','healthy'): 0.1,
            ('sick','party','sick'): 0.9
        }.get((s,a,s_next), 0)

        # Reward function
        self.R = lambda s,a,s_next: {
            ('healthy','relax','healthy'): 7,
            ('healthy','relax','sick'): 7,
            ('healthy','party','healthy'): 10,
            ('healthy','party','sick'): 10,
            ('sick','relax','healthy'): 0,
            ('sick','relax','sick'): 0,
            ('sick','party','healthy'): 2,
            ('sick','party','sick'): 2
        }.get((s,a,s_next), 0)

        self.gamma = 0.9

    def test_full_convergence(self):
        """Test VIA until full convergence."""
        policy, V, delta_list = value_iteration(
            self.states, self.actions, self.P, self.R,
            gamma=self.gamma, epsilon=0.001, max_iterations=10000
        )

        # Expected approximate values from full convergence
        expected_values = {
            'healthy': 67.066, 
            'sick': 54.872
        }
        tolerance = 0.5

        # Check policy
        self.assertEqual(policy['healthy'], 'party', "Healthy state policy should be 'party'")
        self.assertEqual(policy['sick'], 'relax', "Sick state policy should be 'relax'")

        # Check value function within tolerance
        for s in self.states:
            self.assertAlmostEqual(V[s], expected_values[s], delta=tolerance,
                                   msg=f"Value for state {s} out of tolerance")

    def test_two_iterations(self):
        """Test VIA after 2 iterations."""
        policy, V, delta_list = value_iteration(
            self.states, self.actions, self.P, self.R,
            gamma=self.gamma, epsilon=0.001, max_iterations=2
        )

        # After 2 iterations, values should be roughly (from hand calculation)
        expected_values_2iter = {
            'healthy': 17.515,  # approximate, might differ slightly
            'sick': 9.90675
        }
        tolerance = 5.0  # loose tolerance since it's only 2 iterations

        # Policy sanity check (emerging optimal)
        self.assertEqual(policy['healthy'], 'party', "Healthy state policy should be 'party'")
        self.assertEqual(policy['sick'], 'relax', "Sick state policy should be 'relax'")

        # Value check (loose tolerance)
        for s in self.states:
            self.assertAlmostEqual(V[s], expected_values_2iter[s], delta=tolerance,
                                   msg=f"Value for state {s} after 2 iterations out of tolerance")


if __name__ == "__main__":
    unittest.main()


