# Value Iteratation Algorithm Project

This is a git repository for VIA package, which uses the value iteration algorithm to solve MDP. More more information see Artificial Intelligence: Foundations and Computational Agents 2nd edition.

There are three examples: 
- example1: a simple example of 3 states and two action
- example2: exercise 9.27 in section 9.5 of Artificial Intelligence: Foundations and Computational Agents 2nd edition, see the jupyter notebook ex2_problem for more details
- example3 the grid world problem, see the jupyter notebook grid_world_problem for implementation and results


## How to Use

To install package from github:

python -m pip install 'git+https://github.com/Jay-Sty/Stor609cw2'

## Detail of value_iteration function

The main function is value_iterations which takes list of states and actions; the probabailities and rewards and functions (note that some examples use these as dictionaries but this must be wrapped in a function); gamma value; and finally the convergence specification and max number of iterations.
The function returns the optimals policy, the optimal value of each state and list of convergence differences (this is used for plotting convergence)


## Psudo Code
