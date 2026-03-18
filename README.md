# Value Iteratation Algorithm Project

This is a git repository for VIA package, which uses the value iteration algorithm to solve MDP. More more information see Artificial Intelligence: Foundations and Computational Agents 2nd edition.

There are three examples: 
- example1: a simple example of 3 states and two action
- example2: exercise 9.27 in section 9.5 of Artificial Intelligence: Foundations and Computational Agents 2nd edition, see the jupyter notebook ex2_problem for more details
- example3 the grid world problem, see the jupyter notebook grid_world_problem for implementation and results


## How to Use

To install package from github:

python -m pip install 'git+https://github.com/Jay-Sty/Stor609cw2'

## Details of value_iteration function

The main function is value_iterations which takes list of states and actions; the probabailities and rewards and functions (note that some examples use these as dictionaries but this must be wrapped in a function); gamma value; and finally the convergence specification and max number of iterations.
The function returns the optimals policy, the optimal value of each state and list of convergence differences (this is used for plotting convergence)


## Psudocode

The psudo code for the value_iteration function is slightly different to that given in figure 9.16 of Artificial Intelligence: Foundations and Computational Agents 2nd edition.

- Initialisation: this function initialised all value V to be zero at the start.
- Convergence and Max iterations: in this function we define termination as untill convergence to some amount epsilon or until a maximum number of iterations is reached.
- Bellman Equation: The Bellman equation is calulated slightly differenty to account for if reward depends on next state not just current state.
- Terminal Nodes: This function check for 'terminal nodel' where there is no outgoing actions from the node and therefor does not calculate its value.


### The psudocode for the value_iteration function is:

<pre> ```
Procedure: Value_Iteration(S, A, P, R, gamma, epsilon, max_iterations)
Inputs:
    S              : set of all states
    A              : set of all actions
    P(s_next|s,a)  : state transition probability function
    R(s,a,s_next)  : reward function
    gamma          : discount factor (0 <= gamma < 1)
    epsilon        : convergence threshold
    max_iterations : maximum allowed iterations
Outputs:
    policy         : optimal action for each state
    V              : value function for each state
    delta_list     : list of maximum value changes per iteration

1. Initialize V[s] = 0 for all s in S
2. iteration = 0
3. delta_list = []

4. Repeat:
       iteration += 1
       delta = 0
       
       For each state s in S:
           If s has no outgoing transitions (i.e., terminal):
               Continue to next state
           
           v_old = V[s]
           
           # Update value using Bellman optimality
           V[s] = max over actions a in A of:
                      sum over s_next in S of:
                          P(s_next | s, a) * ( R(s, a, s_next) + gamma * V[s_next] )
           
           delta = max(delta, |v_old - V[s]|)
       
       Append delta to delta_list
       
       If delta < epsilon:
           Break  # converged
       
       If iteration >= max_iterations:
           Print warning about max iterations reached
           Break

5. For each state s in S:
       If s has no outgoing transitions:
           policy[s] = None
           Continue
       policy[s] = argmax over actions a in A of:
                       sum over s_next in S of:
                           P(s_next | s, a) * ( R(s, a, s_next) + gamma * V[s_next] )

6. Return policy, V, delta_list
``` </pre>
