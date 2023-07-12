"""
This is a script for an mdp for assignment submission when difficulty of the task 
is uncertain and the only way to find out is to try the task. This then leads to 
probabilistic transitions to one of N difficulty states which have different efforts 
required to complete the task. Actions and reward structure are like
the simple case for assignment submission in efficacy_model.py
"""

import numpy as np
import matplotlib as mpl
mpl.rcParams['font.size'] = 14
mpl.rcParams['lines.linewidth'] = 2
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
import mdp_algms

#%%
# construct reward functions
def get_reward_functions(states, reward_pass, reward_fail, reward_shirk, reward_completed, 
                         effort_try, effort_work, effort_shirk):
    
    # reward from actions within horizon
    reward_func = [] 
    reward_func.append( [effort_try, reward_shirk + effort_shirk] ) # reward in start state
    
    for i in range( len(states)-2 ):
        reward_func.append( [effort_work[i], reward_shirk + effort_shirk]  )   # rewards in non-completed states
        
    reward_func.append( [reward_completed] ) # reward in completed state
    
    # reward from final evaluation
    reward_func_last = []
    reward_func_last.append(reward_fail)
    
    for i in range( len(states)-2 ):
        reward_func_last.append( 0 )   # rewards in non-completed states

    reward_func_last.append(reward_pass) 
    
    return np.array(reward_func), np.array(reward_func_last)


#%%
# initialising MDP

# states of markov chain
N_DIFFICULTY_STATES = 2 # number of states of difficulty
STATES = np.arange(2 + N_DIFFICULTY_STATES) # all states including start and finished

ACTIONS = []
ACTIONS = [ ['work', 'shirk'] for i in range( len(STATES)-1 ) ]
ACTIONS.append(['completed']) 

HORIZON = 10 # deadline
DISCOUNT_FACTOR = 0.9 # discounting factor
EFFICACY = 0.8 # self-efficacy (probability of progress on working) in non-start/finished state
DIFFICULTY_PROBABILITY = [0.9, 0.1] # probability for difficulty states

# utilities :
REWARD_PASS = 4.0 
REWARD_FAIL = -4.0
REWARD_SHIRK = 0.5
EFFORT_TRY = -0.1 # effort to check 
EFFORT_WORK = [-0.1, -5.0] # effort to complete task from one of the difficulty states
EFFORT_SHIRK = -0 
REWARD_COMPLETED = REWARD_SHIRK

# transition function for each state:
T = []
T.append( [ np.array([0, DIFFICULTY_PROBABILITY[0], DIFFICULTY_PROBABILITY[1], 0]), 
            np.array([1, 0, 0, 0]) ] ) # transitions for check, shirk in start state
T.append( [ np.array([0, 1-EFFICACY, 0, EFFICACY]), 
            np.array([0, 1, 0, 0]) ] ) # transitions for work, shirk
T.append( [ np.array([0, 0, 1-EFFICACY, EFFICACY]), 
            np.array([0, 0, 1, 0]) ] ) # transitions for work, shirk
T.append( [ np.array([0, 0, 0, 1]) ] ) # transitions for completed
T = np.array(T)


#%%
# example policy 

reward_func, reward_func_last = get_reward_functions(STATES, REWARD_PASS, REWARD_FAIL, REWARD_SHIRK, 
                                                     REWARD_COMPLETED, EFFORT_TRY, EFFORT_WORK, EFFORT_SHIRK)
V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy(STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR, 
                              reward_func, reward_func_last, T)
