import numpy as np
import matplotlib as mpl
mpl.rcParams['font.size'] = 14
mpl.rcParams['lines.linewidth'] = 2
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
import matplotlib.patches as mpatches
import mdp_algms

#%%

# instantiating MDP

# states of markov chain
N_DIFFICULTY_STATES = 3 # number of states of difficulty
STATES = np.arange(1 + N_DIFFICULTY_STATES) # final state and other initial states with varying difficulties of task completion

ACTIONS = []
ACTIONS = [ ['work', 'shirk'] for i in range( len(STATES)-1 ) ]
ACTIONS.append(['completed']) 

HORIZON = 10 # deadline
DISCOUNT_FACTOR = 1.0 # discounting factor
EFFICACY = 0.5 # self-efficacy (probability of progress on working) in non-start/finished state

# utilities :
REWARD_PASS = 4.0 
REWARD_FAIL = -4.0
REWARD_SHIRK = 0.5
EFFORT_TRY = -0.1 # effort to check 
EFFORT_WORK = [-0.2, -0.5, -1.0] # effort to complete task from one of the difficulty states
EFFORT_SHIRK = -0 
REWARD_COMPLETED = REWARD_SHIRK

# transition function for each state:
T = []
T.append( [ np.array([0, 1-EFFICACY, DIFFICULTY_PROBABILITY[1], 0]), 
            np.array([1, 0, 0, 0]) ] ) # transitions for work, shirk in start state
T.append( [ np.array([0, 1-EFFICACY, 0, EFFICACY]), 
            np.array([0, 1, 0, 0]) ] ) # transitions for work, shirk
T.append( [ np.array([0, 0, 1-EFFICACY, EFFICACY]), 
            np.array([0, 0, 1, 0]) ] ) # transitions for work, shirk
T.append( [ np.array([0, 0, 0, 1]) ] ) # transitions for completed
T = np.array(T)

