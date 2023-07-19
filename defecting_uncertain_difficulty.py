"""
This is a script for an mdp when difficulty of the task 
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
    
    return np.array(reward_func, dtype=object), np.array(reward_func_last, dtype=object)

def get_transition_prob(states, efficacy, difficulty_probability):
    
    T = []
    T.append( [ np.array([0, difficulty_probability[0], difficulty_probability[1], 0]), 
                np.array([1, 0, 0, 0]) ] ) # transitions for check, shirk in start state
    T.append( [ np.array([0, 1-efficacy, 0, efficacy]), 
                np.array([0, 1, 0, 0]) ] ) # transitions for work, shirk
    T.append( [ np.array([0, 0, 1-efficacy, efficacy]), 
                np.array([0, 0, 1, 0]) ] ) # transitions for work, shirk
    T.append( [ np.array([0, 0, 0, 1]) ] ) # transitions for completed
    
    return np.array(T, dtype=object)

#%%
# instantiating MDP

# states of markov chain
N_DIFFICULTY_STATES = 2 # number of states of difficulty
STATES = np.arange(2 + N_DIFFICULTY_STATES) # all states including start and finished

ACTIONS = []
ACTIONS = [ ['work', 'shirk'] for i in range( len(STATES)-1 ) ]
ACTIONS.append(['completed']) 

HORIZON = 10 # deadline
DISCOUNT_FACTOR = 1.0 # discounting factor
EFFICACY = 0.5 # self-efficacy (probability of progress on working) in non-start/finished state
DIFFICULTY_PROBABILITY = [0.9, 0.1] # probability for difficulty states

# utilities :
REWARD_PASS = 4.0 
REWARD_FAIL = -4.0
REWARD_SHIRK = 0.5
EFFORT_TRY = -0.1 # effort to check 
EFFORT_WORK = [-0.2, -1.0] # effort to complete task from one of the difficulty states
EFFORT_SHIRK = -0 
REWARD_COMPLETED = REWARD_SHIRK

# transition function for each state:



#%%
# example policy 

reward_func, reward_func_last = get_reward_functions(STATES, REWARD_PASS, REWARD_FAIL, REWARD_SHIRK, 
                                                     REWARD_COMPLETED, EFFORT_TRY, EFFORT_WORK, EFFORT_SHIRK)
T = get_transition_prob(STATES, EFFICACY, DIFFICULTY_PROBABILITY)
V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy(STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR, 
                              reward_func, reward_func_last, T)

# plots of policies and values
colors = plt.cm.Blues(np.linspace(0.4,0.9,len(STATES)))
lines = ['--', ':']
fig, axs = plt.subplots( figsize = (8, 6) )
fig1, axs1 = plt.subplots( figsize = (5, 3) )

for i_state, state in enumerate(STATES[:-1]):
    
    #plt.figure( figsize = (8, 6) )
    
    axs.plot(V_opt[i_state], color = colors[i_state], marker = 'o', linestyle = 'None', label = f'$V^*({i_state})$',)
    axs1.plot(policy_opt[i_state], color = colors[i_state], label = f'State {i_state}')
    
    for i_action, action in enumerate(ACTIONS[i_state]):
        
        axs.plot(Q_values[i_state][i_action, :], color = colors[i_state], linestyle = lines[i_action])
        
        
handles, labels = axs.get_legend_handles_labels()   
handles.append(axs.plot([], [], color = 'black', linestyle = '--', label = '$Q(a=$ check or work$)$'))
handles.append(axs.plot([], [], color = 'black', linestyle = ':', label = '$Q(a=$ shirk$)$'))

axs.legend()
axs.set_xlabel('timesteps')
axs1.legend()
axs1.set_xlabel('timesteps')
axs1.set_yticks([0,1])
axs1.set_yticklabels(['WORK', 'SHIRK'])
axs1.set_ylabel('policy')



#%%

# solving for policies for a range of efforts
probabilities = np.linspace(0.0, 1.0, 5)
# optimal starting point for 4 reward regimes (change manually)
start_works = np.full( (len(probabilities), N_DIFFICULTY_STATES+1), np.nan ) 

for i_prob, difficulty_probability in enumerate(probabilities):

    reward_func, reward_func_last = get_reward_functions(STATES, REWARD_PASS, REWARD_FAIL, REWARD_SHIRK, 
                                                         REWARD_COMPLETED, EFFORT_TRY, EFFORT_WORK, EFFORT_SHIRK)
    
    T = get_transition_prob(STATES, EFFICACY, [difficulty_probability, 1-difficulty_probability])
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy(STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR, 
                                  reward_func, reward_func_last, T)
    
    for i_state in range(N_DIFFICULTY_STATES+1):
        
        # find timepoints where it is optimal to work (when task not completed, state=0)
        start_work = np.where( policy_opt[i_state, :] == 0 )[0]
        
        if len(start_work) > 0 :
            start_works[i_prob, i_state] = start_work[0] # first time to start working
            #print( policy_opt[0, :])
            
plt.figure(figsize=(8,6))   
colors = plt.cm.Blues(np.linspace(0.4,0.9,N_DIFFICULTY_STATES+1))         
for i_state in range(N_DIFFICULTY_STATES+1):
    
    plt.plot(probabilities, start_works[:, i_state], label = f'state = {i_state}', color = colors[i_state])
    
plt.xlabel('probability of task being easy')
plt.ylabel('time to start work/ check')
plt.legend()
plt.title(f'efficacy = {EFFICACY}')