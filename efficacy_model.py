"""
script for mdp for assignment submission: There is an initial state (when assignment is not started),
potential intermediate states, and final state of completion. At each non-completed state, there is a choice
between actions to WORK which has an immediate effort cost and SHIRK which has an immediate reward. 
The final state is absorbing and also has a reward (equivalent to rewards from shirk). The outcome from evaluation 
only comes at the deadline (which can be negative to positive based on state at final timepoint).  
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
def get_reward_functions(states, reward_pass, reward_fail, reward_shirk, reward_completed, effort_work, effort_shirk):
    
    # reward from actions within horizon
    reward_func = [] 
    reward_func = [([effort_work, reward_shirk + effort_shirk]) for i in range( len(states)-1 )] # rewards in non-completed states
    reward_func.append( [reward_completed] ) # reward in completed state
    
    # reward from final evaluation
    reward_func_last =  np.linspace(reward_fail, reward_pass, len(states)) 
    
    return np.array(reward_func), reward_func_last

# construct transition matrix
def get_transition_prob(states, efficacy):
    
    T = []
    
    # for 3 states:
    T.append( [ np.array([1-efficacy, efficacy, 0]), 
                np.array([1, 0, 0]) ] ) # transitions for work, shirk
    T.append( [ np.array([0, 1-efficacy, efficacy]), 
                np.array([0, 1, 0]) ] ) # transitions for work, shirk
    T.append( [ np.array([0, 0, 1]) ] ) # transitions for completed
    
#    # for 2 states:
#    T[0] = [ np.array([1-efficacy, efficacy]), 
#             np.array([1, 0]) ] # transitions for work, shirk
#    T[1] = [ np.array([0, 1]) ] # transitions for completed
    
    return np.array(T)

#%%

# states of markov chain
N_INTERMEDIATE_STATES = 1
STATES = np.arange(2 + N_INTERMEDIATE_STATES) # intermediate + initial and finished states (2)

# actions available in each state 
ACTIONS = np.full(len(STATES), np.nan, dtype = object)
ACTIONS[:-1] = [ ['work', 'shirk'] for i in range( len(STATES)-1 )] # actions for all states but final
ACTIONS[-1] =  ['completed'] # actions for final state

HORIZON = 10 # deadline
DISCOUNT_FACTOR = 0.9 # discounting factor
EFFICACY = 0.6 # self-efficacy (probability of progress on working)

# utilities :
REWARD_PASS = 4.0 
REWARD_FAIL = -4.0
REWARD_SHIRK = 0.5
EFFORT_WORK = -0.4
EFFORT_SHIRK = -0 
REWARD_COMPLETED = REWARD_SHIRK

#%% 
# example policy

reward_func, reward_func_last = get_reward_functions(STATES, REWARD_PASS, REWARD_FAIL, REWARD_SHIRK, 
                                                     REWARD_COMPLETED, EFFORT_WORK, EFFORT_SHIRK)
T = get_transition_prob(STATES, EFFICACY)
V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy(STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR, 
                              reward_func, reward_func_last, T)

# plots of policies and values
plt.figure( figsize = (8, 6) )
for i_state, state in enumerate(STATES):
    
    #plt.figure( figsize = (8, 6) )
    
    plt.plot(V_opt[i_state], label = f'V*{i_state}', marker = i_state+4, linestyle = '--')
    #plt.plot(policy_opt[i_state], label = 'policy*')
    
    for i_action, action in enumerate(ACTIONS[i_state]):
        
        plt.plot(Q_values[i_state][i_action, :], label = r'Q'+action, marker = i_state+4, linestyle = '--')
       
    plt.legend()

#%%
# solving for policies for a range of efficacies

efficacys = np.linspace(0, 1, 50)
rewards_task = np.array([0.5, 1, 2, 4])
# optimal starting point for 4 reward regimes (change manually)
start_works = np.full( (len(efficacys), N_INTERMEDIATE_STATES+1, len(rewards_task)), np.nan )


for i_reward_task, reward_task in enumerate(rewards_task):

    for i_efficacy, efficacy in enumerate(efficacys):
        
        reward_pass = reward_task
        reward_fail = -1 * reward_task
        
        reward_func, reward_func_last = get_reward_functions(STATES, reward_pass, reward_fail, REWARD_SHIRK, 
                                                             REWARD_COMPLETED, EFFORT_WORK, EFFORT_SHIRK)
        T = get_transition_prob(STATES, efficacy)
        V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy(STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR, 
                                      reward_func, reward_func_last, T)
        
        for i_state in range(N_INTERMEDIATE_STATES+1):
            # find timepoints where it is optimal to work (when task not completed, state=0
            start_work = np.where( policy_opt[i_state, :] == 0 )[0]
            
            if len(start_work) > 0 :
                start_works[i_efficacy, i_state, i_reward_task] = start_work[0] # first time to start working 
                
for i_state in range(N_INTERMEDIATE_STATES+1):
    
    plt.figure(figsize=(8,6))
    for i_reward_task, reward_task in enumerate(rewards_task):
        plt.plot(efficacys, start_works[:, i_state, i_reward_task], label = f'{reward_task}:0.5')
    plt.xlabel('efficacy')
    plt.ylabel('time to start work')
    plt.legend()
    plt.title(f'effort = {EFFORT_WORK}, state = {i_state}')

#%%
# solving for policies for a range of efforts

efforts = np.linspace(-8, 1, 50)
rewards_task = np.array([0.5, 1, 2, 4])
# optimal starting point for 4 reward regimes (change manually)
start_works = np.full( (len(efforts), N_INTERMEDIATE_STATES+1, 4), np.nan ) 

for i_reward_task, reward_task in enumerate(rewards_task):
    
    for i_effort, effort_work in enumerate(efforts):
        
        reward_pass = reward_task
        reward_fail = -1 * reward_task
    
        reward_func, reward_func_last = get_reward_functions(STATES, reward_pass, reward_fail, REWARD_SHIRK,
                                                             REWARD_COMPLETED, effort_work, EFFORT_SHIRK)
        T = get_transition_prob(STATES, EFFICACY)
        V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy(STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR, 
                                      reward_func, reward_func_last, T)
        
        for i_state in range(N_INTERMEDIATE_STATES+1):
            
            # find timepoints where it is optimal to work (when task not completed, state=0)
            start_work = np.where( policy_opt[i_state, :] == 0 )[0]
            
            if len(start_work) > 0 :
                start_works[i_effort, i_state, i_reward_task] = start_work[0] # first time to start working
                #print( policy_opt[0, :])
            
for i_state in range(N_INTERMEDIATE_STATES+1):
    
    plt.figure(figsize=(8,6))
    
    for i_reward_task, reward_task in enumerate(rewards_task):
         plt.plot(efforts, start_works[:, i_state, i_reward_task], label = f'{reward_task}:0.5')
    plt.xlabel('effort to work')
    plt.ylabel('time to start work')
    plt.legend()
    plt.title(f'efficacy = {EFFICACY}, state = {i_state}')
    plt.show()
    
    
#%%
# forward runs
    
efficacys = np.linspace(0, 1, 10) # vary efficacy 
policy_always_work = np.full(np.shape(policy_opt), 0) # always work policy
V_always_work = np.full(np.shape(V_opt), 0.0)

# arrays to store no. of runs where task was finished 
count_opt = np.full( (len(efficacys), 1), 0) 
count_always_work = np.full( (len(efficacys), 1), 0) # policy of always work
N_runs = 5000 # no. of runs for each parameter set

for i_efficacy, efficacy in enumerate(efficacys):
    
    # get optimal policy for current parameter set
    reward_func, reward_func_last = get_reward_functions(STATES, REWARD_PASS, REWARD_FAIL, REWARD_SHIRK, 
                                                         REWARD_COMPLETED, EFFORT_WORK, EFFORT_SHIRK)
    T = get_transition_prob(STATES, efficacy)
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy(STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR, 
                                  reward_func, reward_func_last, T)
    
    # run forward (N_runs no. of times), count number of times task is finished for each policy
    initial_state = 0
    for i in range(N_runs):
         
        s, a, v = mdp_algms.forward_runs(policy_opt, V_opt, initial_state, HORIZON, STATES, T)
        if s[-1] == len(STATES)-1: count_opt[i_efficacy,0] +=1
        
        s, a, v = mdp_algms.forward_runs(policy_always_work, V_opt, initial_state, HORIZON, STATES, T)
        if s[-1] == len(STATES)-1: count_always_work[i_efficacy,0] +=1

plt.figure(figsize=(8,6))
plt.bar( efficacys, count_always_work[:, 0]/N_runs, alpha = 0.5, width=0.1, color='tab:blue', label = 'always work')
plt.bar( efficacys, count_opt[:, 0]/N_runs, alpha = 1, width=0.1, color='tab:blue', label = 'optimal policy')
plt.legend()
plt.xlabel('efficacy')
plt.ylabel('Proportion of finished runs')