import numpy as np
import matplotlib as mpl
mpl.rcParams['font.size'] = 14
mpl.rcParams['lines.linewidth'] = 2
import matplotlib.pyplot as plt
import itertools
import mdp_algms

#%%
# generate return from policy

def policy_eval_diff_discount_factors(states, i_timestep, reward_func_last, effort_func_last, reward_func, effort_func,
                                      T, discount_factor_reward, discount_factor_effort, policy_all):

    V_r = np.full( (len(states), i_timestep+2), np.nan)
    V_c = np.full( (len(states), i_timestep+2), np.nan)     
    
    V_r[:, -1] = reward_func_last
    V_c[:, -1] = effort_func_last
    
    for i_iter in range(i_timestep, -1, -1): 
        
        for i_state, state in enumerate(states):
            
            V_r[i_state, i_iter] = reward_func[i_state][policy_all[i_state, i_iter]] + discount_factor_reward * (
                                   T[i_state][policy_all[i_state, i_iter]] @ V_r[states, i_iter+1])
            
            V_c[i_state, i_iter] = effort_func[i_state][policy_all[i_state, i_iter]] + discount_factor_effort * (
                           T[i_state][policy_all[i_state, i_iter]] @ V_c[states, i_iter+1])
    
    return V_r, V_c

# construct reward functions
def get_reward_functions(states, reward_do, effort_do):
    
    # reward from actions within horizon
    reward_func = np.full(len(states), np.nan, dtype = object)
    effort_func = np.full(len(states), np.nan, dtype = object)
    reward_func[:-1] = [ [reward_do, 0.0] for i in range( len(states)-1 )]
    effort_func[:-1] = [ [effort_do, 0.0] for i in range( len(states)-1 )]
    reward_func[-1] = [0.0]
    effort_func[-1] = [0.0]
    # reward from final evaluation
    reward_func_last =  [0.0, 4.0]
    effort_func_last =  [0.0, 0.0]
    
    return np.array(reward_func), np.array(effort_func), np.array(reward_func_last), np.array(effort_func_last)

# construct transition matrix
def get_transition_prob(states, efficacy):
    
    T = np.full(len(states), np.nan, dtype = object)
    
#    # for 3 states:
#    T[0] = [ np.array([1-efficacy, efficacy, 0]), 
#             np.array([1, 0, 0]) ] # transitions for work, shirk
#    T[1] = [ np.array([0, 1-efficacy, efficacy]), 
#             np.array([0, 1, 0]) ] # transitions for work, shirk
#    T[2] = [ np.array([0, 0, 1]) ] # transitions for completed
    
    # for 2 states:
    T[0] = [ np.array([1-efficacy, efficacy]), 
             np.array([1, 0]) ] # transitions for work, shirk
    T[1] = [ np.array([0, 1]) ] # transitions for completed
    
    return T

#%%
# setting up the MDP     

# states of markov chain   
N_intermediate_states = 0
states = np.arange(2 + N_intermediate_states) # intermediate + initial and finished states (2)

# actions available in each state 
actions = np.full(len(states), np.nan, dtype = object)
actions[:-1] = [ ['do', 'dont'] for i in range( len(states)-1 )] # actions for all states but final
actions[-1] =  ['done'] # actions for final state

horizon = 10 # deadline
discount_factor_reward = 0.9 # discounting factor
discount_factor_effort = 0.8 # discounting factor
efficacy = 0.7 # self-efficacy (probability of progress on working)

# utilities :
reward_do = 0.0 
effort_do = -2.0

#%%
# my algorithm!!! for finding optimal policy with different discount factors for positive and negative rewards

V_opt_full = []
policy_opt_full = []
Q_values_full = []

reward_func, effort_func, reward_func_last, effort_func_last = get_reward_functions(states, reward_do, effort_do)
T = get_transition_prob(states, efficacy)

for i_iter in range(horizon-1, -1, -1):
    
    V_opt = np.full( (len(states), horizon+1), np.nan)
    policy_opt = np.full( (len(states), horizon), np.nan)
    Q_values = np.full( len(states), np.nan, dtype = object)

    for i_state, state in enumerate(states):
        
        # V_opt for last time-step 
        V_opt[i_state, -1] = ( discount_factor_reward**(horizon-i_iter) ) * reward_func_last[i_state] + (
                               discount_factor_effort**(horizon-i_iter) ) * effort_func_last[i_state]
        # arrays to store Q-values for each action in each state
        Q_values[i_state] = np.full( (len(actions[i_state]), horizon), np.nan)
    
    # backward induction to derive optimal policy  
    for i_timestep in range(horizon-1, i_iter-1, -1):
        
        for i_state, state in enumerate(states):
            
            Q = np.full( len(actions[i_state]), np.nan) 
            
            for i_action, action in enumerate(actions[i_state]):
                
                # q-value for each action (bellman equation)
                Q[i_action] = ( discount_factor_reward**(i_timestep-i_iter) ) * reward_func[i_state][i_action] + (
                                discount_factor_effort**(i_timestep-i_iter) ) * effort_func[i_state][i_action] + (
                                T[i_state][i_action] @ V_opt[states, i_timestep+1] )
            
            # find optimal action (which gives max q-value)
            V_opt[i_state, i_timestep] = np.max(Q)
            policy_opt[i_state, i_timestep] = np.argmax(Q)
            Q_values[i_state][:, i_timestep] = Q    
        
    V_opt_full.append(V_opt)
    policy_opt_full.append(policy_opt)
    Q_values_full.append(Q_values)
        
#%%
reward_func, effort_func, reward_func_last, effort_func_last = get_reward_functions(states, reward_do, effort_do)
T = get_transition_prob(states, efficacy)

V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy(states, actions, horizon, discount_factor_reward, 
                              effort_func, reward_func_last, T)

#%%
# brute force algorithm for finding optimal policy at each time step with different discount factors for rewards and efforts
# (this is particular to a simple mdp structure with two states and only one of them has a choice in actions)

V_opt_bf = [] # to store optimal values for all time steps
policy_opt_bf = [] # to store collection of all optimal policies from each timestep onwards

# reward, effort and transition functions
reward_func, effort_func, reward_func_last, effort_func_last = get_reward_functions(states, reward_do, effort_do)
T = get_transition_prob(states, efficacy)

# optimal values for rest of timesteps
for i_timestep in range(horizon):
    
    # find optimal v and policy for i_timestep
    v_opt_bf = [-np.inf, -np.inf]
    pol_opt_bf = []
    
    # generate all combinations of policies (for state=0 with 2 possible actions) for i_timestep+1's
    policy_list = list( map(list, itertools.product([0,1], repeat = i_timestep+1)) )
    
    # evaluate each policy 
    for i_policy, policy in enumerate(policy_list):
        
        # policy for state = 1 is all 0's, append this to policy for state=0 for all timesteps
        policy_all = np.vstack( (np.array(policy), np.zeros(len(policy), dtype=int) ) )
        
        # positive and negative returns for all states
        v_r, v_c = policy_eval_diff_discount_factors(states, i_timestep, reward_func_last, effort_func_last, 
                   reward_func, effort_func, T, discount_factor_reward, discount_factor_effort, policy_all)
        
        # find opt policy for state = 0, no need for state=1 (as only one action available)
        if v_r[0,0] + v_c[0,0] > v_opt_bf[0]:
            
            v_opt_bf[0] = v_r[0,0] + v_c[0,0]
            v_opt_bf[1] = v_r[1,0] + v_c[1,0]
            pol_opt_bf = policy_all
              
    V_opt_bf.append( np.array(v_opt_bf) )
    policy_opt_bf.append( np.array([pol_opt_bf]) )
    
    
    
    
    