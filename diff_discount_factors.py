import numpy as np
import matplotlib as mpl
mpl.rcParams['font.size'] = 14
mpl.rcParams['lines.linewidth'] = 2
import matplotlib.pyplot as plt
import itertools
import mdp_algms

#%%

# my algorithm!!! for finding optimal policy with different discount factors for positive and negative rewards
def find_optimal_policy_diff_discount_factors(states, actions, horizon, discount_factor_reward, discount_factor_cost, 
                                             reward_func, cost_func, reward_func_last, cost_func_last, T):

    V_opt_full = []
    policy_opt_full = []
    Q_values_full = []

    # solve for optimal policy at every time step
    for i_iter in range(horizon-1, -1, -1):
        
        V_opt = np.zeros( (len(states), horizon+1) )
        policy_opt = np.full( (len(states), horizon), np.nan )
        Q_values = np.zeros( len(states), dtype = object)
    
        for i_state, state in enumerate(states):
            
            # V_opt for last time-step 
            V_opt[i_state, -1] = ( discount_factor_reward**(horizon-i_iter) ) * reward_func_last[i_state] + (
                                   discount_factor_cost**(horizon-i_iter) ) * cost_func_last[i_state]
            # arrays to store Q-values for each action in each state
            Q_values[i_state] = np.full( (len(actions[i_state]), horizon), np.nan)
        
        # backward induction to derive optimal policy starting from timestep i_iter 
        for i_timestep in range(horizon-1, i_iter-1, -1):
            
            for i_state, state in enumerate(states):
                
                Q = np.full( len(actions[i_state]), np.nan) 
                
                for i_action, action in enumerate(actions[i_state]):
                    
                    # q-value for each action (bellman equation)
                    Q[i_action] = ( discount_factor_reward**(i_timestep-i_iter) ) * reward_func[i_state][i_action] + (
                                    discount_factor_cost**(i_timestep-i_iter) ) * cost_func[i_state][i_action] + (
                                    T[i_state][i_action] @ V_opt[states, i_timestep+1] )
                
                # find optimal action (which gives max q-value)
                V_opt[i_state, i_timestep] = np.max(Q)
                policy_opt[i_state, i_timestep] = np.argmax(Q)
                Q_values[i_state][:, i_timestep] = Q    
            
        V_opt_full.append(V_opt)
        policy_opt_full.append(policy_opt)
        Q_values_full.append(Q_values)
        
    return V_opt_full, policy_opt_full, Q_values_full


def find_optimal_policy_diff_discount_factors_brute_force(states, actions, horizon, discount_factor_reward, discount_factor_cost, 
                                                          reward_func, cost_func, reward_func_last, cost_func_last, T):
    
    '''
    brute force algorithm for finding optimal policy at each time step with different discount factors for rewards and efforts
    (this is particular to a simple mdp with two states and only one of them has a choice in actions)
    '''

    V_opt_bf = [] # to store optimal values for all time steps
    policy_opt_bf = [] # to store collection of all optimal policies from each timestep onwards
    
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
            v_r, v_c = policy_eval_diff_discount_factors(states, i_timestep, reward_func_last, cost_func_last, 
                       reward_func, cost_func, T, discount_factor_reward, discount_factor_cost, policy_all)
            
            # find opt policy for state = 0, no need for state=1 (as only one action available)
            if v_r[0,0] + v_c[0,0] > v_opt_bf[0]:
                
                v_opt_bf[0] = v_r[0,0] + v_c[0,0]
                v_opt_bf[1] = v_r[1,0] + v_c[1,0]
                pol_opt_bf = policy_all
                  
        V_opt_bf.append( np.array(v_opt_bf) )
        policy_opt_bf.append( np.array([pol_opt_bf]) )
        
    return V_opt_bf, policy_opt_bf
    

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


# construct reward functions separately for rewards and costs
def get_reward_functions(states, reward_do, effort_do, reward_completed, cost_completed):
    
    # reward from actions within horizon
    reward_func = np.full(len(states), np.nan, dtype = object)
    cost_func = np.full(len(states), np.nan, dtype = object)
    # for last but one states
    reward_func[:-1] = [ [reward_do, 0.0] for i in range( len(states)-1 )]
    cost_func[:-1] = [ [effort_do, 0.0] for i in range( len(states)-1 )]
    # for last state
    reward_func[-1] = [0.0]
    cost_func[-1] = [0.0]
    # reward from final evaluation
    reward_func_last =  [0.0, reward_completed]
    cost_func_last =  [0.0, cost_completed]
    
    return np.array(reward_func), np.array(cost_func), np.array(reward_func_last), np.array(cost_func_last)

# construct common reward functions
def get_reward_functions_common(states, reward_do, effort_do, reward_completed, cost_completed):
    
    # reward from actions within horizon
    reward_func = np.full(len(states), np.nan, dtype = object)
    # for last but one states
    reward_func[:-1] = [ [reward_do+effort_do, 0.0] for i in range( len(states)-1 )]
    # for last state
    reward_func[-1] = [0.0]
    # reward from final evaluation
    reward_func_last =  [0.0, reward_completed+cost_completed]
    
    return np.array(reward_func), np.array(reward_func_last)

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
N_INTERMEDIATE_STATES = 0
STATES = np.arange(2 + N_INTERMEDIATE_STATES) # intermediate + initial and finished states (2)

# actions available in each state 
ACTIONS = np.full(len(STATES), np.nan, dtype = object)
ACTIONS[:-1] = [ ['do', 'dont'] for i in range( len(STATES)-1 )] # actions for all states but final
ACTIONS[-1] =  ['done'] # actions for final state

HORIZON = 10 # deadline
DISCOUNT_FACTOR_REWARD = 0.9 # discounting factor for rewards
DISCOUNT_FACTOR_COST = 0.8 # discounting factor for costs
DISCOUNT_FACTOR_COMMON = 0.9 # common discount factor for both 
EFFICACY = 0.7 # self-efficacy (probability of progress on working)

# utilities :
REWARD_DO = 0.0 
EFFORT_DO = -2.0
REWARD_COMPLETED = 4.0
COST_COMPLETED = -0.0

#%%
# solve for different discount case
reward_func, cost_func, reward_func_last, cost_func_last = get_reward_functions( STATES, REWARD_DO, 
                                                                                 EFFORT_DO, REWARD_COMPLETED, COST_COMPLETED )
T = get_transition_prob(STATES, EFFICACY)

V_opt_full, policy_opt_full, Q_values_full =  find_optimal_policy_diff_discount_factors( STATES, ACTIONS, 
                                              HORIZON, DISCOUNT_FACTOR_REWARD, DISCOUNT_FACTOR_COST, 
                                              reward_func, cost_func, reward_func_last, cost_func_last, T )

#%%
### THIS WORKS ONLY FOR A SMALL MDP WHERE ONLY ONE STATE HAS TWO CHOICES OF ACTIONS WITH A SMALL HORIZON ###

# solve for different discount case using brute force
reward_func, cost_func, reward_func_last, cost_func_last = get_reward_functions( STATES, REWARD_DO, 
                                                                                 EFFORT_DO, REWARD_COMPLETED, COST_COMPLETED )
T = get_transition_prob(STATES, EFFICACY)

V_opt_bf, policy_opt_bf =  find_optimal_policy_diff_discount_factors_brute_force( 
                           STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR_REWARD, DISCOUNT_FACTOR_COST, 
                           reward_func, cost_func, reward_func_last, cost_func_last, T )

#%%
# solve for common discount case
reward_func, reward_func_last = get_reward_functions_common( STATES, REWARD_DO, EFFORT_DO, 
                                                             REWARD_COMPLETED, COST_COMPLETED )
T = get_transition_prob(STATES, EFFICACY)

V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy( STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR_COMMON, 
                                                             reward_func, reward_func_last, T )
    
#%%
# plots

v_opt_real = np.array([V_opt_full[HORIZON-1-i][:][:, i] for i in range(HORIZON)]) # the real optimal returns for each timestep and state
q_val_real = np.array([Q_values_full[HORIZON-1-i][0][:, i] for i in range(HORIZON)]) # real returns for both options in state=0

work = [np.where( policy_opt_full[HORIZON-1-i][0, :] == 0 )[0][0] for i in range(HORIZON)] # planned times for working

plt.figure(figsize=(8,6))
plt.plot(work, label = f"$\gamma_c$ = {DISCOUNT_FACTOR_COST}, $\gamma_r$ = {DISCOUNT_FACTOR_REWARD}")
plt.hlines( np.where(policy_opt[0, :] == 0)[0][0], 0, 10, label = f"$\gamma$")
plt.xlabel('timesteps')
plt.ylabel('optimal time to start working')
plt.legend()