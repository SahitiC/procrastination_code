import numpy as np
import itertools

def find_optimal_policy(states, actions, horizon, discount_factor, 
                        reward_func, reward_func_last, T):
    '''
    function to find optimal policy for an MDP with finite horizon, discrete states,
    deterministic rewards and actions using dynamic programming
    
    inputs: states, actions available in each state, rewards from actions and final rewards, 
    transition probabilities for each action in a state, discount factor, length of horizon
    
    outputs: optimal values, optimal policy and action q-values for each timestep and state 
        
    '''

    V_opt = np.full( (len(states), horizon+1), np.nan)
    policy_opt = np.full( (len(states), horizon), np.nan)
    Q_values = np.full( len(states), np.nan, dtype = object)
    
    for i_state, state in enumerate(states):
        
        # V_opt for last time-step 
        V_opt[i_state, -1] = reward_func_last[i_state]
        # arrays to store Q-values for each action in each state
        Q_values[i_state] = np.full( (len(actions[i_state]), horizon), np.nan)
    
    # backward induction to derive optimal policy  
    for i_timestep in range(horizon-1, -1, -1):
        
        for i_state, state in enumerate(states):
            
            Q = np.full( len(actions[i_state]), np.nan) 
            
            for i_action, action in enumerate(actions[i_state]):
                
                # q-value for each action (bellman equation)
                Q[i_action] = reward_func[i_state][i_action] + discount_factor * (
                              T[i_state][i_action] @ V_opt[states, i_timestep+1] )
            
            # find optimal action (which gives max q-value)
            V_opt[i_state, i_timestep] = np.max(Q)
            policy_opt[i_state, i_timestep] = np.argmax(Q)
            Q_values[i_state][:, i_timestep] = Q
            
    return V_opt, policy_opt, Q_values


def forward_runs( policy, V, initial_state, horizon, states, T):
    
    '''
    function to simulate actions taken and states reached forward in time given 
    a policy and initial state in an mdp
    
    inputs: policy, corresponding values, initial state, horizon, states available, T
    
    outputs: actions taken according to policy, corresponding values and states reached based on T
    '''
    
    # arrays to store states, actions taken and values of actions in time
    states_forward = np.full( horizon+1, 100 )
    actions_forward = np.full( horizon, 100 )
    values_forward = np.full( horizon, np.nan )
    
    states_forward[0] = initial_state # initial state
    
    for i_timestep in range(horizon):
        
        # action at a state and timestep as given by policy
        actions_forward[i_timestep] = policy[ states_forward[i_timestep], i_timestep ]
        # corresponding value
        values_forward[i_timestep] = V[ states_forward[i_timestep], i_timestep ]
        # next state given by transition probabilities
        states_forward[i_timestep+1] = np.random.choice ( len(states), 
                                       p = T[ states_forward[i_timestep] ][ actions_forward[i_timestep] ] )
    
    return states_forward, actions_forward, values_forward

# my algorithm!!! for finding optimal policy with different discount factors for positive and negative rewards
def find_optimal_policy_diff_discount_factors(states, actions, horizon, discount_factor_reward, discount_factor_cost, 
                                             reward_func, cost_func, reward_func_last, cost_func_last, T):
    
    '''
    algorithm for finding optimal policy with different exponential discount factors for 
    rewards and efforts, for a finite horizon and discrete states/ actions; 
    since the optimal policy can shift in time, it is found starting at every timestep
    
    inputs: states, actions available in each state, rewards from actions and final rewards, 
    transition probabilities for each action in a state, discount factor, length of horizon
    outputs: optimal values, optimal policy and action q-values for each timestep and state 

    '''

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
