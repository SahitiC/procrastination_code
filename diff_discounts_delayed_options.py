import numpy as np
import matplotlib as mpl
mpl.rcParams['font.size'] = 14
mpl.rcParams['lines.linewidth'] = 2
import matplotlib.pyplot as plt
import itertools

#%%
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
                    Q[i_action] = ( discount_factor_reward**(i_timestep-i_iter) ) * reward_func[i_state][i_action][i_timestep] + (
                                    discount_factor_cost**(i_timestep-i_iter) ) * cost_func[i_state][i_action][i_timestep] + (
                                    T[i_state][i_action] @ V_opt[:, i_timestep+1] )
                
                # find optimal action (which gives max q-value)
                V_opt[i_state, i_timestep] = np.max(Q)
                policy_opt[i_state, i_timestep] = np.argmax(Q)
                Q_values[i_state][:, i_timestep] = Q    
            
        V_opt_full.append(V_opt)
        policy_opt_full.append(policy_opt)
        Q_values_full.append(Q_values)
        
    return V_opt_full, policy_opt_full, Q_values_full

# construct separate reward and effort functions
def get_reward_functions(states, actions, reward_1, effort_1, delay_1, 
                                reward_2, effort_2, delay_2, horizon):
    
    # for 2x2 states
    
    reward_func = []
    # for first state
    reward_func.append( np.zeros((len(actions[0]), horizon)) )
    reward_func[0][0, delay_1] = reward_1 #action 1
    reward_func[0][1, delay_2] = reward_2 #action 2
    # for second state
    reward_func.append( np.zeros((len(actions[1]), horizon)) ) 
    reward_func[1][0, delay_1] = reward_1 #action 1
    # for third state
    reward_func.append( np.zeros((len(actions[2]), horizon)) ) 
    reward_func[2][0, delay_2] = reward_2 #action 1
    # for fourth state
    reward_func.append( np.zeros((len(actions[3]), horizon)) ) 
    
    reward_func_last = [0., 0, 0, 0]
    
    cost_func = []
    # for first state
    cost_func.append( np.zeros((len(actions[0]), horizon)) )
    cost_func[0][0, :] = effort_1 #action 1
    cost_func[0][1, :] = effort_2 #action 2
    # for second state
    cost_func.append( np.zeros((len(actions[1]), horizon)) ) 
    cost_func[1][0, :] = effort_1 #action 1
    # for third state
    cost_func.append( np.zeros((len(actions[2]), horizon)) ) 
    cost_func[2][0, :] = effort_2 #action 1
    # for fourth state
    cost_func.append( np.zeros((len(actions[3]), horizon)) ) 
    
    cost_func_last = [0., 0, 0, 0]
    
    return np.array(reward_func), np.array(reward_func_last), np.array(cost_func), np.array(cost_func_last)

# construct transition matrix
def get_transition_prob(states, efficacy):
    
    T = []
    
    # for 2x2 states:
    T.append( np.array([ [1-efficacy, 0, efficacy, 0], 
                         [1-efficacy, efficacy, 0, 0], 
                         [1., 0, 0, 0] ]) ) # transitions for do_1, do_2, dont
    
    T.append( np.array([ [0, 1-efficacy, 0, efficacy], 
                         [0, 1., 0, 0] ]) ) # transitions for do_1, dont
    
    T.append( np.array([ [0, 0, 1-efficacy, efficacy], 
                         [0, 0, 1., 0] ]) ) # transitions for do_2, dont
    
    T.append( np.array([ [0, 0, 0, 1.] ]) ) # transitions for dont
    
    return np.array(T)

#%%
    
# setting up the MDP     

# states of markov chain   
N_OPTIONS = 2 # NO. OF OPTIONS AVAILABLE
STATES = list( map(list, itertools.product([0,1], repeat = N_OPTIONS)) ) # STATES WITH 2 POSSIBLE STATES IN EACH OPTION

# ACTIONS AVAILABLE IN EACH STATE 
ACTIONS = np.full(len(STATES), np.nan, dtype = object)
ACTIONS[0] = ['DO_1', 'DO_2', 'DONT']  
ACTIONS[1] = ['DO_1', 'DONT']
ACTIONS[2] = ['DO_2', 'DONT']
ACTIONS[3] = ['DONT'] 

HORIZON = 10 # DEADLINE
DISCOUNT_FACTOR_REWARD = 0.9 # DISCOUNTING FACTOR FOR REWARDS
DISCOUNT_FACTOR_COST = 0.9 # DISCOUNTING FACTOR FOR COSTS
EFFICACY = 0.7 # SELF-EFFICACY (PROBABILITY OF PROGRESS ON WORKING)

# REWARDS AND CORRESPONDING DELAYS, EFFORTS FOR THE N_OPTIONS
DELAY_1 = 9
REWARD_1 = 8
EFFORT_1 = -1
DELAY_2 = 0
REWARD_2 = 0
EFFORT_2 = 0


#%%

reward_func, reward_func_last, cost_func, cost_func_last = get_reward_functions(STATES, ACTIONS, REWARD_1, EFFORT_1, DELAY_1, REWARD_2, EFFORT_2, DELAY_2, HORIZON)
T = get_transition_prob(STATES, EFFICACY)

V_opt_full, policy_opt_full, Q_values_full = find_optimal_policy_diff_discount_factors(
                                             STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR_REWARD, DISCOUNT_FACTOR_COST, 
                                             reward_func, cost_func, reward_func_last, cost_func_last, T)
