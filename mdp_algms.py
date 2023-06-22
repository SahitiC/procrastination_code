import numpy as np

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
    function to simulate actions taken and states reached forward time given a policy 
    and initial state in an mdp
    
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
