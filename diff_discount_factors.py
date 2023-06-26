import numpy as np
import matplotlib as mpl
mpl.rcParams['font.size'] = 14
mpl.rcParams['lines.linewidth'] = 2
import matplotlib.pyplot as plt
import mdp_algms

#%%

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
DISCOUNT_FACTOR_COST = 0.9 # discounting factor for costs
DISCOUNT_FACTOR_COMMON = 0.9 # common discount factor for both 
EFFICACY = 0.7 # self-efficacy (probability of progress on working)

# utilities :
REWARD_DO = 1.5
EFFORT_DO = -1.0
REWARD_COMPLETED = 0.0
COST_COMPLETED = -0.0

#%%
# solve for different discount case
reward_func, cost_func, reward_func_last, cost_func_last = get_reward_functions( STATES, REWARD_DO, 
                                                                                 EFFORT_DO, REWARD_COMPLETED, COST_COMPLETED )
T = get_transition_prob(STATES, EFFICACY)

V_opt_full, policy_opt_full, Q_values_full =  mdp_algms.find_optimal_policy_diff_discount_factors( STATES, ACTIONS, 
                                              HORIZON, DISCOUNT_FACTOR_REWARD, DISCOUNT_FACTOR_COST, 
                                              reward_func, cost_func, reward_func_last, cost_func_last, T )

#%%
### THIS WORKS ONLY FOR A SMALL MDP WHERE ONLY ONE STATE HAS TWO CHOICES OF ACTIONS WITH A SMALL HORIZON ###

# solve for different discount case using brute force
reward_func, cost_func, reward_func_last, cost_func_last = get_reward_functions( STATES, REWARD_DO, 
                                                                                 EFFORT_DO, REWARD_COMPLETED, COST_COMPLETED )
T = get_transition_prob(STATES, EFFICACY)

V_opt_bf, policy_opt_bf =  mdp_algms.find_optimal_policy_diff_discount_factors_brute_force( 
                           STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR_REWARD, DISCOUNT_FACTOR_COST, 
                           reward_func, cost_func, reward_func_last, cost_func_last, T )

#%%
# solve for common discount case
reward_func, cost_func, reward_func_last, cost_func_last = get_reward_functions( STATES, REWARD_DO, 
                                                                                 EFFORT_DO, REWARD_COMPLETED, COST_COMPLETED )
T = get_transition_prob(STATES, EFFICACY)

V_opt, policy_opt, Q_values =  mdp_algms.find_optimal_policy( STATES, ACTIONS, 
                                              HORIZON, DISCOUNT_FACTOR_COMMON,  
                                              reward_func, reward_func_last, T )

# solve for common discount case
reward_func, reward_func_last = get_reward_functions_common( STATES, REWARD_DO, EFFORT_DO, 
                                                             REWARD_COMPLETED, COST_COMPLETED )
T = get_transition_prob(STATES, EFFICACY)

V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy( STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR_COMMON, 
                                                             reward_func, reward_func_last, T )
    

#%%
# changing relative discounts, plot
discounts_reward = np.array([0.8, 0.65, 0.5]) 

colors = plt.cm.Blues(np.linspace(0.3,0.9,4)) 
plt.figure(figsize = (8,6))
for i_d_r, discount_factor_reward in enumerate(discounts_reward):
    
    reward_func, cost_func, reward_func_last, cost_func_last = get_reward_functions( 
    STATES, REWARD_DO, EFFORT_DO, REWARD_COMPLETED, COST_COMPLETED )
     
    T = get_transition_prob(STATES, EFFICACY)
    
    V_opt_full, policy_opt_full, Q_values_full = mdp_algms.find_optimal_policy_diff_discount_factors( STATES, ACTIONS, 
                                       HORIZON, discount_factor_reward, DISCOUNT_FACTOR_COST, 
                                       reward_func, cost_func, reward_func_last, cost_func_last, T )
    
    v_opt_real = np.array([V_opt_full[HORIZON-1-i][:][:, i] for i in range(HORIZON)]) # the real optimal returns for each timestep and state
    q_val_real = np.array([Q_values_full[HORIZON-1-i][0][:, i] for i in range(HORIZON)]) # real returns for both options in state=0
    
    work = np.full(HORIZON, np.nan)
    for i in range(HORIZON):
        
        #policy_opt_full[HORIZON-1-i][0, :-1]-policy_opt_full[HORIZON-1-i][0, 1:] == 1 , w[0]+1
        w = np.where(policy_opt_full[HORIZON-1-i][0, 1:]-policy_opt_full[HORIZON-1-i][0, :-1] == 1)[0]
        if w.size > 0 : work[i] = w[0] 
    
    plt.plot(work, label = f"$\gamma_c$ = {DISCOUNT_FACTOR_COST}, $\gamma_r$ = {discount_factor_reward}", color = colors[i_d_r], linestyle = '--')
 
plt.hlines( np.where(policy_opt[0, :] == 0)[0][0], 0, 9, label = f"$\gamma_r$ = $\gamma_c$ = {DISCOUNT_FACTOR_COMMON}", color = 'tab:red')
plt.xlabel('timesteps')
plt.ylabel('optimal time to stop working')
plt.legend(loc = 'center right')