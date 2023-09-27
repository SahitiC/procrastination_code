import numpy as np
import matplotlib as mpl
mpl.rcParams['font.size'] = 16
mpl.rcParams['lines.linewidth'] = 2
import matplotlib.pyplot as plt
import mdp_algms
import seaborn as sns
plt.rcParams['text.usetex'] = True

#%%

# construct reward functions separately for rewards and costs
def get_reward_functions(states, reward_do, effort_do, reward_completed, 
                         cost_completed):
    
    reward_func = [] 
    cost_func = []
    # reward for actions (dependis on current state and next state)
    reward_func.append( [ np.array([0, reward_do]), 
                          np.array([0, 0]) ] ) # rewards for do and don't
    reward_func.append( [ np.array([0, 0]) ] ) # rewards for completed
    
    # reward from final evaluation for the two states
    reward_func_last =  np.array( [0, reward_completed] )
    
    # effort for actions (dependis on current state and next state)
    cost_func.append( [ np.array([effort_do, effort_do]), 
                          np.array([0, 0]) ] ) # rewards for do and don't
    cost_func.append( [ np.array([0, 0]) ] ) # rewards for completed
    
    # reward from final evaluation for the two states
    cost_func_last =  np.array( [cost_completed, 0] )
    
    return reward_func, cost_func, reward_func_last, cost_func_last

# construct common reward functions
def get_reward_functions_common(states, reward_do, effort_do, reward_completed, 
                                cost_completed):
    
    reward_func = [] 
    # reward for actions (dependis on current state and next state)
    reward_func.append( [ np.array([effort_do, effort_do+reward_do]), 
                          np.array([0, 0]) ] ) # rewards for do and don't
    reward_func.append( [ np.array([0, 0]) ] ) # rewards for completed
    
    # reward from final evaluation for the two states
    reward_func_last =  np.array( [cost_completed, reward_completed] )
    
    return reward_func, reward_func_last

# construct transition matrix
def get_transition_prob(states, efficacy):
    
    T = np.full(len(states), np.nan, dtype = object)
    
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
DISCOUNT_FACTOR_COST = 0.7 # discounting factor for costs
DISCOUNT_FACTOR_COMMON = 0.9 # common discount factor for both 
EFFICACY = 0.6 # self-efficacy (probability of progress on working)

# utilities :
REWARD_DO = 2.0
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

effective_policy =  np.array([policy_opt_full[HORIZON-1-i][0][i] for i in range(HORIZON)]) # actual policy followed by agent


#%%
# policy evaluation to get positive and negative values associated with the optimal policy, seprately
V_r = []
V_c = []
for i_timestep in range(HORIZON-1, -1, -1):
    
    print(i_timestep)
    
    v_r, v_c = mdp_algms.policy_eval_diff_discount_factors(STATES, i_timestep, HORIZON, reward_func_last, cost_func_last, reward_func, cost_func,
                                          T, DISCOUNT_FACTOR_REWARD, DISCOUNT_FACTOR_COST, policy_opt_full[i_timestep].astype(int) )

    V_r.append(v_r[0,HORIZON-i_timestep-1])
    V_c.append(v_c[0,HORIZON-i_timestep-1])
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
# heat map of full policy in state = 0

policy_init_state = [ policy_opt_full[i][0] for i in range(HORIZON) ]
policy_init_state = np.array( policy_init_state )
f, ax = plt.subplots(figsize=(5, 4), dpi=100)
cmap = sns.color_palette('hls', 2)
sns.heatmap(policy_init_state, linewidths=.5, cmap=cmap, cbar=True)
ax.set_xlabel('timestep')
ax.set_ylabel('horizon')
ax.tick_params()
colorbar = ax.collections[0].colorbar
colorbar.set_ticks([0.25, 0.75])
colorbar.set_ticklabels(['WORK', 'SHIRK'])  

#%%
# changing relative discounts, plot
discounts_cost = np.array([0.8, 0.7, 0.6, 0.5]) 

colors = plt.cm.Blues(np.linspace(0.3,0.9,4)) 
fig1, axs1 = plt.subplots(figsize = (8,6))
fig2, axs2 = plt.subplots(figsize = (8,6))

for i_d_r, discount_factor_cost in enumerate(discounts_cost):
    
    reward_func, cost_func, reward_func_last, cost_func_last = get_reward_functions( 
    STATES, REWARD_DO, EFFORT_DO, REWARD_COMPLETED, COST_COMPLETED )
     
    T = get_transition_prob(STATES, EFFICACY)
    
    V_opt_full, policy_opt_full, Q_values_full = mdp_algms.find_optimal_policy_diff_discount_factors( STATES, ACTIONS, 
                                       HORIZON, DISCOUNT_FACTOR_REWARD, discount_factor_cost, 
                                       reward_func, cost_func, reward_func_last, cost_func_last, T )
    
    v_opt_real = np.array([V_opt_full[HORIZON-1-i][:][:, i] for i in range(HORIZON)]) # the real optimal returns for each timestep and state
    q_val_real = np.array([Q_values_full[HORIZON-1-i][0][:, i] for i in range(HORIZON)]) # real returns for both options in state=0
    effective_policy =  np.array([policy_opt_full[HORIZON-1-i][0][i] for i in range(HORIZON)]) # actual policy followed by agent
    
    work = np.full(HORIZON, np.nan)
    for i in range(HORIZON):
        
        #policy_opt_full[HORIZON-1-i][0, :-1]-policy_opt_full[HORIZON-1-i][0, 1:] == 1 , w[0]+1
        w = np.where(policy_opt_full[HORIZON-1-i][0, :-1]-policy_opt_full[HORIZON-1-i][0, 1:] == 1)[0]
        if w.size > 0 : work[i] = w[0] + 1
    
    axs1.plot(work, label = f"$\gamma_c$ = {discount_factor_cost}, $\gamma_r$ = {DISCOUNT_FACTOR_REWARD}", color = colors[i_d_r], linestyle = '--')
    axs2.plot(effective_policy, label = f"$\gamma_c$ = {discount_factor_cost}, $\gamma_r$ = {DISCOUNT_FACTOR_REWARD}", color = colors[i_d_r], linestyle = '--')

axs1.hlines( np.where(policy_opt[0, :] == 0)[0][0], 0, 9, label = f"$\gamma_r$ = $\gamma_c$ = {DISCOUNT_FACTOR_COMMON}", color = 'tab:red')
axs1.set_xlabel('timesteps')
axs1.set_ylabel('optimal time to start working')
axs1.legend(loc = 'center right')

axs2.plot(policy_opt[0,:], label = f"$\gamma_r$ = $\gamma_c$ = {DISCOUNT_FACTOR_COMMON}", color = 'tab:red', linestyle = '--')
axs2.set_xlabel('timesteps')
axs2.set_ylabel('effective policy')
axs2.set_yticks([0, 1])
axs2.set_yticklabels(['DO', 'DON\'T'])
axs2.legend(loc = 'center left')

#%%
# avg finishing times and rates for same discount, diff, precommit 

N_runs = 1000
initial_state = 0

# diff discounts: defecting
finishing_times_diff_discount = []
finishing_rates_diff_discount = []
V_opt = np.zeros((len(STATES), HORIZON))

reward_func, cost_func, reward_func_last, cost_func_last = get_reward_functions( STATES, REWARD_DO, 
                                                                                 EFFORT_DO, REWARD_COMPLETED, COST_COMPLETED )
T = get_transition_prob(STATES, EFFICACY)

V_opt_full, policy_opt_full, Q_values_full =  mdp_algms.find_optimal_policy_diff_discount_factors( STATES, ACTIONS, 
                                              HORIZON, DISCOUNT_FACTOR_REWARD, DISCOUNT_FACTOR_COST, 
                                              reward_func, cost_func, reward_func_last, cost_func_last, T )
   
effective_policy = np.zeros((len(STATES), HORIZON))+10
effective_policy[0,:] =  np.array([policy_opt_full[HORIZON-1-i][0][i] for i in range(HORIZON)]) # actual policy followed by agent
effective_policy[1,:] =  np.array([policy_opt_full[HORIZON-1-i][1][i] for i in range(HORIZON)]) # actual policy followed by agent
   
for i in range(N_runs):
     
    s, a, v = mdp_algms.forward_runs(effective_policy, V_opt, initial_state, HORIZON, STATES, T)
    
    if 1 in s: 
        finishing_times_diff_discount.append(np.where(s==1)[0][0])
        finishing_rates_diff_discount.append(1)
    else:
        finishing_rates_diff_discount.append(0)
        
        
# same discounts
finishing_times_same_discount = []
finishing_rates_same_discount = []
reward_func, cost_func, reward_func_last, cost_func_last = get_reward_functions( STATES, REWARD_DO, 
                                                                                 EFFORT_DO, REWARD_COMPLETED, COST_COMPLETED )
T = get_transition_prob(STATES, EFFICACY)

V_opt_full, policy_opt_full, Q_values_full =  mdp_algms.find_optimal_policy_diff_discount_factors( STATES, ACTIONS, 
                                              HORIZON, DISCOUNT_FACTOR_REWARD, DISCOUNT_FACTOR_REWARD, 
                                              reward_func, cost_func, reward_func_last, cost_func_last, T )

effective_policy = np.zeros((len(STATES), HORIZON))+10
effective_policy[0,:] =  np.array([policy_opt_full[HORIZON-1-i][0][i] for i in range(HORIZON)]) # actual policy followed by agent
effective_policy[1,:] =  np.array([policy_opt_full[HORIZON-1-i][1][i] for i in range(HORIZON)]) # actual policy followed by agent

for i in range(N_runs):
     
    s, a, v = mdp_algms.forward_runs(effective_policy, V_opt, initial_state, HORIZON, STATES, T)
    
    if 1 in s: 
        finishing_times_same_discount.append(np.where(s==1)[0][0])
        finishing_rates_same_discount.append(1)
    else:
        finishing_rates_same_discount.append(0)
        
            
# precommit
finishing_times_precommit = []
finishing_rates_precommit = []
reward_func, cost_func, reward_func_last, cost_func_last = get_reward_functions( STATES, REWARD_DO, 
                                                                                 EFFORT_DO, REWARD_COMPLETED, COST_COMPLETED )
T = get_transition_prob(STATES, EFFICACY)

V_opt_full, policy_opt_full, Q_values_full =  mdp_algms.find_optimal_policy_diff_discount_factors( STATES, ACTIONS, 
                                              HORIZON, DISCOUNT_FACTOR_REWARD, DISCOUNT_FACTOR_COST, 
                                              reward_func, cost_func, reward_func_last, cost_func_last, T )   
 
policy_precommit = np.array(policy_opt_full)[HORIZON-1,:,:]  
for i in range(N_runs):
     
    s, a, v = mdp_algms.forward_runs(policy_precommit, V_opt, initial_state, HORIZON, STATES, T)
    
    if 1 in s: 
        finishing_times_precommit.append(np.where(s==1)[0][0])
        finishing_rates_precommit.append(1)
    else:
        finishing_rates_precommit.append(0)
        
fig, axs = plt.subplots(figsize=(5,4), dpi=100)
axs.hist([finishing_times_same_discount, 
          finishing_times_precommit,
          finishing_times_diff_discount],
         density =True,
         bins = np.arange(0,HORIZON+2,1),
         color=[mpl.colors.to_rgba('tab:blue', alpha=0.3), 
                mpl.colors.to_rgba('tab:blue', alpha=0.6),
                'tab:blue'],
         stacked=True)
axs.set_xlabel('finishing times')
axs.set_ylabel('density')
axs.set_xticks([0, 2, 4, 6, 8, 10])
sns.despine()

plt.figure(figsize=(5,4), dpi = 100)
plt.bar(['same \n discount', 'pre-commit', 'different \n discounts'],
        [np.mean(finishing_rates_same_discount), 
         np.mean(finishing_rates_precommit),
         np.mean(finishing_rates_diff_discount)],
         color = 'brown')
plt.xlabel('policy')
plt.ylabel('completion rates')
sns.despine()

#%%
# showing preference reversals
rewards = 1.5 * ( 0.9**np.arange(0,10,1) )
efforts = 1.0 * ( 0.9**np.arange(0,10,1) )
plt.figure(figsize=(4,4), dpi=100)
plt.plot(rewards, label='reward', color='tab:blue', linewidth=2)
plt.plot(efforts, label ='efforts', color='brown', linewidth=2)
plt.xlabel('timestep')
plt.ylabel('value')
plt.vlines(np.where(rewards-efforts == np.max(rewards-efforts))[0][0],
           0.2, 1.5, color='black')
plt.ylim(0.2, 1.5)
plt.legend(frameon=False)
sns.despine()

#%%
# show reversals with delay 

reward_do = 0.0
effort_do = -1.0
reward_completed = 4.0
cost_completed = -0.0

# solve for common discount case
reward_func, reward_func_last = get_reward_functions_common( STATES, reward_do, effort_do, 
                                                             reward_completed, cost_completed )
T = get_transition_prob(STATES, EFFICACY)

V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy_prob_rewards( STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR_COMMON, 
                                                             reward_func, reward_func_last, T )


plt.figure(figsize=(6,4), dpi=100)
plt.hlines( np.where(policy_opt[0, :] == 0)[0][0], 
           0, 8, color = 'brown', linewidth=2)

discount_factor_reward = 0.9
discount_factor_cost = 0.7
reward_func, cost_func, reward_func_last, cost_func_last = get_reward_functions( STATES, reward_do, 
                                                                                 effort_do, reward_completed, cost_completed )
T = get_transition_prob(STATES, EFFICACY)

V_opt_full, policy_opt_full, Q_values_full =  mdp_algms.find_optimal_policy_diff_discount_factors( STATES, ACTIONS, 
                                              HORIZON, discount_factor_reward, discount_factor_cost, 
                                              reward_func, cost_func, reward_func_last, cost_func_last, T )

work = np.full(HORIZON, np.nan)
for i in range(HORIZON):
    
    w = np.where(policy_opt_full[HORIZON-1-i][0, :] == 0)[0]
    if w.size > 0 : work[i] = w[0]

plt.plot(work[:-1], color = 'tab:blue', linewidth=2)

discount_factor_reward = 0.7
discount_factor_cost = 0.9
reward_func, cost_func, reward_func_last, cost_func_last = get_reward_functions( STATES, reward_do, 
                                                                                 effort_do, reward_completed, cost_completed )
T = get_transition_prob(STATES, EFFICACY)

V_opt_full, policy_opt_full, Q_values_full =  mdp_algms.find_optimal_policy_diff_discount_factors( STATES, ACTIONS, 
                                              HORIZON, discount_factor_reward, discount_factor_cost, 
                                              reward_func, cost_func, reward_func_last, cost_func_last, T )

work = np.full(HORIZON, np.nan)
for i in range(HORIZON):
    
    w = np.where(policy_opt_full[HORIZON-1-i][0, :] == 0)[0]
    if w.size > 0 : work[i] = w[0]

plt.plot(work[:-1], color = 'tab:blue', alpha=0.5, linewidth=2)

plt.xlabel('timestep')
plt.ylabel('intended time of starting')