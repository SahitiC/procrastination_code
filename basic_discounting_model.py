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
plt.rcParams['text.usetex'] = False
import mdp_algms
import seaborn as sns

#%%

# construct reward functions
def get_reward_functions(states, reward_pass, reward_fail, reward_shirk, reward_completed, effort_work, effort_shirk):
    
    # reward from actions within horizon
    reward_func = [] 
    reward_func = [([effort_work, reward_shirk + effort_shirk]) for i in range( len(states)-1 )] # rewards in non-completed states
    reward_func.append( [reward_completed] ) # reward in completed state
    
    # reward from final evaluation
    reward_func_last =  np.linspace(reward_fail, reward_pass, len(states)) 
    
    return reward_func, reward_func_last

#immediate rewards
def get_reward_functions_immediate(states, reward_work, reward_shirk, effort_work, effort_shirk):
    
    # reward from actions within horizon
    reward_func = [] 
    reward_func = [([reward_work + effort_work, reward_shirk + effort_shirk]) for i in range( len(states)-1 )] # rewards in non-completed states
    reward_func.append( [reward_shirk + effort_shirk] ) # reward in completed state
    
    # reward from final evaluation
    reward_func_last =  np.linspace(reward_fail, reward_pass, len(states)) 
    
    return reward_func, reward_func_last


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
    
    return T

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
EFFICACY = 0.9 # self-efficacy (probability of progress on working)

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

plt.figure(figsize=(8,6), dpi=100)
plt.bar( efficacys, count_always_work[:, 0]/N_runs, alpha = 0.5, width=0.1, color='tab:blue', label = 'always work')
plt.bar( efficacys, count_opt[:, 0]/N_runs, alpha = 1, width=0.1, color='tab:blue', label = 'optimal policy')
plt.legend(fontsize=16)
plt.xlabel('efficacy', fontsize=20)
plt.ylabel('Proportion of finished runs', fontsize=20)
plt.tick_params(labelsize = 20)
plt.savefig('planned_delay.png', dpi=100)

#%%
# final plots

# demonstration of discounting

discount_factor = 0.9
reward_pass = 2
reward_fail = -2

reward_func, reward_func_last = get_reward_functions(STATES, reward_pass, reward_fail, REWARD_SHIRK, 
                                                     REWARD_COMPLETED, EFFORT_WORK, EFFORT_SHIRK)
T = get_transition_prob(STATES, EFFICACY)
V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy(STATES, ACTIONS, HORIZON, 
                              discount_factor, reward_func, reward_func_last, T)

# plots of policies and values
plt.figure( figsize = (4, 4) , dpi = 100)
colors = ['tab:blue', 'brown'] # work:blue, shirk: brown
linestyles = ['solid', 'dashed']

for i_state, state in enumerate(STATES[:-1]):
    
    
    for i_action, action in enumerate(ACTIONS[i_state]):
        
        plt.plot(Q_values[i_state][i_action, :], 
                 color = colors[i_action],
                 linestyle = linestyles[i_state],
                 linewidth = 2)

plt.xlabel('timesteps')
plt.ylabel('q values')
sns.despine()

# plot trajectories
initial_state = 0
s, a, v = mdp_algms.forward_runs(policy_opt, V_opt, initial_state, HORIZON, STATES, T)

# change action value of shirk in state=2 to get correct color 
ac = np.where(s[:-1]==2, 1, a)
plt.figure( figsize = (5,4), dpi = 100 )
plt.plot(np.arange(0, HORIZON+1, 1),
         s, linewidth = 2,
         linestyle = 'dashed',
         color='gray')
plt.scatter(np.arange(0, HORIZON, 1), 
            s[:-1],
            marker='s', s=100,
            c=np.where(ac==0, colors[0], colors[1]))
plt.xlabel('timesteps')
plt.ylabel('state')
plt.yticks([0,1,2])
sns.despine()

#%%
# completion times and rate improved by greater rewards for completion

N_runs  = 1000
initial_state = 0
completion_times = np.full((N_runs, 6), np.nan)
completion_rates = np.zeros((N_runs, 6))
efforts = np.array([0.0, -0.1, -0.2, -0.4, -0.6, -0.8])

efficacy = 0.6

for i_e, effort_work in enumerate(efforts):
    
    
    reward_func, reward_func_last = get_reward_functions(STATES, REWARD_PASS, REWARD_FAIL, REWARD_SHIRK, 
                                                         REWARD_COMPLETED, effort_work, EFFORT_SHIRK)
    T = get_transition_prob(STATES, efficacy)
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy(STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR, 
                                  reward_func, reward_func_last, T)
    
    for i in range(N_runs):
         
        s, a, v = mdp_algms.forward_runs(policy_opt, V_opt, initial_state, HORIZON, STATES, T)
        #append completion time if task is completed
        if 2 in s: 
            completion_rates[i, i_e] = 1.0
            completion_times[i, i_e] = np.where(s==2)[0][0]
        
fig, axs = plt.subplots(figsize=(6,4), dpi=100)

mean = np.nanmean(completion_times, axis = 0) 
std = np.nanstd(completion_times, axis = 0)/np.sqrt(1000)
axs.plot(efforts,
         mean, 
         linestyle = '--',
         linewidth = 2,
         marker = 'o', markersize = 5,
         color = 'brown')

axs.fill_between(efforts,
                 mean-std,
                 mean+std,
                 alpha=0.3,
                 color = 'brown')

axs.set_xlabel('effort to work')
axs.set_ylabel('avg completion time', color='brown')
axs.tick_params(axis='y', labelcolor='brown')


ax2 = axs.twinx()
mean = np.nanmean(completion_rates, axis = 0)
ax2.plot(efforts,
         mean,
         linewidth = 3,
         color='tab:blue')
ax2.set_ylabel('avg completion rate', color='tab:blue', rotation = 270, labelpad=15)
ax2.tick_params(axis='y', labelcolor='tab:blue')

#%%

    
efficacys = np.linspace(0, 1, 10) # vary efficacy 
policy_always_work = np.full(np.shape(policy_opt), 0) # always work policy
V_always_work = np.full(np.shape(V_opt), 0.0)

# arrays to store no. of runs where task was finished 
count_opt = np.full( (len(efficacys), 1), 0) 
count_imm = np.full( (len(efficacys), 1), 0) # policy of always work
N_runs = 1000 # no. of runs for each parameter set
reward_fail = 0
reward_pass = 5.0

for i_efficacy, efficacy in enumerate(efficacys):
    
    # get optimal policy for current parameter set
    reward_func, reward_func_last = get_reward_functions(STATES, reward_pass, reward_fail, REWARD_SHIRK, 
                                                         REWARD_COMPLETED, EFFORT_WORK, EFFORT_SHIRK)
    T = get_transition_prob(STATES, efficacy)
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy(STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR, 
                                  reward_func, reward_func_last, T)
    
    # get optimal policy for current parameter set
    reward_func, reward_func_last =  get_reward_functions_immediate(STATES, 
                                reward_pass, REWARD_SHIRK, EFFORT_WORK, EFFORT_SHIRK)
    T = get_transition_prob(STATES, efficacy)
    V_opt, policy_opt_imm, Q_values = mdp_algms.find_optimal_policy(STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR, 
                                  reward_func, reward_func_last, T)
    
    # run forward (N_runs no. of times), count number of times task is finished for each policy
    initial_state = 0
    for i in range(N_runs):
         
        s, a, v = mdp_algms.forward_runs(policy_opt, V_opt, initial_state, HORIZON, STATES, T)
        if s[-1] == len(STATES)-1: count_opt[i_efficacy,0] +=1
        
        s, a, v = mdp_algms.forward_runs(policy_opt_imm, V_opt, initial_state, HORIZON, STATES, T)
        if s[-1] == len(STATES)-1: count_imm[i_efficacy,0] +=1

plt.figure(figsize=(5,4), dpi=100)
plt.bar( efficacys, count_imm[:, 0]/N_runs, alpha = 0.5, width=0.1, color='tab:blue', label = 'immediate rewards')
plt.bar( efficacys, count_opt[:, 0]/N_runs, alpha = 1, width=0.1, color='tab:blue', label = 'delayed rewards')
plt.xlabel('efficacy')
plt.ylabel('Proportion of finished runs')
sns.despine()

#%%
# legends 
colors = ["tab:blue"#mpl.colors.to_rgba('tab:blue', alpha=0.5),
          "brown"]
cmap= mpl.colormaps.get_cmap('viridis')
colors = [cmap(1.0), cmap(0.5), cmap(0.0)]
plt.figure(figsize=(1,1), dpi=100)
f = lambda m,c: plt.plot([],[],marker=m, markersize=15, color=c, ls="none")[0]
handles = [f("s", colors[i]) for i in range(3)]
labels = ["submit", "work", "check"]
legend = plt.legend(handles, labels, loc=3, 
                    framealpha=1, frameon=False) 
                    #title='condition', title_fontsize=18)
fig  = legend.figure
fig.canvas.draw()
plt.axis('off')
plt.show()

