import numpy as np
import matplotlib as mpl
mpl.rcParams['font.size'] = 16
mpl.rcParams['lines.linewidth'] = 2
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
import mdp_algms
import seaborn as sns

#%%

# construct reward functions
def get_reward_functions(states, reward_pass, reward_fail, reward_shirk, reward_completed, 
                         effort_work, effort_shirk):
    
    # reward from actions within horizon
    reward_func = [] 
    
    for i in range( len(states)-1 ):
        reward_func.append( [effort_work[i], reward_shirk + effort_shirk]  )  # rewards in non-completed states
        
    reward_func.append( [reward_completed] ) # reward in completed state
    
    # reward from final evaluation
    reward_func_last = []
    
    for i in range( len(states)-1 ):
        reward_func_last.append( reward_fail )   # rewards in non-completed states

    reward_func_last.append(reward_pass) 
    
    return np.array(reward_func, dtype=object), np.array(reward_func_last, dtype=object)

def get_transition_prob(states, efficacy, dynamics):
    
    T = []
    T.append( [ np.array([(1-efficacy)*dynamics[0,0], (1-efficacy)*dynamics[0,1], (1-efficacy)*dynamics[0,2], efficacy]), 
                np.array([dynamics[0,0], dynamics[0,1], dynamics[0,2], 0]) ] ) # transitions for work, shirk
    
    T.append( [ np.array([(1-efficacy)*dynamics[1,0], (1-efficacy)*dynamics[1,1], (1-efficacy)*dynamics[1,2], efficacy]), 
                np.array([dynamics[1,0], dynamics[1,1], dynamics[1,2], 0]) ] ) # transitions for work, shirk
    
    T.append( [ np.array([(1-efficacy)*dynamics[2,0], (1-efficacy)*dynamics[2,1], (1-efficacy)*dynamics[2,2], efficacy]), 
                np.array([dynamics[2,0], dynamics[2,1], dynamics[2,2], 0]) ] ) # transitions for work, shirk
    
    T.append( [ np.array([0, 0, 0, 1]) ] ) # transitions for completed
    
    return np.array(T, dtype=object)

#%%

# instantiating MDP

# states of markov chain
N_DIFFICULTY_STATES = 3 # number of states of difficulty
STATES = np.arange(1 + N_DIFFICULTY_STATES) # final state and other initial states with varying difficulties of task completion

ACTIONS = []
ACTIONS = [ ['work', 'shirk'] for i in range( len(STATES)-1 ) ]
ACTIONS.append(['completed']) 

HORIZON = 10 # deadline
DISCOUNT_FACTOR = 1.0# discounting factor
EFFICACY = 0.7# self-efficacy (probability of progress on working) in non-start/finished state

# utilities :
REWARD_PASS = 4.0 
REWARD_FAIL = -4.0
REWARD_SHIRK = 0.5 
EFFORT_WORK = [-0.2, -0.5, -1.0] # effort to complete task from one of the difficulty states
EFFORT_SHIRK = -0 
REWARD_COMPLETED = REWARD_SHIRK

# envt dynmics : transitions between difficulty states independent of actions

DYNAMICS = np.array( [[0.9, 0.1, 0.0],
                      [0.9, 0.1, 0.0],
                      [0.0, 0.9, 0.1]] ) 

# DYNAMICS = np.array( [[1.0, 0.0, 0.0],
#                       [0.05, 0.95, 0.0],
#                       [0.0, 0.05, 0.95]] ) 
# DYNAMICS = np.array( [[0.2, 0.2, 0.6],
#                       [0.6, 0.2, 0.2],
#                       [0.2, 0.6, 0.2]] ) # cyclic
# DYNAMICS = np.array( [[1.0, 0.0, 0.0],
#                       [0.0, 1.0, 0.0],
#                       [0.0, 0.0, 1.0]] ) # identity

    
#%%
# get optimal policy

reward_func, reward_func_last = get_reward_functions(STATES, REWARD_PASS, REWARD_FAIL, REWARD_SHIRK, 
                                                     REWARD_COMPLETED, EFFORT_WORK, EFFORT_SHIRK)
T = get_transition_prob(STATES, EFFICACY, DYNAMICS)
V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy(STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR, 
                              reward_func, reward_func_last, T)

# plots of policies and values
colors = plt.cm.Reds(np.linspace(0.3,0.9,len(STATES)))
lines = ['--', ':']
fig, axs = plt.subplots( figsize = (5, 4), dpi=100 )
fig1, axs1 = plt.subplots( figsize = (5, 4), dpi=100 )
state_names = ['easy', 'medium', 'hard']

for i_state, state in enumerate(STATES[:-1]):
    
    
    axs.plot(V_opt[i_state], color = colors[i_state], marker = 'o', linestyle = 'None', label = f'$V^*({i_state})$',)
    axs1.plot(policy_opt[i_state], color = colors[i_state], label = f'{state_names[i_state]}')
    
    for i_action, action in enumerate(ACTIONS[i_state]):
        
        axs.plot(Q_values[i_state][i_action, :], color = colors[i_state], linestyle = lines[i_action])
        
handles, labels = axs.get_legend_handles_labels()   
handles.append(axs.plot([], [], color = 'black', linestyle = '--', label = '$Q(a=$ work$)$'))
handles.append(axs.plot([], [], color = 'black', linestyle = ':', label = '$Q(a=$ shirk$)$'))
axs.legend()
axs.set_xlabel('timesteps')
#axs1.legend(frameon=False, title='States', title_fontsize=18)
axs1.set_xlabel('timesteps', fontsize=20)
axs1.set_yticks([0,1])
axs1.set_yticklabels(['WORK', 'SHIRK'])
axs1.set_ylabel('policy', fontsize=20)
sns.despine()

#%%
# final plots

    
# run forward (N_runs no. of times), 
# get distribution of finish times and compare with 
N_runs  = 1000
initial_state = 2
policy_always_work = np.zeros(np.shape(policy_opt))

efficacy = 0.7
reward_func, reward_func_last = get_reward_functions(STATES, REWARD_PASS, REWARD_FAIL, REWARD_SHIRK, 
                                                     REWARD_COMPLETED, EFFORT_WORK, EFFORT_SHIRK)
T = get_transition_prob(STATES, efficacy, DYNAMICS)
V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy(STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR, 
                              reward_func, reward_func_last, T)

completion_times_eff1 = []
completion_times_eff1_ideal = []
for i in range(N_runs):
     
    s, a, v = mdp_algms.forward_runs(policy_opt, V_opt, initial_state, HORIZON, STATES, T)
    #append completion time if task is completed
    if 3 in s: completion_times_eff1.append(np.where(s==3)[0][0])
    
    s, a, v = mdp_algms.forward_runs(policy_always_work, V_opt, initial_state, HORIZON, STATES, T)
    #append completion time if task is completed
    if 3 in s: completion_times_eff1_ideal.append(np.where(s==3)[0][0])
    
  

plt.figure(figsize=(5,4), dpi=100)
plt.hist([completion_times_eff1, completion_times_eff1_ideal],
         density =True,
         bins = np.arange(0,HORIZON,1),
         color=[ 'tab:blue', mpl.colors.to_rgba('tab:blue', alpha=0.5)],
         stacked=True)

plt.xlabel('completion times')
plt.ylabel('density')
sns.despine()

#%%
# completion times and rate improved by greater rewards for completion

N_runs  = 1000
initial_state = 2
completion_times = np.full((N_runs, 5), np.nan)
completion_rates = np.zeros((N_runs, 5))
rewards = np.array([2.5, 3.0, 4.0, 5.0, 6.0])
efficacy=0.6

for i_r, reward_pass in enumerate(rewards):
    
    REWARD_FAIL = 0.0
    
    reward_func, reward_func_last = get_reward_functions(STATES, reward_pass, REWARD_FAIL, REWARD_SHIRK, 
                                                         REWARD_COMPLETED, EFFORT_WORK, EFFORT_SHIRK)
    T = get_transition_prob(STATES, efficacy, DYNAMICS)
    V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy(STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR, 
                                  reward_func, reward_func_last, T)
    
    for i in range(N_runs):
         
        s, a, v = mdp_algms.forward_runs(policy_opt, V_opt, initial_state, HORIZON, STATES, T)
        #append completion time if task is completed
        if 3 in s: 
            completion_rates[i, i_r] = 1.0
            completion_times[i, i_r] = np.where(s==3)[0][0]
        
fig, axs = plt.subplots(figsize=(6,4), dpi=100)

mean = np.nanmean(completion_times, axis = 0) 
std = np.nanstd(completion_times, axis = 0)/np.sqrt(1000)
axs.plot(rewards,
         mean, 
         linestyle = '--',
         linewidth = 2,
         marker = 'o', markersize = 5,
         color = 'brown')

axs.fill_between(rewards,
                 mean-std,
                 mean+std,
                 alpha=0.3,
                 color = 'brown')

axs.set_xlabel('reward on completion')
axs.set_ylabel('avg completion time', color='brown')
axs.tick_params(axis='y', labelcolor='brown')


ax2 = axs.twinx()
mean = np.nanmean(completion_rates, axis = 0)
ax2.plot(rewards,
         mean,
         linewidth = 3,
         color='tab:blue')
ax2.set_ylabel('avg completion rate', 
               color='tab:blue', 
               rotation=270,
               labelpad=15)
ax2.tick_params(axis='y', labelcolor='tab:blue')

#%%
# overestimating probability of task becoming easier
efficacy = 0.6
initial_state = 2
policy_always_work = np.zeros(np.shape(policy_opt))

# policy according to this optimism
dynamics_optimistic = np.array( [[0.9, 0.1, 0.0],
                      [0.9, 0.1, 0.0],
                      [0.0, 0.9, 0.1]] )

reward_pass = 4.0
reward_func, reward_func_last = get_reward_functions(STATES, reward_pass, REWARD_FAIL, REWARD_SHIRK, 
                                                     REWARD_COMPLETED, EFFORT_WORK, EFFORT_SHIRK)
T = get_transition_prob(STATES, efficacy, dynamics_optimistic)
V_optimistic, policy_optimistic, Q_values = mdp_algms.find_optimal_policy(STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR, 
                              reward_func, reward_func_last, T)

dynamics_real = np.array( [[1.0, 0.0, 0.0],
                        [0.05, 0.95, 0.0],
                        [0.0, 0.05, 0.95]] ) 
reward_func, reward_func_last = get_reward_functions(STATES, reward_pass, REWARD_FAIL, REWARD_SHIRK, 
                                                     REWARD_COMPLETED, EFFORT_WORK, EFFORT_SHIRK)
T = get_transition_prob(STATES, efficacy, dynamics_real)
V_opt, policy_opt, Q_values = mdp_algms.find_optimal_policy(STATES, ACTIONS, HORIZON, DISCOUNT_FACTOR, 
                              reward_func, reward_func_last, T)

N_runs = 1000
completion_times = np.full((N_runs, 3), np.nan)
completion_rates = np.zeros((N_runs, 3))
for i in range(N_runs):

    s, a, v = mdp_algms.forward_runs(policy_always_work, V_opt, initial_state, HORIZON, STATES, T)
    if 3 in s: 
        completion_rates[i, 0] = 1
        completion_times[i, 0] = np.where(s==3)[0][0]
        
    s, a, v = mdp_algms.forward_runs(policy_optimistic, V_optimistic, initial_state, HORIZON, STATES, T)
    if 3 in s: 
        completion_rates[i, 1] = 1
        completion_times[i, 1] = np.where(s==3)[0][0]
        
    s, a, v = mdp_algms.forward_runs(policy_opt, V_opt, initial_state, HORIZON, STATES, T)
    if 3 in s: 
        completion_rates[i, 2] = 1
        completion_times[i, 2] = np.where(s==3)[0][0]
 
plt.figure(figsize=(4,4), dpi =100)
plt.bar(['always \n work', 'optimistic', 'optimal \n policy'],
        [np.nanmean(completion_times[:,0]),
         np.nanmean(completion_times[:,1]),
         np.nanmean(completion_times[:,2])],
        yerr = [np.nanstd(completion_times[:,0]),
                np.nanstd(completion_times[:,1]),
                np.nanstd(completion_times[:,2])],
        color = 'brown')
plt.xlabel('policy', fontsize=18)
plt.ylabel('avg completion time', fontsize=18)
sns.despine()

plt.figure(figsize=(4,4), dpi =100)
plt.bar(['always \n work', 'optimistic', 'optimal \n policy'],
        [np.nanmean(completion_rates[:,0]),
         np.nanmean(completion_rates[:,1]),
         np.nanmean(completion_rates[:,2])],
        color = 'tab:blue')
plt.xlabel('policy', fontsize=18)
plt.ylabel('avg completion rate', fontsize=18)
sns.despine()
