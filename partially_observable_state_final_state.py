"""
This is the script for a task where there might be another state where more work can be done. 
However, there is uncertainty whether the task can actaully be improved and can be
resolved only by checking. Here,  it is a pomdp with 3 states 
We can have terminal states or loop through infinite trials.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 16
mpl.rcParams['lines.linewidth'] = 2
plt.rcParams['text.usetex'] = False
import seaborn as sns
import time
import pomdp_algms

#%%
# define the pomdp 
terminal_state = 1 # 1 if terminal state existis 0 otherwise
states = np.array( [0, 1, 2] ) # (1,0), (1,1), 2 : all non-terminal states
actions = np.array( [0,1,2]) #'check', 'work', 'submit'
observations = np.array( [0, 1, 2] )
efficacy = 0.7
noise = 0.3
discount_factor = 1.0
db = 0.05 # discretisation of belief space
max_iter = 100 # maximum value iteration rounds
eps = 1e-3 # diff in value (diff_value) required for value iteration convergence

# transition probabilities between states for each action 
t_prob = np.array( [[[1.0, 0.0, 0.0], 
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 0.0]], 
          
                    [[1.0, 0.0, 0.0], 
                     [0.0, 1.0-efficacy, efficacy],
                     [0.0, 0.0, 0.0]], 
          
                     [[0.0, 0.0, 0.0], 
                      [0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0]]] )

# only submit action takes to terminal state from each of the other states
t_prob_terminal = np.array([[0,0,0],
                            [0,0,0],
                            [1,1,1]])

# observation probabilities for each action
e_prob =  np.array( [[[1.0-noise, noise, 0.0], 
                    [noise, 1.0-noise, 0.0],
                    [0.0, 0.0, 1.0]], 
                  
                   [[0.5, 0.5, 0.0], 
                    [0.5, 0.5, 0.0],
                    [0.0, 0.0, 1.0]], 
                  
                   [[0.5, 0.5, 0.0], 
                    [0.5, 0.5, 0.0],
                    [0.0, 0.0, 1.0]]] )

# rewards for each action in each state
rewards = np.array([[-0.1, -0.1, -0.1], 
                    [-1.5, -1.5, -1.5], 
                    [0.0, 0.0, 5.0]])

#%%

# define value and policy variables in grid of belief states
# represent the beliefs of first N-1 states (N: total number of states)

start = time.time()

policy, value = pomdp_algms.get_optimal_policy_2D(states, actions, observations, e_prob, t_prob,
                          rewards, discount_factor, db, max_iter, eps)

end = time.time()
print(f"time taken: {end-start}s")

# plot policy
fig, ax = plt.subplots( figsize=(5,5) )
cmap = viridis = mpl.colormaps['viridis'].resampled(3)
p = ax.imshow(policy, cmap=cmap, origin='lower')
ax.set_xlabel('belief in state = 1')
ax.set_ylabel('belief in state = 0')
ax.set_xticks([0, 10, 20], [0, 0.5, 1])
ax.set_yticks([0, 10, 20], [0, 0.5, 1])
cbar = fig.colorbar(p, shrink=0.7)
cbar.set_ticks([0.3, 1.0, 1.7])
cbar.set_ticklabels(['check', 'work', 'submit'])

#%%
# forward runs
# given a policy and an initial belief and state, sample trajectories of actions
plt.figure( figsize = (7, 5) )
initial_belief = np.array( [0.5, 0.5, 0.0] )
initial_hidden_state = 0 #np.random.choice([0, 1], p = [0.5, 0.5])

for i_run in range(50):
    
    trajectory = pomdp_algms.forward_runs_2D(initial_belief, initial_hidden_state, policy, db, 
                                             states, observations, e_prob, t_prob, t_prob_terminal)                                          
    
    plt.plot( trajectory[2], marker = 'o', linestyle = '--' )
    
plt.xlabel('timestep')
plt.ylabel('action')
plt.yticks(actions, labels=['check', 'work', 'submit'])
        
#%%
# final plots

# get 1-D policy (the real confusion is only between states 0 or 1)
policy_1d = np.flipud(policy).diagonal()

plt.figure(figsize=(8,6), dpi=100)
policy_1d=np.expand_dims(policy_1d, axis=0)
plt.imshow(policy_1d, cmap=cmap)
plt.xticks(ticks=[0, 0.5/db, 1/db], labels=[0., .5, 1.])
plt.yticks([0.0], ['policy'])
plt.xlim([0,1/db])
plt.xlabel('belief (S=1) ')
sns.despine()

#%%

# plot trajectory
initial_belief = np.array( [0.5, 0.5, 0.0] )
initial_hidden_state = 0 
trajectory = pomdp_algms.forward_runs_2D(initial_belief, initial_hidden_state, policy, db, 
                                         states, observations, e_prob, t_prob, t_prob_terminal)
trajectory_init_0 = trajectory

initial_belief = np.array( [0.5, 0.5, 0.0] )
initial_hidden_state = 1
trajectory = pomdp_algms.forward_runs_2D(initial_belief, initial_hidden_state, policy, db, 
                                         states, observations, e_prob, t_prob, t_prob_terminal)
trajectory_init_1 = trajectory


plt.figure(figsize=(6,4), dpi=100)

plt.plot(np.array(trajectory_init_0[0])[:,1], # belief(s=1)
         linestyle=(5, (10, 3)), # long dashed line
         linewidth = 2,
         color = 'darkgray',
         label='0') 

plt.scatter(np.arange(len(trajectory_init_0[1])), 
            np.array(trajectory_init_0[0])[:,1],
            marker='s', s=100,
            c=cmap(trajectory_init_0[2]))

plt.plot(np.array(trajectory_init_1[0])[:,1], # belief(s=1)
         linestyle=(5, (10, 3)), # long dashed line
         linewidth = 2,
         color = 'black',
         label='1') 

plt.scatter(np.arange(len(trajectory_init_1[1])), 
            np.array(trajectory_init_1[0])[:,1],
            marker='s', s=100,
            c=cmap(trajectory_init_1[2]))

plt.xlabel('timesteps')
plt.ylim(top=1.0)
plt.ylabel('belief(s=1)')
plt.legend(title='hidden state', title_fontsize=18, 
           frameon=False, loc='upper right')
sns.despine()

#%%

# average time of submission and correct submission rates

rewards_compl = np.array( [2.0, 3.0, 5.0, 6.0, 7.0] )
submission_times = np.zeros((200, len(rewards_compl), 2))
correct_submissions = np.zeros((200, len(rewards_compl)))

for i_reward, reward in enumerate(rewards_compl):
    
    rewards = np.array([[-0.1, -0.1, -0.1], 
                        [-1.5, -1.5, -1.5], 
                        [0.0, 0.0, reward]])
    
    policy, value = pomdp_algms.get_optimal_policy_2D(states, actions, observations, e_prob, t_prob,
                              rewards, discount_factor, db, max_iter, eps)
        
    for i in range(200):
        
        initial_belief = np.array( [0.5, 0.5, 0.0] )
        
        for initial_hidden_state in range(2):    
            
            trajectory = pomdp_algms.forward_runs_2D(initial_belief, initial_hidden_state, policy, db, 
                                                 states, observations, e_prob, t_prob, t_prob_terminal)  
        
            submission_times[i, i_reward, initial_hidden_state] = len(trajectory[1])
        
            if initial_hidden_state == 1:
                correct_submissions[i, i_reward] = int(trajectory[1][-1]==2)

fig, axs = plt.subplots(figsize=(6,4), dpi=100)

mean = np.mean(submission_times[:,:,0], axis = 0)
std = np.std(submission_times[:,:,0], axis = 0)/np.sqrt(100)
axs.plot(rewards_compl,
         mean, 
         linestyle = '--',
         linewidth = 2,
         marker = 'o', markersize = 5,
         label = '0',
         color = 'brown')

axs.fill_between(rewards_compl,
                 mean-std,
                 mean+std,
                 alpha=0.3,
                 color = 'brown')

mean = np.mean(submission_times[:,:,1], axis = 0)
std = np.std(submission_times[:,:,1], axis = 0)/np.sqrt(100)
axs.plot(rewards_compl,
         mean, 
         linestyle = '--',
         linewidth = 2,
         marker = 'o', markersize = 5,
         label = '1',
         color = 'tomato')

axs.fill_between(rewards_compl,
                 mean-std,
                 mean+std,
                 alpha=0.3,
                 color = 'tomato')

axs.set_xlabel('reward on completion')
axs.set_ylabel('avg submission time', color='brown')
axs.tick_params(axis='y', labelcolor='brown')


ax2 = axs.twinx()
mean = np.mean(correct_submissions, axis = 0)
ax2.plot(rewards_compl,
         mean,
         linewidth = 3,
         color='tab:blue')
ax2.set_ylabel('avg completion rate', color='tab:blue')
ax2.tick_params(axis='y', labelcolor='tab:blue')
fig.legend(frameon=False, title = 'hidden state', bbox_to_anchor=(0.4, 0.6))


#%%

priors = np.linspace(0.2, 0.9, 30)
submission_times = np.zeros((1000, len(priors)))
correct_submissions = np.full((1000, len(priors)), np.nan)
efficacy = 0.8
policy, value = pomdp_algms.get_optimal_policy_2D(states, actions, observations, e_prob, t_prob,
                          rewards, discount_factor, db, max_iter, eps)

for i_prior, prior in enumerate(priors):
        
    for i in range(1000):
        
        initial_belief = np.array( [1-prior, prior, 0.0] )
        
        initial_hidden_state = np.random.choice([0, 1], p = [0.6, 0.4])   
            
        trajectory = pomdp_algms.forward_runs_2D(initial_belief, initial_hidden_state, policy, db, 
                                             states, observations, e_prob, t_prob, t_prob_terminal)  
    
        submission_times[i, i_prior] = len(trajectory[1])
    
        if initial_hidden_state == 1:
            correct_submissions[i, i_prior] = int(trajectory[1][-1]==2)

fig, axs = plt.subplots(figsize=(6,4), dpi=100)

mean = np.mean(submission_times, axis = 0) 
std = np.std(submission_times, axis = 0)/np.sqrt(100)
axs.plot(priors,
         mean, 
         linestyle = '--',
         linewidth = 2,
         marker = 'o', markersize = 5,
         color = 'brown')

axs.fill_between(priors,
                 mean-std,
                 mean+std,
                 alpha=0.3,
                 color = 'brown')

axs.set_xlabel('prior prob (s=1)')
axs.set_ylabel('avg submission time', color='brown')
axs.tick_params(axis='y', labelcolor='brown')
axs.vlines(0.4,
           0, 7, color='black', linestyle='dashed')
axs.set_ylim(0,7)


ax2 = axs.twinx()
mean = np.nanmean(correct_submissions, axis = 0)
ax2.plot(priors,
         mean,
         linewidth = 3,
         color='tab:blue')
ax2.set_ylabel('avg completion rate', color='tab:blue',
               rotation=270, labelpad=15)
ax2.tick_params(axis='y', labelcolor='tab:blue')
