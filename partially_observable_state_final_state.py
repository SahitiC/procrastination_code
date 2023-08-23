import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.size'] = 16
mpl.rcParams['lines.linewidth'] = 2
import time
import pomdp_algms

#%%
# define the pomdp 
states = np.array( [0, 1, 2] ) # (1,0), (1,1), 2
actions = np.array( [0,1,2]) #'check', 'work', 'submit'
observations = np.array( [0, 1, 2] )
# transition probabilities between states for each action 
efficacy = 0.7
noise = 0.3
discount_factor = 0.95
db = 0.05 # discretisation of belief space
max_iter = 100 # maximum value iteration rounds
eps = 1e-3 # diff in value (diff_value) required for value iteration convergence
# transition probabilities between states for each action 
t_prob = np.array( [[[1.0, 0.0, 0.0], 
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]], 
          
                    [[1.0, 0.0, 0.0], 
                     [0.0, 1.0-efficacy, efficacy],
                     [0.0, 0.0, 1.0]], 
          
                     [[0.5, 0.5, 0.0], 
                      [0.5, 0.5, 0.0],
                      [0.5, 0.5, 0.0]]] )

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
                    [-1.0, -1.0, -1.0], 
                    [-1.0, -1.0, 3.0]])

#%%

# define value and policy variables in grid of belief states
# represent the beliefs of first N-1 states (N: total number of states)

start = time.time()

policy, value = pomdp_algms.get_optimal_policy_2D(states, actions, observations, e_prob, t_prob,
                          rewards, discount_factor,
                          db, max_iter, eps)

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
initial_hidden_state = 1 #np.random.choice([0, 1], p = [0.5, 0.5])

for i_run in range(50):
    
    trajectory = pomdp_algms.forward_runs_2D(initial_belief, initial_hidden_state, policy)                                          
    
    plt.plot( trajectory[2], marker = 'o', linestyle = '--' )
    
plt.xlabel('timestep')
plt.ylabel('action')
plt.yticks(actions, labels=['check', 'work', 'submit'])
        