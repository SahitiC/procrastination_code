import numpy as np
import matplotlib.pyplot as plt
from pomdp_algms import *

#%%
def line_coeff(point_1, point_2):
    """
    find coefficients of a 2D line given two points 
    
    inputs: coordinates of the two points (tuples)
    
    outputs: slope and intercept of line
    """
    slope = ( point_2[1] - point_1[1] ) / ( point_2[0] - point_1[0] )
    
    intercept = point_1[1] - slope * point_1[0]
    
    return slope, intercept
    
#%%
# defining POMDP 

# 0: stay/check, 1: work, 2: submit
ACTIONS = ('0', '1', '2')
# 0: initial, 1: final
STATES = ('0', '1')

GAMMA = 0.95

EFFICACY = 0.7

# transition probabilities between states for each action 
T_PROB = [[[1.0, 0.0], 
           [0.0, 1.0]], 
          
          [[1.0-EFFICACY, EFFICACY], 
           [0.0, 1.0]], 
          
          [[1.0, 0.0], 
           [0.0, 1.0]]]

# OBSERVATION PROBABILITIES FOR EACH ACTION
E_PROB = [[[0.9, 0.1], 
           [0.1, 0.9]], 
          
          [[0.5, 0.5], 
           [0.5, 0.5]], 
          
          [[0.5, 0.5], 
           [0.5, 0.5]]]

# REWARDS FOR EACH ACTION IN EACH STATE
REWARDS = [[-0.1, -0.1], 
           [-1.0, -1.0], 
           [-4.0, 4.0]]

#%%
# solve pomdp and get regions 

pomdp = POMDP(ACTIONS, T_PROB, E_PROB, REWARDS, STATES, GAMMA)

utility = pomdp_value_iteration(pomdp, epsilon=0.1)

intersections, a_best, u_best = get_regions(pomdp, utility)

# plot values and actions
colors = ['g', 'b', 'k']
actions_names = ['check', 'work', 'submit']
fig, axs = plt.subplots( figsize = (8, 6) )
for i, action in enumerate(a_best):
    m, b = line_coeff([0, u_best[i][0]], [1, u_best[i][1]])
    axs.plot([intersections[i], intersections[i+1]], 
             [m * intersections[i] + b, m * intersections[i+1] + b],
             color=colors[int(action)])
axs.set_xlabel('belief state')
axs.set_ylabel('value')
handles, labels = axs.get_legend_handles_labels() 
for i_color, color in enumerate(colors):
    handles.append(axs.plot([], [], color = f'{colors[i_color]}', label = f'{actions_names[i_color]}'))
axs.legend()

#%%
# change params and run above in loop
p_os = [0.7]

fig, axs = plt.subplots( figsize = (8, 6) )

for i_p_o, p_o in enumerate(p_os):
    
    e_prob = [[[p_o, 1.0-p_o], 
               [1.0-p_o, p_o]], 
              
              [[0.5, 0.5], 
               [0.5, 0.5]], 
              
              [[0.5, 0.5], 
               [0.5, 0.5]]]

    pomdp = POMDP(ACTIONS, T_PROB, e_prob, REWARDS, STATES, GAMMA)
    
    utility = pomdp_value_iteration(pomdp, epsilon=0.1)
    
    intersections, a_best, u_best = get_regions(pomdp, utility)
    
    # plot values and actions
    colors = ['g', 'b', 'k']
    actions_names = ['check', 'work', 'submit']
    
    for i, action in enumerate(a_best):
        m, b = line_coeff([0, u_best[i][0]], [1, u_best[i][1]])
        axs.plot([intersections[i], intersections[i+1]], 
                 [m * intersections[i] + b, m * intersections[i+1] + b],
                 color=colors[int(action)])
        axs.text(0, np.max(np.array(u_best)[:,0])+0.5, f'p_obs={p_o}')
    
axs.set_xlabel('belief state', fontsize = 16)
axs.set_ylabel('value', fontsize = 16)
handles, labels = axs.get_legend_handles_labels() 
for i_color, color in enumerate(colors):
    handles.append(axs.plot([], [], color = f'{colors[i_color]}', label = f'{actions_names[i_color]}'))
axs.legend(fontsize=14)
axs.tick_params(labelsize = 14)