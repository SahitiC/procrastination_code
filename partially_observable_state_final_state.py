import numpy as np
import matplotlib.pyplot as plt
import math
import time

#%%
def pO_ba(belief, action, observation, e_prob, t_prob):
    """
    probability of observation given the belief and action
    """
    return e_prob[action][:, observation].T @ (t_prob[action].T @ belief)

def get_next_belief(belief, action, observation, e_prob, t_prob):
    """
    finding the next belief state given belief, action and observation
    """
    return e_prob[action][:, observation] * (t_prob[action].T @ belief) / (
           pO_ba(belief, action, observation, e_prob, t_prob) )

def round_down(num,digits):
    factor = 10.0 ** digits
    return math.floor(num * factor) / factor

def round_up(num,digits):
    factor = 10.0 ** digits
    return math.ceil(num * factor) / factor

def interpolate1d(x, x_0, x_1, dx, func_0, func_1):
    """
    interpolate for a discrete 1D function
    """
    interpolated_func = 0.0
    interpolated_func = (((x-x_0)*func_1) + ((x_1-x)*func_0))/dx
    return interpolated_func

def interpolate_bilinear(x, x_0, x_1, dx, y, y_0, y_1, dy,
                  funcx0_y0, funcx1_y0, funcx0_y1, funcx1_y1):
    """
    interpolate for a discrete 2D function using 1D-interoplation (bilinear interpolation)
    """
    interpolate_y0 = interpolate1d(x,x_0,x_1,dx,funcx0_y0, funcx1_y0)
    interpolate_y1 = interpolate1d(x,x_0,x_1,dx,funcx0_y1, funcx1_y1)
    interpolated_func = interpolate1d(y,y_0,y_1,dy,interpolate_y0,interpolate_y1)
    return interpolated_func

def interpolate_triangle(x, x1, x2, x3, y, y1, y2, y3,
                        func1, func2, func3):
    """
    interpolate on a 2d triangle using barycentric coordinates (weights)
    three vertices: (x1, y1), (x2,y2), (x3,y3)
    point to interpolate on: (x, y)
    """
    # check if denominator is 0: happens when triangle is actually a line or a point
    if ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)) == 0.0:
        return func1
        
    else:
        w1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
        w2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
        w3 = 1.0 - w1 - w2
        return w1 * func1 + w2 * func2 + w3 * func3

def interpolate_value(belief, value, db):
    """
    interpolate value for an arbitrary belief given a discretised value function on a belief grid 
    """
    mult = 1/db
    
    # if belief point lies exactly on grid, find exact value
    if round_down(belief[0], 2) == round_up(belief[0], 2) and round_down(belief[1], 2) == round_up(belief[1], 2):
        return value[int(belief[0] * mult), int(belief[1] * mult)]
    
    # if the outermost interpolation point is outside the grid, barycentric interpolate on triangle
    elif round_up(belief[0], 2) + round_up(belief[1], 2) > 1.0:
        return interpolate_triangle(belief[0], round_down(belief[0], 2), 
                                    round_up(belief[0], 2), round_down(belief[0], 2),
                                    belief[1], round_down(belief[1], 2),
                                    round_down(belief[1], 2), round_up(belief[1], 2),
                             value[int(round_down(belief[0], 2)*mult), int(round_down(belief[1], 2)*mult)],
                             value[int(round_up(belief[0], 2)*mult), int(round_down(belief[1], 2)*mult)],
                             value[int(round_down(belief[0], 2)*mult), int(round_up(belief[1], 2)*mult)])
   
    # for all other points, bilinear interpolate
    else:
        return interpolate_bilinear(belief[0], round_down(belief[0], 2), round_up(belief[0], 2), db,
                             belief[1], round_down(belief[1], 2), round_up(belief[1], 2), db,
                             value[int(round_down(belief[0], 2)*mult), int(round_down(belief[1], 2)*mult)],
                             value[int(round_up(belief[0], 2)*mult), int(round_down(belief[1], 2)*mult)],
                             value[int(round_down(belief[0], 2)*mult), int(round_up(belief[1], 2)*mult)],
                             value[int(round_up(belief[0], 2)*mult), int(round_up(belief[1], 2)*mult)])
    
    
#%%
# define the pomdp 
states = np.array( [0, 1, 2] ) # (1,0), (1,1), 2
actions = np.array( [0,1,2]) #'check', 'work', 'submit'
observations = np.array( [0, 1, 2] )
# transition probabilities between states for each action 
efficacy = 0.9
noise = 0.2
discount_factor = 0.95
db = 0.01 # discretisation of belief space
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
                    [-4.0, 0.0, 4.0]])

#%%

# define value and policy variables in grid of belief states
# represent the beliefs of first N-1 states (N: total number of states)

start = time.time()

value = np.full( (int(1/db)+1, int(1/db)+1), np.nan )
policy = np.full( (int(1/db)+1, int(1/db)+1), np.nan )
b = np.arange(0, 1+db, db)

# initialise value function 
for i_b1 in range(len(b)):
    
    for i_b2 in range(len(b)):
        
        if b[i_b1] + b[i_b2] <= 1.0: value[i_b1, i_b2] = 0.0
                    
i_iter = 0 # number of value iterations
value_new = np.copy(value)

for i_b1 in range(len(b)):
    
    for i_b2 in range(len(b)):
        
        if b[i_b1] + b[i_b2] <= 1.0:
            
            belief = np.array( [b[i_b1], b[i_b2], 1-b[i_b1]-b[i_b2]] )
            
            q = np.zeros((len(actions),)) # q-value for each action
            
            for i_action, action in enumerate(actions):
                
                future_value = 0.0
                
                for i_observation, observation in enumerate(observations):
                    
                    if pO_ba(belief, action, observation, e_prob, t_prob) > 0.0:
                    
                        next_belief = get_next_belief(belief, action, observation, e_prob, t_prob)
                        if sum(np.round(next_belief, 6))>1.0: print("encountered belief > 1.0")
                        future_value = future_value + \
                                       pO_ba(belief, action, observation, e_prob, t_prob) * \
                                       interpolate_value(np.round(next_belief, 6), value, db)
                        if np.isnan(future_value): print("NaN encountered")
                q[action] = belief.T @ rewards[action, :] + discount_factor * future_value
                              
            value_new[i_b1, i_b2] = np.nanmax(q)
            policy[i_b1, i_b2] = np.nanargmax(q)
                
end = time.time()
print(end-start)
    

