import pomdp_py
import random

class State(pomdp_py.State):
    def __init__(self, name):
        if name != "tiger-left" and name != "tiger-right":
            raise ValueError("Invalid state: %s" % name)
        self.name = name
        
class Action(pomdp_py.Action):
    def __init__(self, name):
        if name != "open-left" and name != "open-right"\
           and name != "listen":
            raise ValueError("Invalid action: %s" % name)
        self.name = name
        
class Observation(pomdp_py.Observation):
    def __init__(self, name):
        if name != "tiger-left" and name != "tiger-right":
            raise ValueError("Invalid action: %s" % name)
        self.name = name
        
class ObservationModel(pomdp_py.ObservationModel):
    def __init__(self, noise=0.15):
        self.noise = noise

    def probability(self, observation, next_state, action):
        if action.name == "listen":
            # heard the correct growl
            if observation.name == next_state.name:
                return 1.0 - self.noise
            else:
                return self.noise
        else:
            return 0.5

    def sample(self, next_state, action):
        if action.name == "listen":
            thresh = 1.0 - self.noise
        else:
            thresh = 0.5

        if random.uniform(0,1) < thresh:
            return Observation(next_state.name)
        else:
            return Observation(next_state.other().name)

    def get_all_observations(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation
        space (e.g. value iteration)"""
        return [Observation(s)
                for s in {"tiger-left", "tiger-right"}]
    
class TransitionModel(pomdp_py.TransitionModel):
    def probability(self, next_state, state, action):
        """According to problem spec, the world resets once
        action is open-left/open-right. Otherwise, it
        stays the same"""
        if action.name.startswith("open"):
            return 0.5
        else:
            if next_state.name == state.name:
                return 1.0 - 1e-9
            else:
                return 1e-9

    def sample(self, state, action):
        if action.name.startswith("open"):
            return random.choice(self.get_all_states())
        else:
            return State(state.name)

    def get_all_states(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the
        observation space (e.g. value iteration)"""
        return [State(s) for s in {"tiger-left", "tiger-right"}]
    
    
class RewardModel(pomdp_py.RewardModel):
    def _reward_func(self, state, action):
        if action.name == "open-left":
            if state.name == "tiger-right":
                return 10
            else:
                return -100
        elif action.name == "open-right":
            if state.name == "tiger-left":
                return 10
            else:
                return -100
        else: # listen
            return -1

    def sample(self, state, action, next_state):
        # deterministic
        return self._reward_func(state, action)
    
    
class TigerProblem(pomdp_py.POMDP):

    def __init__(self, obs_noise, init_true_state):
        """init_belief is a Distribution."""
        agent = pomdp_py.Agent(TransitionModel(),
                               ObservationModel(obs_noise),
                               RewardModel())
        env = pomdp_py.Environment(init_true_state,
                                   TransitionModel(),
                                   RewardModel())
        super().__init__(agent, env, name="TigerProblem")
