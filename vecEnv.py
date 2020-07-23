
class VecEnv:
    def __init__(self, observation_space, state_space, action_space):
        self.observation_space = observation_space
        self.state_space = state_space
        self.action_space = action_space

    def reset(self, z, skel):
        pass

    def step_norm(self, actions, z, skel):
        pass

    def step(self, actions, z, skel):
        self.step_norm(actions, z, skel)
        pass
