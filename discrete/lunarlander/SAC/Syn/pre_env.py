import gymnasium as gym



class TimeLimit(gym.Wrapper):
    """
    限制最大步数
    """
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, ac):
        observation, reward, done, truncated,info = self.env.step(ac)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
            info['TimeLimit.truncated'] = True
        return observation, reward, done,truncated,info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)
    
    
    
    
    
def ProcessEnv(env):
    
    env = TimeLimit(env,env.spec.max_episode_steps)
    
    return env