from tensorforce.contrib.openai_gym import OpenAIGym
import numpy as np

class WrappedEnv(OpenAIGym):    
    featurized_obs_shape = (366,)

    def __init__(self, gym, visualize=False):
        self.gym = gym
        self.visualize = visualize
    
    def execute(self, action):
        if self.visualize:
            self.gym.render()

        actions = OpenAIGym.unflatten_action(action=action)
            
        obs = self.gym.get_observations()
        all_actions = self.gym.act(obs)
        all_actions.insert(self.gym.training_agent, actions)
        state, reward, terminal, _ = self.gym.step(all_actions)
        agent_state = featurize(state[self.gym.training_agent])
        agent_reward = reward[self.gym.training_agent]
        return agent_state, terminal, agent_reward
    
    def reset(self):
        obs = self.gym.reset()
        agent_obs = featurize(obs[3])
        return agent_obs
def make_np_float(feature):
    return np.array(feature).astype(np.float32)

def convert_bombs(bomb_map):
            '''Flatten outs the bomb array'''
            ret = []
            locations = np.where(bomb_map > 0)
            for r, c in zip(locations[0], locations[1]):
                ret.append({
                    'position': (r, c),
                    'blast_strength': int(bomb_map[(r, c)])
                })
            return ret


def featurize(obs):
    # extract relevant features from the observation space
    # expand this using logic from SimpleAgent

    board = obs["board"].reshape(-1).astype(np.float32)
    bomb_blast_strength = obs["bomb_blast_strength"].reshape(-1).astype(np.float32)
    bomb_life = obs["bomb_life"].reshape(-1).astype(np.float32)
    position = make_np_float(obs["position"])
    ammo = make_np_float([obs["ammo"]])
    blast_strength = make_np_float([obs["blast_strength"]])
    can_kick = make_np_float([obs["can_kick"]])
    teammate = obs["teammate"]
    if teammate is not None:
        teammate = teammate.value
    else:
        teammate = -1
    teammate = make_np_float([teammate])

    enemies = obs["enemies"]
    enemies = [e.value for e in enemies]
    if len(enemies) < 3:
        enemies = enemies + [-1]*(3 - len(enemies))
    enemies = make_np_float(enemies)

    # start with a simpler state space
    #return np.concatenate((board, bomb_blast_strength, bomb_life, position, ammo, blast_strength, can_kick, teammate, enemies))
    return np.concatenate((board, bomb_blast_strength, bomb_life, position, ammo))#, blast_strength, can_kick, teammate, enemies))