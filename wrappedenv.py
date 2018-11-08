from tensorforce.contrib.openai_gym import OpenAIGym
import numpy as np
from pommerman import constants

class WrappedEnv(OpenAIGym):    
    featurized_obs_shape = [(366,), (372,)] # index in this array indicates the feature space version

    def __init__(self, gym, feature_version=0, visualize=False):
        self.gym = gym
        self.visualize = visualize
        self.feature_version = feature_version
    
    def execute(self, action):
        if self.visualize:
            self.gym.render()

        actions = OpenAIGym.unflatten_action(action=action)
            
        obs = self.gym.get_observations()
        all_actions = self.gym.act(obs)
        all_actions.insert(self.gym.training_agent, actions)
        state, reward, terminal, _ = self.gym.step(all_actions)
        agent_state = featurize(state[self.gym.training_agent], self.gym._game_type, self.feature_version)
        agent_reward = reward[self.gym.training_agent]
        return agent_state, terminal, agent_reward
    
    def reset(self):
        obs = self.gym.reset()
        agent_obs = featurize(obs[3], self.gym._game_type, self.feature_version)
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


def featurize(obs, game_type, version=0):
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
    own_identity = obs['board'][int(position[0])][int(position[1])] 

    enemies = obs["enemies"]
    enemies = [e.value for e in enemies]

    # re-map all power-ups to the same value, to simplify things for now
    np.place(board, np.isin(board, [7, 8]), 6)

    if version == 0:
        if len(enemies) < 3:
            enemies = enemies + [-1]*(3 - len(enemies))
        enemies = make_np_float(enemies)
        neural_net_input = np.concatenate((board, bomb_blast_strength, bomb_life, position, ammo))#, blast_strength, can_kick, teammate, enemies))
    elif version == 1:
        # give our own agent a set identity
        np.place(board, board == own_identity, 7)
        # if we're in the team game, identify individual enemies
        # in the ffa game, give all enemies a shared identity (for faster learning)
        if game_type == constants.GameType.FFA:
            # in FFA, label all enemies '9'
            for i in enemies:
                np.place(board, board == i, 9)
        elif game_type == constants.GameType.Team:
            # in the team game, give enemies identities of '-1' and '-2', and give our teammate identity '14'
            enemies.sort()
            np.place(board, board == enemies[0], -1)
            np.place(board, board == enemies[1], -2)
            np.place(board, board == teammate, 14)

        # normalise the inputs
        board = board/10
        ammo = ammo/10
        blast_strength = blast_strength/10
        bomb_blast_strength = bomb_blast_strength/10
        bomb_life = bomb_life/10

        # placeholder
        # replace this with logic from simpleagent to indicate whether directions are safe or not
        safe_action_heuristics = [0] * 6

        neural_net_input = np.concatenate((board, bomb_blast_strength, bomb_life, ammo, blast_strength, can_kick, safe_action_heuristics))

    return neural_net_input
    