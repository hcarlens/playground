from tensorforce.contrib.openai_gym import OpenAIGym
import numpy as np
from pommerman import constants
from pommerman.agents import SimpleAgent

class WrappedEnv(OpenAIGym):
    featurized_obs_shape = [(366,), (372,), (134,), (1, 11, 11), (7, 11, 11), (1098,)] # index in this array indicates the feature space version

    def __init__(self, gym, feature_version=0, visualize=False):
        self.gym = gym
        self.visualize = visualize
        self.feature_version = feature_version
        self.simple_agent = SimpleAgent()
    
    def execute(self, action):
        if self.visualize:
            self.gym.render()

        actions = OpenAIGym.unflatten_action(action=action)
            
        obs = self.gym.get_observations()
        all_actions = self.gym.act(obs)
        all_actions.insert(self.gym.training_agent, actions)
        state, reward, terminal, _ = self.gym.step(all_actions)
        agent_state = featurize(state[self.gym.training_agent], self.gym._game_type, self.simple_agent, self.gym.action_space, self.feature_version)
        agent_reward = reward[self.gym.training_agent]
        return agent_state, terminal, agent_reward
    
    def reset(self):
        obs = self.gym.reset()
        agent_obs = featurize(obs[3], self.gym._game_type, self.simple_agent, self.gym.action_space, self.feature_version)
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


def featurize(obs, game_type, simple_agent, action_space, version=0):
    # extract relevant features from the observation space
    # expand this using logic from SimpleAgent

    board = obs["board"].reshape(-1).astype(np.float32)
    board2d = obs["board"].copy()
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
    else:
        # give our own agent a set identity
        np.place(board, board == own_identity, 7)
        np.place(board2d, board2d == own_identity, 7)
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
            np.place(board2d, board2d == enemies[0], -1)
            np.place(board2d, board2d == enemies[1], -2)
            np.place(board2d, board2d == teammate, 14)
        
        passage_board = np.zeros((board.shape))
        rigid_board = np.zeros((board.shape))
        wood_board = np.zeros((board.shape))
        bomb_board = np.zeros((board.shape))
        powerup_board = np.zeros((board.shape))
        enemy_board = np.zeros((board.shape))
        us_board = np.zeros((board.shape))

        if version <= 3:
            # normalise the inputs
            board = board/10
            board2d = board2d/10
            ammo = ammo/10
            blast_strength = blast_strength/10
            bomb_blast_strength = bomb_blast_strength/10
            bomb_life = bomb_life/10
        elif version == 4:
            # skip normalization and one-hot encode the board
            board = obs["board"].copy()

            # make a bunch of empty boards
            
            # populate those boards with 1s in the position of relevant elements,
            # using the main board as a source
            multi_place(passage_board, board,
                constants.Item.Passage.value, constants.Item.Flames.value)
            np.place(rigid_board, board == constants.Item.Rigid.value, 1)
            np.place(wood_board, board == constants.Item.Wood.value, 1)
            np.place(bomb_board, board == constants.Item.Bomb.value, 1)
            multi_place(powerup_board, board,
                constants.Item.ExtraBomb.value, constants.Item.IncrRange.value, constants.Item.Kick.value)
            multi_place(enemy_board, board,
                constants.Item.AgentDummy.value,
                constants.Item.Agent0.value,
                constants.Item.Agent1.value,
                constants.Item.Agent2.value)
            np.place(us_board, board == constants.Item.Agent3.value, 1)
        elif version == 5:
            # skip normalization and one-hot encode the board
            board = obs["board"].copy()

            # compress the number of discrete elements to encode.
            # passage, rigid, wood, bomb stay as they are (0,1,2,3)
            passage = 0 # from 0,4 (flames are only dangerous for one timestep)
            powerup = 4 # from 6,7,8
            enemy = 5 # from 9,10,11,12
            us = 6 # from 13
            board_max_value = 6

            np.place(board, board == 4, passage)
            np.place(board, board == 6, powerup)
            np.place(board, board == 7, powerup)
            np.place(board, board == 8, powerup)
            np.place(board, board == 9, enemy)
            np.place(board, board == 10, enemy)
            np.place(board, board == 11, enemy)
            np.place(board, board == 12, enemy)
            np.place(board, board == 13, us)

            board = np.eye(board_max_value + 1)[board]

        # placeholder
        # replace this with logic from simpleagent to indicate whether directions are safe or not
        safe_action_heuristics = [0] * 6
        # action = simple_agent.act(obs, action_space)
        # safe_action_heuristics[action] = 1

        if version == 1:
            neural_net_input = np.concatenate((board, bomb_blast_strength, bomb_life, ammo, blast_strength, can_kick, safe_action_heuristics))
        elif version == 2:
            neural_net_input = np.concatenate((board, np.array(bomb_blast_strength.min(), ndmin=1), np.array(bomb_blast_strength.max(), ndmin=1), np.array(bomb_life.min(), ndmin=1), np.array(bomb_life.max(), ndmin=1), ammo, blast_strength, can_kick, safe_action_heuristics))
        elif version == 3:
            neural_net_input = np.expand_dims(board2d, axis=0)
        elif version == 4:
            # ConvNet
            neural_net_input = np.array([passage_board, rigid_board, wood_board, bomb_board, powerup_board, enemy_board, us_board])
        elif version == 5:
            # one-hot encoding
            neural_net_input = np.concatenate((
                board.reshape(-1).astype(np.float32),
                bomb_blast_strength.reshape(-1).astype(np.float32),
                bomb_life.reshape(-1).astype(np.float32),
                ammo.reshape(-1).astype(np.float32),
                blast_strength.reshape(-1).astype(np.float32),
                can_kick.reshape(-1).astype(np.float32),
                safe_action_heuristics
            ))
        else:
            raise "Config version not supported"

    return neural_net_input


# Similar to numpy.place, but accepts several mask inputs,
# and places a 1 in the matching position in the target.
def multi_place(target, source, *masks):
    for i in range(len(masks)):
        np.place(target, source == masks[i], 1)