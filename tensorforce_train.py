# Make sure you have tensorforce installed: pip install tensorforce
import numpy as np
import sys

from pommerman.agents import SimpleAgent, RandomAgent, PlayerAgent, BaseAgent
from pommerman.configs import ffa_v0_fast_env
from pommerman.envs.v0 import Pomme


from tensorforce.agents import PPOAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

class WrappedEnv(OpenAIGym):    
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

    return np.concatenate((board, bomb_blast_strength, bomb_life, position, ammo, blast_strength, can_kick, teammate, enemies))

class TensorforceAgent(BaseAgent):
    def act(self, obs, action_space):
        pass

def main():

    # Instantiate the environment
    config = ffa_v0_fast_env()
    env = Pomme(**config["env_kwargs"])
    env.seed(0)

    # Create a Proximal Policy Optimization agent
    agent = PPOAgent(
        states=dict(type='float', shape=env.observation_space.shape),
        actions=dict(type='int', num_actions=env.action_space.n),
        network=[
            dict(type='dense', size=64),
            dict(type='dense', size=64)
        ],
        batching_capacity=1000,
        step_optimizer=dict(
            type='adam',
            learning_rate=1e-4
        )
    )

    if len(sys.argv) > 1 and sys.argv[1] == 'load':
        restore_directory = 'C:/Users/Harald/Documents/Pommerman/Playground/models/'
        agent.restore_model(restore_directory)
        print('Model restored from', restore_directory)

    # Add agents to train against
    agents = []
    agents.append(SimpleAgent(config["agent"](0, config["game_type"])))
    agents.append(RandomAgent(config["agent"](1, config["game_type"])))
    agents.append(RandomAgent(config["agent"](2, config["game_type"])))

    # Add TensorforceAgent
    agents.append(TensorforceAgent(config["agent"](3, config["game_type"])))
    env.set_agents(agents)
    env.set_training_agent(agents[-1].agent_id)
    env.set_init_game_state(None)


    # Instantiate and run the environment for 5 episodes.
    wrapped_env = WrappedEnv(env, True)
    runner = Runner(agent=agent, environment=wrapped_env)
    runner.run(episodes=3, max_episode_timesteps=2000)
    print("Stats: ", runner.episode_rewards, runner.episode_timesteps, runner.episode_times)
    model_directory = agent.save_model('C:/Users/Harald/Documents/Pommerman/Playground/models/')
    print("Model saved in", model_directory)

    try:
        runner.close()
    except AttributeError as e:
        pass


if __name__ == '__main__':
    main()
