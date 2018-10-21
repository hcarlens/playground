# Make sure you have tensorforce installed: pip install tensorforce
import argparse

from pommerman.agents import SimpleAgent, RandomAgent, BaseAgent
from pommerman.configs import ffa_v0_fast_env
from pommerman.envs.v0 import Pomme


from tensorforce.agents import PPOAgent, DQNAgent
from tensorforce.execution import Runner
from tensorboard_logger import configure
import datetime
from wrappedenv import featurize, WrappedEnv

class TensorforceAgent(BaseAgent):
    def act(self, obs, action_space):
        pass

class TrainingConfig:
    def __init__(self, rl_agent, num_episodes, opponents, render, load_most_recent_model, discount):
        self.rl_agent = rl_agent if rl_agent else 'DQN'
        self.num_episodes = num_episodes if num_episodes else 1000
        self.opponents = opponents if opponents else 'SSS'
        self.render = render if render else False
        self.discount = discount if discount else 0.99
        self.load_most_recent_model = load_most_recent_model if load_most_recent_model else False

def main():

    '''CLI interface to bootstrap taining'''
    parser = argparse.ArgumentParser(description="Playground Flags.")
    parser.add_argument("--load_model", default=False, action='store_true', help="Boolean. Load the most recent model? (otherwise it will train a new model from scratch)")
    parser.add_argument(
        "--episodes", type=int,
        default=None,
        help="Integer. Number of episodes to run.")
    parser.add_argument(
        "--render",
        default=None,
        action='store_true',
        help="Whether to render or not. Defaults to False.")
    parser.add_argument(
        "--agent",
        default=None,
        help="What type of RL agent to train. Options: DQN, PPO. ")
    parser.add_argument(
        "--opponents",
        default=None,
        help="Which agents to train against, out of simple and random. E.g. SSS = three simple agents, SRR = 1 simple and 2 random. ")
    parser.add_argument(
        "--discount",
        default=None,
        help="Gamma parameter, defining how much to value future timesteps vs current timesteps.")
    args = parser.parse_args()

    # create an object to define this training run. Args loaded from CLI, but can also be loaded from config.
    training_config = TrainingConfig(rl_agent=args.agent, num_episodes=args.episodes, opponents=args.opponents,
    render=args.render, load_most_recent_model=args.load_model, discount=args.discount)

    # set up tensorboard logging directory for this run
    configure('tensorboard/' + str(round(datetime.datetime.utcnow().timestamp() * 1000)))

    print('Loading environment...')

    # Instantiate the environment
    config = ffa_v0_fast_env()
    env = Pomme(**config["env_kwargs"])
    env.seed(0)
    print(env.observation_space.shape)
    agent = []
    if training_config.rl_agent == 'PPO':
        # Create a Proximal Policy Optimization agent
        agent = PPOAgent(states=dict(type='float', shape=WrappedEnv.featurized_obs_shape),
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
    elif training_config.rl_agent == 'DQN':
        # Create a DQN agent
        agent = DQNAgent(states=dict(type='float', shape=WrappedEnv.featurized_obs_shape),
            actions=dict(type='int', num_actions=env.action_space.n),
            network=[
                dict(type='dense', size=64),
                dict(type='dense', size=64)
            ],
            double_q_model=False,
            target_sync_frequency=10000,
            discount=0.99
            )


    print('Instantiating agent...')

    if training_config.load_most_recent_model:
        restore_directory = './models/'
        agent.restore_model(restore_directory)
        print('Model restored from', restore_directory)
    else:
        print('Creating new model with random actions...')

    # Add agents to train against
    agents = []
    num_agents = 0
    for i in range(training_config.opponents.count('S')):
        agents.append(SimpleAgent(config["agent"](num_agents, config["game_type"])))
        num_agents+=1
    for i in range(training_config.opponents.count('R')):
        agents.append(RandomAgent(config["agent"](num_agents, config["game_type"])))
        num_agents+=1

    # Add TensorforceAgent
    agents.append(TensorforceAgent(config["agent"](3, config["game_type"])))
    env.set_agents(agents)
    env.set_training_agent(agents[-1].agent_id)
    env.set_init_game_state(None)


    # Instantiate and run the environment for 5 episodes.
    wrapped_env = WrappedEnv(env, training_config.render)
    runner = Runner(agent=agent, environment=wrapped_env)

    num_episodes = training_config.num_episodes
    print('Running training loop for', num_episodes, 'episodes')
    runner.run(episodes=num_episodes, max_episode_timesteps=2000)
    print("Stats: ", runner.episode_rewards, runner.episode_timesteps, runner.episode_times)
    model_directory = agent.save_model('./models/')
    print("Model saved in", model_directory)

    try:
        runner.close()
    except AttributeError as e:
        pass


if __name__ == '__main__':
    main()
