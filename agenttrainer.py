# Make sure you have tensorforce installed: pip install tensorforce
import argparse
import os
import datetime
import yaml

from pommerman.agents import SimpleAgent, RandomAgent, BaseAgent
from pommerman.configs import ffa_v0_fast_env, team_competition_env
from pommerman.envs.v0 import Pomme

from tensorforce.agents import PPOAgent, DQNAgent
from tensorforce.execution import Runner
import tensorboard_logger
from wrappedenv import WrappedEnv

class TensorforceAgent(BaseAgent):
    def act(self, obs, action_space):
        pass

class TrainingConfig:
    """
    This class defines a single training run. 
    """
    # defaults are specified in this awkward way so we can take None inputs from CLI and override them
    def __init__(self, rl_agent=None, num_episodes=None, opponents=None, render=None,
            model_directory=None, discount=None, variable_noise=None, neural_net=None,
            batching_capacity=None, double_q_model=None, target_sync_frequency=None,
            optimizer_type=None, optimizer_lr=None, max_episode_timesteps=None,
            environment=None):
        self.rl_agent = rl_agent if rl_agent else 'DQN'
        self.num_episodes = num_episodes if num_episodes else 10000
        self.opponents = opponents if opponents else 'SSS'
        self.render = render if render else False
        self.discount = discount if discount else 0.99
        self.variable_noise = variable_noise if variable_noise else None
        self.batching_capacity = batching_capacity if batching_capacity else 1000
        self.double_q_model = double_q_model if double_q_model else True
        self.target_sync_frequency = target_sync_frequency if target_sync_frequency  else 10000
        self.neural_net = neural_net if neural_net else [
                dict(type='dense', size=64),
                dict(type='dense', size=64)
            ]
        self.model_directory = model_directory if model_directory else False
        self.optimizer_type = optimizer_type if optimizer_type else 'adam'
        self.optimizer_lr = optimizer_lr if optimizer_lr else 1e-3
        self.max_episode_timesteps = max_episode_timesteps if max_episode_timesteps else 2000
        self.environment = environment if environment else 'FFA'

def createAgent(training_config, action_space_dim):
    """ Create an agent based on a set of configs """
    if training_config.rl_agent == 'PPO':
        # Create a Proximal Policy Optimization agent
        return PPOAgent(states=dict(type='float', shape=WrappedEnv.featurized_obs_shape),
            actions=dict(type='int', num_actions=action_space_dim),
            network=training_config.neural_net,
            batching_capacity=training_config.batching_capacity,
            step_optimizer=dict(
                type=training_config.optimizer_type,
                learning_rate=training_config.optimizer_lr
            ),
            discount=training_config.discount
        )
    elif training_config.rl_agent == 'DQN':
        # Create a DQN agent
        return DQNAgent(states=dict(type='float', shape=WrappedEnv.featurized_obs_shape),
            actions=dict(type='int', num_actions=action_space_dim),
            network=training_config.neural_net,
            batching_capacity=training_config.batching_capacity,
            double_q_model=training_config.double_q_model,
            target_sync_frequency=training_config.target_sync_frequency,
            discount=training_config.discount,
            optimizer=dict(
                type=training_config.optimizer_type,
                learning_rate=training_config.optimizer_lr
            )
            )

def initialiseEnvironment(environment_type):
    config = None
    if environment_type.lower() == 'ffa':
        config = ffa_v0_fast_env()
    elif environment_type.lower() == 'team':
        config = team_competition_env()
    env = Pomme(**config["env_kwargs"])
    env.seed(0)
    return env, config


class AgentTrainer:
    """
    This class deals with setting up a training run from a config file, and running it.
    """
    def __init__(self, training_config):
        if training_config.model_directory:
            self.data_directory = training_config.model_directory
        else:
            self.data_directory = 'data/' + datetime.datetime.now().strftime('%d_%m/%H_%M_%S.%f') + '-' + training_config.rl_agent + '-' + training_config.opponents + '/'

        # set up tensorboard logging directory for this run
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)
        tensorboard_logger.configure(self.data_directory)

        # add config file to logging directory, so we know what settings this run used
        with open(self.data_directory + 'config.yml', 'w+') as outfile:
            yaml.dump(training_config, outfile, default_flow_style=False)
        print('Loading environment...')

        env, config = initialiseEnvironment(training_config.environment)

        self.agent = createAgent(training_config, env.action_space.n)

        print('Instantiating agent...')

        if training_config.model_directory:
            self.agent.restore_model(training_config.model_directory)
            print('Model restored from', training_config.model_directory)
        else:
            print('Creating new model with random actions...')

        # Add agents to train against
        agents = []
        num_agents = 0
        for i in range(training_config.opponents.count('S')):
            agents.append(SimpleAgent(config["agent"](num_agents, config["game_type"])))
            num_agents += 1
        for i in range(training_config.opponents.count('R')):
            agents.append(RandomAgent(config["agent"](num_agents, config["game_type"])))
            num_agents += 1

        # Add TensorforceAgent
        agents.append(TensorforceAgent(config["agent"](3, config["game_type"])))
        env.set_agents(agents)
        env.set_training_agent(agents[-1].agent_id)
        env.set_init_game_state(None)

        self.env = env
        self.training_config = training_config

    def run(self):
        # Instantiate and run the environment.
        wrapped_env = WrappedEnv(self.env, self.training_config.render)
        runner = Runner(agent=self.agent, environment=wrapped_env)

        num_episodes = self.training_config.num_episodes
        print('Running training loop for', num_episodes, 'episodes')
        runner.run(episodes=num_episodes, max_episode_timesteps=self.training_config.max_episode_timesteps)
        print("Stats: ", runner.episode_rewards, runner.episode_timesteps, runner.episode_times)
        model_directory = self.agent.save_model(self.data_directory)
        print("Model saved in", model_directory)

        try:
            runner.close()
        except AttributeError as e:
            pass

def main():
    '''CLI interface to bootstrap taining'''
    parser = argparse.ArgumentParser(description="Agent Training Flags.")
    parser.add_argument("--model_directory", default=None, help="Directory from which to load an existing model. ")
    parser.add_argument(
        "--environment", default=None, help="FFA or Team. (not case sensitive)")
    parser.add_argument(
        "--config_file", default=None, help="Yaml config file from which to load training_config settings")
    parser.add_argument(
        "--episodes", type=int, default=None, help="Integer. Number of episodes to run.")
    parser.add_argument(
        "--render", default=None, action='store_true', help="Whether to render or not. Defaults to False.")
    parser.add_argument(
        "--agent", default=None, help="What type of RL agent to train. Options: DQN, PPO. ")
    parser.add_argument(
        "--opponents", default=None, help="Which agents to train against, out of simple and random. E.g. SSS = three simple agents, SRR = 1 simple and 2 random. ")
    parser.add_argument(
        "--discount", default=None, help="Gamma parameter, defining how much to value future timesteps vs current timesteps.")
    parser.add_argument(
        "--variable_noise", default=None, help="Standard deviation of noise to add to parameter space (see NoisyNets paper).")
    args = parser.parse_args()

    # create an object to define this training run. Args loaded from CLI, but can also be loaded from config.
    if args.config_file is not None:
        with open(args.config_file, 'r') as f:
            training_config = yaml.load(f)
    else:
        training_config = TrainingConfig(rl_agent=args.agent, num_episodes=args.episodes, opponents=args.opponents,
            render=args.render, model_directory=args.model_directory, discount=args.discount, variable_noise=args.variable_noise, environment=args.environment)

    agent_trainer = AgentTrainer(training_config)
    agent_trainer.run()

if __name__ == '__main__':
    main()