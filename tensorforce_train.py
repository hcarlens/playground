# Make sure you have tensorforce installed: pip install tensorforce
import argparse
import yaml
import os

from pommerman.agents import SimpleAgent, RandomAgent, BaseAgent
from pommerman.configs import ffa_v0_fast_env
from pommerman.envs.v0 import Pomme


from tensorforce.agents import PPOAgent, DQNAgent
from tensorforce.execution import Runner
import tensorboard_logger 
import datetime
from wrappedenv import featurize, WrappedEnv

class TensorforceAgent(BaseAgent):
    def act(self, obs, action_space):
        pass

class TrainingConfig:
    # defaults are specified in this awkward way so we can take None inputs from CLI and override them
    def __init__(self, rl_agent=None, num_episodes=None, opponents=None, render=None,
            load_most_recent_model=None, discount=None, variable_noise=None, neural_net=None,
            batching_capacity=None, double_q_model=None, target_sync_frequency=None,
            optimizer_type=None, optimizer_lr=None):
        self.rl_agent = rl_agent if rl_agent else 'DQN'
        self.num_episodes = num_episodes if num_episodes else 1000
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
        self.load_most_recent_model = load_most_recent_model if load_most_recent_model else False
        self.optimizer_type = optimizer_type if optimizer_type else 'adam'
        self.optimizer_lr = optimizer_lr if optimizer_lr else 1e-3

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
    parser.add_argument(
        "--variable_noise",
        default=None,
        help="Standard deviation of noise to add to parameter space (see NoisyNets paper).")
    args = parser.parse_args()

    # create an object to define this training run. Args loaded from CLI, but can also be loaded from config.
    training_config = TrainingConfig(rl_agent=args.agent, num_episodes=args.episodes, opponents=args.opponents,
    render=args.render, load_most_recent_model=args.load_model, discount=args.discount, variable_noise=args.variable_noise)

    # set up tensorboard logging directory for this run
    log_directory = 'data/' + datetime.datetime.now().strftime('%d_%m/%H_%M_%S') + '-' + training_config.rl_agent + '-' + training_config.opponents
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    tensorboard_logger.configure(log_directory)

    # add config file to logging directory, so we know what settings this run used
    with open(log_directory + '/config.yml', 'w+') as outfile:
        yaml.dump(training_config, outfile, default_flow_style=False)

    print('Loading environment...')

    # Instantiate the environment
    config = ffa_v0_fast_env()
    env = Pomme(**config["env_kwargs"])
    env.seed(0)
    agent = []
    if training_config.rl_agent == 'PPO':
        # Create a Proximal Policy Optimization agent
        agent = PPOAgent(states=dict(type='float', shape=WrappedEnv.featurized_obs_shape),
            actions=dict(type='int', num_actions=env.action_space.n),
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
        agent = DQNAgent(states=dict(type='float', shape=WrappedEnv.featurized_obs_shape),
            actions=dict(type='int', num_actions=env.action_space.n),
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
