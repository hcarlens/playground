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

def main():

    # set up tensorboard logging directory for this run
    configure('tensorboard/' + str(round(datetime.datetime.utcnow().timestamp() * 1000)))

    '''CLI interface to bootstrap taining'''
    parser = argparse.ArgumentParser(description="Playground Flags.")
    parser.add_argument("--load_model", default=False, action='store_true', help="Boolean. Load the most recent model? (otherwise it will train a new model from scratch)")
    parser.add_argument(
        "--episodes", type=int,
        default=1000,
        help="Integer. Number of episodes to run.")
    parser.add_argument(
        "--render",
        default=False,
        action='store_true',
        help="Whether to render or not. Defaults to False.")
    parser.add_argument(
        "--agent",
        default='PPO',
        help="What type of RL agent to train. Options: DQN, PPO. ")
    parser.add_argument(
        "--opponents",
        default='SSS',
        help="Which agents to train against, out of simple and random. E.g. SSS = three simple agents, SRR = 1 simple and 2 random. ")
    args = parser.parse_args()

    print('Loading environment...')

    # Instantiate the environment
    config = ffa_v0_fast_env()
    env = Pomme(**config["env_kwargs"])
    env.seed(0)
    print(env.observation_space.shape)
    agent = []
    if args.agent == 'PPO':
        # Create a Proximal Policy Optimization agent
        agent = PPOAgent( states=dict(type='float', shape=WrappedEnv.featurized_obs_shape),
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
    elif args.agent == 'DQN':
        # Create a DQN agent
        agent = DQNAgent( states=dict(type='float', shape=WrappedEnv.featurized_obs_shape),
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

    if args.load_model:
        restore_directory = './models/'
        agent.restore_model(restore_directory)
        print('Model restored from', restore_directory)
    else:
        print('Creating new model with random actions...')

    # Add agents to train against
    agents = []
    num_agents = 0
    for i in range(args.opponents.count('S')):
        agents.append(SimpleAgent(config["agent"](num_agents, config["game_type"])))
        num_agents+=1
    for i in range(args.opponents.count('R')):
        agents.append(RandomAgent(config["agent"](num_agents, config["game_type"])))
        num_agents+=1

    # Add TensorforceAgent
    agents.append(TensorforceAgent(config["agent"](3, config["game_type"])))
    env.set_agents(agents)
    env.set_training_agent(agents[-1].agent_id)
    env.set_init_game_state(None)


    # Instantiate and run the environment for 5 episodes.
    wrapped_env = WrappedEnv(env, args.render)
    runner = Runner(agent=agent, environment=wrapped_env)

    num_episodes = args.episodes
    print('Running training loop for', num_episodes, 'episodes')
    runner.run(episodes=num_episodes, max_episode_timesteps=2000)
    print("Stats: ", runner.episode_rewards, runner.episode_timesteps, runner.episode_times)
    model_directory = agent.save_model('C:/Users/Harald/Documents/Pommerman/Playground/models/')
    print("Model saved in", model_directory)

    try:
        runner.close()
    except AttributeError as e:
        pass


if __name__ == '__main__':
    main()
