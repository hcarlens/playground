# Make sure you have tensorforce installed: pip install tensorforce
import numpy as np
import sys
import argparse

from pommerman.agents import SimpleAgent, RandomAgent, PlayerAgent, BaseAgent
from pommerman.configs import ffa_v0_fast_env
from pommerman.envs.v0 import Pomme


from tensorforce.agents import PPOAgent
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
        "--episodes",
        type=int,
        default=100,
        help="Integer. Number of episodes to run.")
    parser.add_argument(
        "--render",
        default=False,
        action='store_true',
        help="Whether to render or not. Defaults to False.")

    args = parser.parse_args()

    print('Loading environment...')

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

    print('Instantiating agent...')

    if args.load_model:
        restore_directory = './models/'
        agent.restore_model(restore_directory)
        print('Model restored from', restore_directory)
    else:
        print('Creating new model with random actions...')


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
