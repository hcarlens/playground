# Make sure you have tensorforce installed: pip install tensorforce
import argparse
import os
import yaml
from agenttrainer import AgentTrainer, TrainingConfig, createAgent, initialiseEnvironment

def main():
    '''CLI interface to evaluate trained agents'''
    parser = argparse.ArgumentParser(description="Agent Evaluation Flags.")
    parser.add_argument("--agent_data_directory", default=None, help="Directory that contains the agent's config.yml and trained model.")
    parser.add_argument(
        "--episodes", type=int, default=None, help="Integer. Number of episodes to run.")
    parser.add_argument(
        "--render", default=None, action='store_true', help="Whether to render or not. Defaults to False.")
    parser.add_argument(
        "--opponents", default=None, help="Which agents to train against, out of simple and random. E.g. SSS = three simple agents, SRR = 1 simple and 2 random. ")
    args = parser.parse_args()

    # load the configs for the agent we'll be evaluating
    if args.agent_data_directory is not None:
        with open(args.agent_data_directory, 'r') as f:
            training_config = yaml.load(args.agent_data_directory + '/config.yml')
    else:
        print('You\'ll need to specify an agent data directory if you want me to evaluate an agent! Seems obvious now you think about it, doesn\'t it?')

    # initialise the environment and agent
    env, config = initialiseEnvironment('ffa_v0')
    agent = createAgent(training_config, env.action_space.n)
    agent.restore_model(args.agent_data_directory)

    # TODO: run the agent in the environment _episodes_ number of times, and output or store the results somewhere. 
    # probably using this: https://github.com/reinforceio/tensorforce/issues/372

if __name__ == '__main__':
    main()