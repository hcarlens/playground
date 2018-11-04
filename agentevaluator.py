# Make sure you have tensorforce installed: pip install tensorforce
import argparse
import os
import yaml
from agenttrainer import AgentTrainer, TrainingConfig, createAgent, initialiseEnvironment
from pommerman.agents import SimpleAgent, RandomAgent, BaseAgent

from tensorforce.agents import PPOAgent, DQNAgent
from tensorforce.execution import Runner
import tensorboard_logger
from wrappedenv import WrappedEnv

class TensorforceAgent(BaseAgent):
    def act(self, obs, action_space):
        pass

def main():
    '''CLI interface to evaluate trained agents'''
    parser = argparse.ArgumentParser(description="Agent Evaluation Flags.")
    parser.add_argument("--agent_data_directory", default=None, required=True, help="Directory that contains the agent's config.yml and trained model.")
    parser.add_argument("--outfile", default=None, required=True, help="Path to file to write results to.")
    parser.add_argument(
        "--episodes", type=int, default=100, help="Integer. Number of episodes to run.")
    parser.add_argument(
        "--max_episode_timesteps", type=int, default=100, help="Integer. Max number of timesteps per episode.")
    parser.add_argument(
        "--render", default=None, action='store_true', help="Whether to render or not. Defaults to False.")
    parser.add_argument(
        "--opponents", default="SSS", help="Which agents to train against, out of simple and random. E.g. SSS = three simple agents, SRR = 1 simple and 2 random. ")
    args = parser.parse_args()

    # load the configs for the agent we'll be evaluating
    with open(args.agent_data_directory + '/config.yml', 'r') as f:
        training_config = yaml.load(f)

    # initialise the environment and agent
    env, config = initialiseEnvironment('ffa_v0')
    agent = createAgent(training_config, env.action_space.n)
    agent.restore_model(args.agent_data_directory)

    # Add agents to test against
    agents = []
    num_agents = 0
    for i in range(args.opponents.count('S')):
        agents.append(SimpleAgent(config["agent"](num_agents, config["game_type"])))
        num_agents += 1
    for i in range(args.opponents.count('R')):
        agents.append(RandomAgent(config["agent"](num_agents, config["game_type"])))
        num_agents += 1

    # Add TensorforceAgent
    agents.append(TensorforceAgent(config["agent"](3, config["game_type"])))
    env.set_agents(agents)
    env.set_training_agent(agents[-1].agent_id)
    env.set_init_game_state(None)

    wrapped_env = WrappedEnv(env, training_config.render)
    runner = Runner(agent, wrapped_env)
    tensorboard_logger.configure("./logs")

    print('Running training loop for', args.episodes, 'episodes')
    runner.run(episodes=args.episodes, testing=True, max_episode_timesteps=args.max_episode_timesteps)
    print("Stats: ", runner.episode_rewards, runner.episode_timesteps)

    episode_scores = []
    for (i, reward) in enumerate(runner.episode_rewards):
        if reward == -1:
            episode_scores.append(0)
        else:
            episode_scores.append(reward)

    fraction_won = sum(episode_scores) / args.episodes
    percent_won = (fraction_won * 100)

    with open(args.outfile, 'a') as f:
        line = ("{dir}\t{won}%\t{discount}\t{optimizer_lr}\t{optimizer_type}\t{double_q}\t{variable_noise}\t{net_layer_1}\t{net_layer_2}\t{runs}\n").format(
            dir=args.agent_data_directory, won=percent_won,
            discount=training_config.discount,
            optimizer_lr=training_config.optimizer_lr,
            optimizer_type=training_config.optimizer_type,
            double_q=training_config.double_q_model,
            variable_noise=training_config.variable_noise,
            net_layer_1=training_config.neural_net[0]["type"]+str(training_config.neural_net[0]["size"]),
            net_layer_2=training_config.neural_net[1]["type"]+str(training_config.neural_net[1]["size"]),
            runs=args.episodes
        )
        f.write(line)

    try:
        runner.close()
    except AttributeError as e:
        pass

if __name__ == '__main__':
    main()
