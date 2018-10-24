import argparse
import os
import datetime
import yaml

from agenttrainer import TrainingConfig

def main():
    '''CLI to generate configs for different settings to try'''
    parser = argparse.ArgumentParser(description="Config generator training flags.")
    parser.add_argument(
        "--config_directory", default='configs', help="Location to store generated config files")
    parser.add_argument(
        "--episodes", type=int, default=3000, help="Integer. Number of episodes to run.")
    parser.add_argument(
        "--opponents", default=None, help="Which agents to train against, out of simple and random. E.g. SSS = three simple agents, SRR = 1 simple and 2 random. ")
    args = parser.parse_args()

    rl_agents = ['PPO', 'DQN']
    optimizer_types = ['adam', 'rmsprop']
    optimizer_lrs = [1e-3, 1e-4, 1e-5]
    neural_nets = [
        [dict(type='dense', size=64), dict(type='dense', size=64)],
        [dict(type='dense', size=20), dict(type='dense', size=20)]
        ]
    discounts = [0.9, 0.99, 1]
    variable_noises = [None, 1, 10]
    target_sync_frequencies = [1000, 10000]
    # also consider trying: different types of memory, batching capacities

    current_config_num = 0

    # create loads of config files!
    for optimizer_type in optimizer_types:
        for optimizer_lr in optimizer_lrs:
            for neural_net in neural_nets:
                for discount in discounts:
                    for variable_noise in variable_noises:
                        for rl_agent in rl_agents:
                            if rl_agent == 'DQN':
                                for target_sync_frequency in target_sync_frequencies: # this only applies to DQNs
                                    # create config file
                                    tc = TrainingConfig(rl_agent=rl_agent, optimizer_type=optimizer_type, optimizer_lr=optimizer_lr,
                                    neural_net=neural_net, discount=discount, variable_noise=variable_noise, num_episodes=args.episodes,
                                    opponents=args.opponents, target_sync_frequency=target_sync_frequency)

                                    # write config file
                                    with open(args.config_directory + '/config_' + str(current_config_num) + '.yml', 'w+') as outfile:
                                        yaml.dump(tc, outfile, default_flow_style=False)
                                        print('Config', current_config_num, 'done.')
                                        current_config_num += 1
                            else:
                                # create config file
                                tc = TrainingConfig(rl_agent=rl_agent, optimizer_type=optimizer_type, optimizer_lr=optimizer_lr,
                                neural_net=neural_net, discount=discount, variable_noise=variable_noise, num_episodes=args.episodes,
                                opponents=args.opponents)
                                # write config file
                                with open(args.config_directory + '/config_' + str(current_config_num) + '.yml', 'w+') as outfile:
                                    yaml.dump(tc, outfile, default_flow_style=False)
                                    print('Config', current_config_num, 'done.')
                                    current_config_num += 1



if __name__ == '__main__':
    main()