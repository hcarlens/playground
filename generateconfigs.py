import argparse
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
    optimizer_types = ['rmsprop'] # from DQN paper
    optimizer_lrs = [0.00025] # from DQN paper
    neural_nets = [
        [dict(type='dense', size=200), dict(type='dense', size=200)],
        [dict(type='dense', size=200, l2_regularization=0.001), dict(type='dense', size=200, l2_regularization=0.001)], # l2 regularisation from https://github.com/lefnire/tforce_btc_trader/blob/master/hypersearch.py
        [dict(type='dense', size=200, l2_regularization=0.001), dict(type='dropout', rate=0.1),dict(type='dense', size=200, l2_regularization=0.001), dict(type='dropout', rate=0.1)] 
        ]
    discounts = [0.99] # from DQN paper
    variable_noises = [None]
    forward_models = ['original', 'firsttodie']
    target_sync_frequencies = [10000] # from DQN paper
    batching_capacities = [32, 1000] # 32 from DQN paper
    feature_versions = [1]
    # also consider trying: different types of memory, batching capacities

    current_config_num = 0

    for feature_version in feature_versions:
        for optimizer_type in optimizer_types:
            for optimizer_lr in optimizer_lrs:
                for neural_net in neural_nets:
                    for discount in discounts:
                        for variable_noise in variable_noises:
                            for forward_model in forward_models:
                                for rl_agent in rl_agents:
                                    if rl_agent == 'DQN':
                                        for target_sync_frequency in target_sync_frequencies: # this only applies to DQNs
                                            # create config file
                                            tc = TrainingConfig(rl_agent=rl_agent, optimizer_type=optimizer_type, optimizer_lr=optimizer_lr,
                                            neural_net=neural_net, discount=discount, variable_noise=variable_noise, num_episodes=args.episodes,
                                            opponents=args.opponents, target_sync_frequency=target_sync_frequency, feature_version=feature_version, 
                                            forward_model=forward_model)

                                            # write config file
                                            with open(args.config_directory + '/config_' + str(current_config_num) + '.yml', 'w+') as outfile:
                                                yaml.dump(tc, outfile, default_flow_style=False)
                                                print('Config', current_config_num, 'done.')
                                                current_config_num += 1
                                    else:
                                        # create config file
                                        tc = TrainingConfig(rl_agent=rl_agent, optimizer_type=optimizer_type, optimizer_lr=optimizer_lr,
                                        neural_net=neural_net, discount=discount, variable_noise=variable_noise, num_episodes=args.episodes,
                                        opponents=args.opponents, feature_version=feature_version, forward_model=forward_model)
                                        # write config file
                                        with open(args.config_directory + '/config_' + str(current_config_num) + '.yml', 'w+') as outfile:
                                            yaml.dump(tc, outfile, default_flow_style=False)
                                            print('Config', current_config_num, 'done.')
                                            current_config_num += 1



if __name__ == '__main__':
    main()