import argparse
import yaml

from agenttrainer import TrainingConfig

def main():
    '''CLI to generate configs for different settings to try'''
    parser = argparse.ArgumentParser(description="Config generator training flags.")
    parser.add_argument(
        "--config_directory", default='configs', help="Location to store generated config files")
    parser.add_argument(
        "--episodes", type=int, default=100000, help="Integer. Number of episodes to run.")
    parser.add_argument(
        "--opponents", default=None, help="Which agents to train against, out of simple and random. E.g. SSS = three simple agents, SRR = 1 simple and 2 random. ")
    args = parser.parse_args()

    rl_agents = ['PPO', 'DQN']
    optimizer_types = ['rmsprop'] # from DQN paper
    optimizer_lrs = [0.00025] # from DQN paper
    neural_nets = [
        [
                dict(type='conv2d', size=64, window=3, stride=2),
                dict(type='flatten'),
                dict(type='dense', size=512)
        [
                dict(type='conv2d', size=36, window=5, stride=1),
                dict(type='conv2d', size=81, window=3, stride=1),
                dict(type='flatten'),
                dict(type='dense', size=512)
            ],
        [
                dict(type='conv2d', size=81, window=3, stride=1),
                dict(type='flatten'),
                dict(type='dense', size=512)
        ],
        [
                dict(type='conv2d', size=32, window=8, stride=4),
                dict(type='conv2d', size=64, window=4, stride=2),
                dict(type='conv2d', size=32, window=3, stride=1),
                dict(type='flatten'),
                dict(type='dense', size=512)
        ]
        ]
    discounts = [0.99] # from DQN paper
    variable_noises = [None]
    actions_explorations = [{'type':'epsilon_decay', 'initial_epsilon':1.0, 'final_epsilon':0.01, 'timesteps':1e5}]
    ppomemories = [{'type':'replay', 'include_next_states': False, 'capacity':1e6}]
    dqnmemories = [{'type':'replay', 'include_next_states': True, 'capacity':1e6}]
    forward_models = ['simple','firsttodie']
    target_sync_frequencies = [10000] # from DQN paper
    batching_capacities = [32] # 32 from DQN paper
    use_simple_rewards = [True, False]
    feature_versions = [3]
    # also consider trying: different types of memory, batching capacities

    current_config_num = 0

    for feature_version in feature_versions:
        for actions_exploration in actions_explorations:
            for batching_capacity in batching_capacities:
                for optimizer_type in optimizer_types:
                    for optimizer_lr in optimizer_lrs:
                        for neural_net in neural_nets:
                            for discount in discounts:
                                for variable_noise in variable_noises:
                                    for forward_model in forward_models:
                                        for rl_agent in rl_agents:
                                            for use_simple_reward in use_simple_reward:
                                                if rl_agent == 'DQN':
                                                    for target_sync_frequency in target_sync_frequencies: # this only applies to DQNs
                                                        for memory in dqnmemories:
                                                            # create config file
                                                            tc = TrainingConfig(rl_agent=rl_agent, optimizer_type=optimizer_type, optimizer_lr=optimizer_lr,
                                                            neural_net=neural_net, discount=discount, variable_noise=variable_noise, num_episodes=args.episodes,
                                                            opponents=args.opponents, target_sync_frequency=target_sync_frequency, feature_version=feature_version, 
                                                            forward_model=forward_model, batching_capacity=batching_capacity, memory=memory, actions_exploration=actions_exploration, use_simple_rewards= use_simple_reward)

                                                            # write config file
                                                            with open(args.config_directory + '/config_' + str(current_config_num) + '.yml', 'w+') as outfile:
                                                                yaml.dump(tc, outfile, default_flow_style=False)
                                                                print('Config', current_config_num, 'done.')
                                                                current_config_num += 1
                                                else:
                                                    for memory in ppomemories:
                                                        # create config file
                                                        tc = TrainingConfig(rl_agent=rl_agent, optimizer_type=optimizer_type, optimizer_lr=optimizer_lr,
                                                        neural_net=neural_net, discount=discount, variable_noise=variable_noise, num_episodes=args.episodes,
                                                        opponents=args.opponents, feature_version=feature_version, forward_model=forward_model, batching_capacity=batching_capacity, 
                                                        memory=memory, actions_exploration=actions_exploration, use_simple_rewards= use_simple_reward)

                                                        # write config file
                                                        with open(args.config_directory + '/config_' + str(current_config_num) + '.yml', 'w+') as outfile:
                                                            yaml.dump(tc, outfile, default_flow_style=False)
                                                            print('Config', current_config_num, 'done.')
                                                            current_config_num += 1
        

if __name__ == '__main__':
    main()