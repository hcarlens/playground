import pommerman
from pommerman import agents

# CONFIG = {
#     "game_type": 'PommeFFACompetition-v0',
# }

# AGENT_LIST = ["random", "simple", "simple", "simple"]

class GameRunner():

    def get_agent(self, agent_list):
        agent_instance = None
        for agent_type in self.agents_config:
            assert agent_type in ["simple", "random", "stupid", "silly"]
            if agent_type == "simple":
                agent_instance = agents.SimpleAgent()
            elif agent_type == "random":
                agent_instance = agents.RandomAgent()
            elif agent_type == "stupid":
                agent_instance = agents.StupidAgent()
            elif agent_type == "silly":
                agent_instance = agents.SillyAgent()
        return agent_instance

    def run_game(self, agent_list):
        agents = []
        for agent in self.agents_config:
            agents.append(self.get_agent(agent))
        env = pommerman.make(self.config["game_type"], agents)
        state = env.reset()
        done = False
        while not done:
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
            if done:
                return info
            env.close()

    def percentage_wins(self, wins, total):
        return (wins/total) * 100

    def run(self, config, agents_config, plays):
        self.agents_config = agents_config
        self.config = config

        results = dict()
        for agent in self.agents_config:
            results[agent] = 0

        i = 0
        while i < plays:
            game_info = self.run_game(agents_config)
            i += 1
            if game_info["result"].value == 0:
                winner = self.agents_config[game_info["winners"][0]]
                results[winner] += 1
                print('{} Winnder: {}'.format(i, winner))
            elif game_info["result"].value == 2:
                print('Tie')

        overallWinner = ""
        highestWins = 0

        for key, value in results.items():
            if value >= highestWins:
                highestWins = value
                overallWinner = key
            percentage = self.percentage_wins(value, plays)
            percentage_string = "{:.2f}".format(percentage)
            print("{}: {} ({}%)".format(key, value, percentage_string))
        print("Overall Winner: {}".format(overallWinner))

config = {"game_type": 'PommeFFACompetition-v0'}
agentconfig = ["random", "simple", "simple", "simple"]
GameRunner().run(config, agentconfig, 10)