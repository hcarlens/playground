import pommerman
from pommerman import agents

CONFIG = {
    "game_type": 'PommeFFACompetition-v0',
}

AGENT_LIST = ["random", "simple", "simple", "simple"]

def get_agent(agent_list):
    agent_instance = None
    for agent_type in AGENT_LIST:
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

def run_game(agent_list):
    agents = []
    for agent in AGENT_LIST:
        agents.append(get_agent(agent))
    env = pommerman.make(CONFIG["game_type"], agents)
    state = env.reset()
    done = False
    while not done:
        actions = env.act(state)
        state, reward, done, info = env.step(actions)
        if done:
            return info
        env.close()

def percentageWins(wins, total):
    return (wins/total) * 100

def main(plays):
    results = dict()
    for agent in AGENT_LIST:
        results[agent] = 0

    i = 0
    while i < plays:
        game_info = run_game(AGENT_LIST)
        i += 1
        if game_info["result"].value == 0:
            winner = AGENT_LIST[game_info["winners"][0]]
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
        percentage = percentageWins(value, plays)
        percentage_string = "{:.2f}".format(percentage)
        print("{}: {} ({}%)".format(key, value, percentage_string))
    print("Overall Winner: {}".format(overallWinner))

main(10)