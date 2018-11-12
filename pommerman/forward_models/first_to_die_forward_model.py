'''Module to manage and advanced game state'''
from collections import defaultdict

import numpy as np

from pommerman import constants
from pommerman import characters
from pommerman import utility
from . import original_forward_model


class FirstToDieForwardModel(original_forward_model.OriginalForwardModel):
    
    @staticmethod
    def get_rewards(agents, game_type, step_count, max_steps):

        def any_lst_equal(lst, values):
            '''Checks if list are equal'''
            return any([lst == v for v in values])

        alive_agents = [num for num, agent in enumerate(agents) \
                        if agent.is_alive]

        if game_type == constants.GameType.FFA:
            if len(alive_agents) == 1:
                # An agent won. Give them +10, others -1.
                return [11 * int(agent.is_alive) - 1 for agent in agents]
            elif step_count >= max_steps:
                # Game is over from time. Everyone gets -5.
                return [-5] * 4
            elif len(alive_agents) == 3:
                # One agent has died. Give them -10. Everyone else gets 0.
                return [10 * int(agent.is_alive) - 10 for agent in agents]
            elif len(alive_agents) == 2:
                # Two agents have died. Give them -2. Everyone else gets 0.
                return [2 * int(agent.is_alive) - 2 for agent in agents]
            else:
                # This shouldn't be invoked, but leaving it here in case we forgot about a scenario.
                # Game running: 0 for alive, -1 for dead.
                return [int(agent.is_alive) - 1 for agent in agents]
        else:
            # We are playing a team game.
            if any_lst_equal(alive_agents, [[0, 2], [0], [2]]):
                # Team [0, 2] wins.
                return [1, -1, 1, -1]
            elif any_lst_equal(alive_agents, [[1, 3], [1], [3]]):
                # Team [1, 3] wins.
                return [-1, 1, -1, 1]
            elif step_count >= max_steps:
                # Game is over by max_steps. All agents tie.
                return [-1] * 4
            elif len(alive_agents) == 0:
                # Everyone's dead. All agents tie.
                return [-1] * 4
            else:
                # No team has yet won or lost.
                return [0] * 4
