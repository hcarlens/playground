'''Module to manage and advanced game state'''
from collections import defaultdict

import numpy as np

from pommerman import constants
from pommerman import characters
from pommerman import utility
from . import original_forward_model


class SimpleForwardModel(original_forward_model.OriginalForwardModel):
    @staticmethod
    def get_rewards(agents, game_type, step_count, max_steps):

        def any_lst_equal(lst, values):
            '''Checks if list are equal'''
            return any([lst == v for v in values])

        alive_agents = [num for num, agent in enumerate(agents) \
                        if agent.is_alive]
        if game_type == constants.GameType.FFA:
            if len(alive_agents) == 1:
                # An agent won. Give them +1000
                return [1000 * int(agent.is_alive) for agent in agents]
            elif step_count >= max_steps:
                # Game is over from time. Everyone gets -20.
                return [-20] * 4
            else:
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
