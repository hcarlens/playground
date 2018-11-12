'''Module to manage and advanced game state'''
from collections import defaultdict

import numpy as np

from pommerman import constants
from pommerman import characters
from pommerman import utility
from . import original_forward_model


class EnhancedFirstToDieForwardModel(original_forward_model.OriginalForwardModel):
    
    @staticmethod
    def get_rewards(agents, game_type, step_count, max_steps):

        def any_lst_equal(lst, values):
            '''Checks if list are equal'''
            return any([lst == v for v in values])

        alive_agents = [num for num, agent in enumerate(agents) \
                        if agent.is_alive]

        if game_type == constants.GameType.FFA:
            if len(alive_agents) == 1:
                # An agent won. Give them +20, others -10.
                return [30 * int(agent.is_alive) - 10 for agent in agents]
            elif step_count >= max_steps:
                # Game is over from time. Everyone gets -5. 
                # This doesn't always seem to be getting called
                return [-1] * 4
            elif len(alive_agents) == 3:
                # One agent has died. Give them -10. Everyone else gets +5.
                return [15 * int(agent.is_alive) - 10 for agent in agents]
            elif len(alive_agents) == 2:
                # Two agents have died. Give them -10. Everyone else gets +10. 
                return [20 * int(agent.is_alive) - 10 for agent in agents]
            else:
                # Game running: 0 for alive, -1 for dead, + some power-up bonuses
                return [int(agent.is_alive) - 1 + 0.01 * (agent.blast_strength - 2) + 0.01 * (agent.can_kick) for agent in agents]
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
