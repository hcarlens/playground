'''An agent that preforms a random action each step, but never bombs'''
import random
from . import BaseAgent


class PacifistRandomAgent(BaseAgent):
    """The Pacifist Random Agent that returns random actions excluding bombing."""

    def act(self, obs, action_space):
        return random.randint(0,4)
