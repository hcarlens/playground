'''An agent that passes each step'''
from . import BaseAgent

class StaticAgent(BaseAgent):
    """The Static Agent that always passes."""

    def act(self, obs, action_space):
        return 0
