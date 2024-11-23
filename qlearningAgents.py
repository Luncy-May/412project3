# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

import random
import util

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Implements the core Q-learning methods:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Uses instance variables:
        - self.epsilon (exploration probability)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Available helper method:
        - self.getLegalActions(state)
    """
    def __init__(self, **args):
        """
          Initialize Q-values and agent parameters.
        """
        ReinforcementAgent.__init__(self, **args)
        self.qValues = util.Counter()  # Stores Q-values as a dictionary

    def getQValue(self, state, action):
        """
          Returns Q(state, action).
          If the (state, action) pair hasn't been encountered, return 0.0.
        """
        return self.qValues[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Compute the maximum Q-value for all legal actions in the state.
          Returns 0.0 if there are no legal actions (terminal state).
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return 0.0  # Terminal state
        return max(self.getQValue(state, action) for action in legalActions)

    def computeActionFromQValues(self, state):
        """
          Compute the best action in the state. Break ties randomly.
          Returns None if there are no legal actions (terminal state).
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None  # Terminal state
        
        # Find the maximum Q-value
        maxQValue = self.computeValueFromQValues(state)
        # Collect all actions with the maximum Q-value
        bestActions = [action for action in legalActions if self.getQValue(state, action) == maxQValue]
        
        # Random tie-breaking
        return random.choice(bestActions)

    def getAction(self, state):
        """
          Select an action based on epsilon-greedy policy:
          - With probability epsilon, take a random action.
          - Otherwise, take the best policy action.
          Returns None if there are no legal actions.
        """
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None  # Terminal state

        # Explore with probability epsilon
        if util.flipCoin(self.epsilon):
            return random.choice(legalActions)
        
        # Exploit the best action
        return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """
          Update Q-values using the Q-learning update rule:
          Q(s, a) = (1 - alpha) * Q(s, a) + alpha * (reward + discount * max Q(s', a'))
        """
        oldQValue = self.getQValue(state, action)
        futureQValue = self.computeValueFromQValues(nextState)  # max_a' Q(s', a')
        # Q-learning update formula
        newQValue = (1 - self.alpha) * oldQValue + self.alpha * (reward + self.discount * futureQValue)
        self.qValues[(state, action)] = newQValue

    def getPolicy(self, state):
        """
          Returns the best action in the state according to the current Q-values.
        """
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        """
          Returns the maximum Q-value for the state over all legal actions.
        """
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            pass
