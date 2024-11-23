# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for each in range(self.iterations):
            copyValues = self.values.copy() # copy so we do not modify original values
            for state in self.mdp.getStates():  
                if self.mdp.isTerminal(state):  # if terminated, stop 
                    continue
                # if not terminated, go ahead
                actions = self.mdp.getPossibleActions(state) # get a list of possible actions
                maxValue = max([self.getQValue(state, action) for action in actions])
                copyValues[state] = maxValue
            self.values = copyValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q = 0 # initializes q value
        for next, prob in self.mdp.getTransitionStatesAndProbs(state, action): # each next state and probability
            q += prob * (self.mdp.getReward(state, action, state) + self.discount * self.values[next])
        return q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        maxAction = 0
        maxValue = float("-inf")
        for action in self.mdp.getPossibleActions(state):
            q = self.getQValue(state,action)
            if q > maxValue:
                maxValue = q
                maxAction = action
        return maxAction
    
    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates() # total states
        for iteration in range(self.iterations): 
            state = states[iteration % len(states)] # get the curr state
            if self.mdp.isTerminal(state): continue # stop if teriminated
            maxValue = float('-inf') # initialize negative infinity 
            for action in self.mdp.getPossibleActions(state):
                maxValue = max(maxValue, self.computeQValueFromValues(state,action)) # compare maxValue and q value
            self.values[state] = maxValue
                
class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # initalizes a priority queue and a dictionary of sets
        pq, predecessors = util.PriorityQueue(), {} 
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state): continue
            maxValue = float('-inf')
            for action in self.mdp.getPossibleActions(state):
                for next, prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if next in predecessors:
                        predecessors[next].add(state)
                    else:
                        predecessors[next] = {state}
                q = self.computeQValueFromValues(state, action)
                maxValue = max(q, maxValue)
            pq.update(state, - abs(maxValue - self.values[state]))
        
                        
        for i in range(self.iterations):
            if pq.isEmpty(): break
            currState = pq.pop()
            if self.mdp.isTerminal(currState): continue
            maxValue = float("-inf")
            for action in self.mdp.getPossibleActions(currState):
                q = self.computeQValueFromValues(currState, action)
                maxValue = max(q, maxValue)
            self.values[currState] = maxValue

            for predecessor in predecessors[currState]:
                if self.mdp.isTerminal(predecessor): continue
                maxValue = float('-inf')
                for action in self.mdp.getPossibleActions(predecessor):
                    q = self.computeQValueFromValues(predecessor, action)
                    maxValue = max(q, maxValue)
                difference = abs(maxValue - self.values[predecessor])
                if difference > self.theta:
                    pq.update(predecessor, - difference)
