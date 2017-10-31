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
	
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        v_temp = util.Counter()
        while iterations > 0:
			v_temp = util.Counter()
			for s in self.mdp.getStates():
				if self.mdp.isTerminal(s):
					v_temp[s] = 0
					continue					
				v = self.values[s]
				v_temp[s] = float('-inf')
				for a in self.mdp.getPossibleActions(s):
					temp_sum = 0
					for s_prime, p in self.mdp.getTransitionStatesAndProbs(s, a):
						r = self.mdp.getReward(s, a, s_prime)
						temp_sum += p*(r + self.discount*self.values[ s_prime ])
					v_temp[s] = max(v_temp[s], temp_sum)
					
					
			iterations -= 1
			self.values = v_temp
						
		


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
        q = 0
        for s_prime, p in self.mdp.getTransitionStatesAndProbs(state, action):
			r = self.mdp.getReward(state, action, s_prime)
			q += p*(r + self.discount*self.values[s_prime])
        return q
			
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        best_action = None
        q_best = float("-inf")			
        for action in self.mdp.getPossibleActions(state):
			q = self.computeQValueFromValues(state, action)
			if q > q_best:
				best_action = action
				q_best = q
        return best_action
			
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
