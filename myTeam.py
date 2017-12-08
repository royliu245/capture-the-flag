# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
from game import Actions
import game
import capture

##

import distanceCalculator
from util import nearestPoint



#################
# Team creation #
#################

#FIRST_AGENT = 'OffensiveReflexAgent'
#FIRST_AGENT = 'HomogenousReflexAgent'
#FIRST_AGENT = 'GreedyBustersAgent'
FIRST_AGENT = 'TimidOffensiveReflexAgent'

#SECOND_AGENT = 'MidfielderAgent'
SECOND_AGENT = 'RefinedDefensiveReflexAgent'

def createTeam(firstIndex, secondIndex, isRed,
							 first = FIRST_AGENT, second = SECOND_AGENT):

	# The following line is an example only; feel free to change it.
	return [eval(first)(firstIndex), eval(second)(secondIndex)]

	# create a global/static wrapper and make the two instances of agents
	# share the wrapper?


##########
# Agents #
##########


class ReflexCaptureAgent(CaptureAgent):
	"""
	A base class for reflex agents that chooses score-maximizing actions
	"""
 
	def registerInitialState(self, gameState):
		self.start = gameState.getAgentPosition(self.index)
		CaptureAgent.registerInitialState(self, gameState)
		
		OFFENSIVE = True
		DEFENSIVE = False
		self.isOffensive = True
		self.touchHome = False
		self.foodTotal = len(self.getFood(gameState).asList())
		self.foodToReturn = 0

		# total initially - score earned already - foodLeft = carrying  



	def chooseAction(self, gameState):
		"""
		Picks among the actions with the highest Q(s,a).
		"""
		actions = gameState.getLegalActions(self.index)

		# You can profile your evaluation time by uncommenting these lines
		# start = time.time()
		values = [self.evaluate(gameState, a) for a in actions]
		# print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

		foodLeft = len(self.getFood(gameState).asList())
		ourFoodLeft = len(self.getFoodYouAreDefending(gameState).asList())
		self.foodToReturn = self.foodTotal - self.getScore(gameState) - foodLeft
		
		#home_returning_condition = (foodLeft <= 2)
		#home_returning_condition = (foodLeft <= 2) or (self.foodToReturn > 5)
		#carrying_food_threshold = 2
		carrying_food_threshold = 4
		carrying_food_threshold = int(7 * (float(foodLeft) / self.foodTotal))
		#print carrying_food_threshold, foodLeft, self.foodTotal
		home_returning_condition = (foodLeft <= 2) or\
								 (self.foodToReturn > carrying_food_threshold and self.isOffensive)

		# If the opponents' food <= 2, go back to the origin
		if home_returning_condition:
			bestDist = 9999
			for action in actions:
				successor = self.getSuccessor(gameState, action)
				pos2 = successor.getAgentPosition(self.index)
				dist = self.getMazeDistance(self.start,pos2)
				if dist < bestDist:
					bestAction = action
					bestDist = dist
			return bestAction
		else:
			maxValue = max(values)
			bestActions = [a for a, v in zip(actions, values) if v == maxValue]

			return random.choice(bestActions)


	def getSuccessor(self, gameState, action):
		"""
		Finds the next successor which is a grid position (location tuple).
		"""
		successor = gameState.generateSuccessor(self.index, action)
		pos = successor.getAgentState(self.index).getPosition()
		if pos != nearestPoint(pos):
			# Only half a grid position was covered
			return successor.generateSuccessor(self.index, action)
		else:
			return successor

	def evaluate(self, gameState, action):
		"""
		Computes a linear combination of features and feature weights
		"""
		features = self.getFeatures(gameState, action)
		weights = self.getWeights(gameState, action)
		return features * weights

	def getFeatures(self, gameState, action):
		"""
		Returns a counter of features for the state
		"""
		features = util.Counter()
		successor = self.getSuccessor(gameState, action)
		features['successorScore'] = self.getScore(successor)
		return features

	def getWeights(self, gameState, action):
		"""
		Normally, weights do not depend on the gamestate.  They can be either
		a counter or a dictionary.
		"""
		return {'successorScore': 1.0}

class MidfielderAgent(ReflexCaptureAgent):

	def getFeatures(self, gameState, action):
		features = util.Counter()
		successor = self.getSuccessor(gameState, action)

		foodList = self.getFood(successor).asList()    
		ourFoodList = self.getFoodYouAreDefending(successor).asList()
		myPos = successor.getAgentState(self.index).getPosition()

		teams = [successor.getAgentState(i) for i in self.getTeam(successor) if i != self.index]
		myTeamPos = teams[0].getPosition()
		distToTeam = self.getMazeDistance(myPos, myTeamPos)

		features['distanceToTeam'] = distToTeam
		
		enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
		invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
		ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]

		features['numInvaders'] = len(invaders)
		if len(invaders) > 0:
			dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
			features['invaderDistance'] = min(dists)


		features['numGhosts'] = len(ghosts)
		if len(ghosts) > 0:
			dists = [self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]
			features['ghostDistance'] = min(dists)


		if len(ourFoodList) > len(foodList):
			self.isOffensive = False
		else:
			self.isOffensive = True

		features['successorScore'] = -len(foodList)#self.getScore(successor)

		# Compute distance to the nearest food

		if len(foodList) > 0: # This should always be True,  but better safe than sorry
			minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
			features['distanceToFood'] = minDistance


		if action == Directions.STOP: features['stop'] = 1
		rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
		if action == rev: features['reverse'] = 1

		return features

	def evaluate(self, gameState, action):
		"""
		Computes a linear combination of features and feature weights
		"""
		features = self.getFeatures(gameState, action)
		"""
		if self.isOffensive:
			weights = self.getOffensiveWeights(gameState, action)
		else:
			#weights = self.getDefensiveWeights(gameState, action)
			weights = self.getOffensiveWeights(gameState, action)
		"""
		weights = self.getOffensiveWeights(gameState, action)

		return features * weights

	def getDefensiveWeights(self, gameState, action):
		return {'invaderDistance': -100,
				'stop': -100,
				'reverse': -100}

	def getOffensiveWeights(self, gameState, action):
		return {'successorScore': 1,
					 'distanceToFood': -2,
					 'ghostDistance': 30,
					 'stop': -10,
					 'reverse': -10}

class TimidOffensiveReflexAgent(ReflexCaptureAgent):
	"""
	A reflex agent that seeks food. This is an agent
	we give you to get an idea of what an offensive agent might look like,
	but it is by no means the best or only way to build an offensive agent.
	"""
	def getFeatures(self, gameState, action):
		features = util.Counter()
		successor = self.getSuccessor(gameState, action)
		foodList = self.getFood(successor).asList()    
		ourFoodList = self.getFoodYouAreDefending(successor).asList()
		myState = successor.getAgentState(self.index)
		myPos = myState.getPosition()

		enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
		ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]

		features['successorScore'] = -len(foodList) #self.getScore(successor)

		# Compute distance to the nearest food

		if len(foodList) > 0: # This should always be True,  but better safe than sorry
			myPos = successor.getAgentState(self.index).getPosition()
			minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
			features['distanceToFood'] = minDistance

		features['numGhosts'] = len(ghosts)
		if len(ghosts) > 0:
			dists = [self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]
			features['ghostDistance'] = min(dists)

		if myState.isPacman:
			features['ghostDistance'] *= (-1)

		return features

	def getWeights(self, gameState, action):
		return {'successorScore': 100, 
				'distanceToFood': -1,
				'ghostDistance': 2}

class OriginalOffensiveReflexAgent(ReflexCaptureAgent):
	"""
	A reflex agent that seeks food. This is an agent
	we give you to get an idea of what an offensive agent might look like,
	but it is by no means the best or only way to build an offensive agent.
	"""
	def getFeatures(self, gameState, action):
		features = util.Counter()
		successor = self.getSuccessor(gameState, action)
		foodList = self.getFood(successor).asList()    
		ourFoodList = self.getFoodYouAreDefending(successor).asList()

		features['successorScore'] = -len(foodList)#self.getScore(successor)

		# Compute distance to the nearest food

		if len(foodList) > 0: # This should always be True,  but better safe than sorry
			myPos = successor.getAgentState(self.index).getPosition()
			minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
			features['distanceToFood'] = minDistance
		return features

	def getWeights(self, gameState, action):
		return {'successorScore': 100, 'distanceToFood': -1}

class DefensiveReflexAgent(ReflexCaptureAgent):
	"""
	A reflex agent that keeps its side Pacman-free. Again,
	this is to give you an idea of what a defensive agent
	could be like.  It is not the best or only way to make
	such an agent.
	"""

	def getFeatures(self, gameState, action):
		features = util.Counter()
		successor = self.getSuccessor(gameState, action)

		myState = successor.getAgentState(self.index)
		myPos = myState.getPosition()

		# Computes whether we're on defense (1) or offense (0)
		features['onDefense'] = 1
		if myState.isPacman: features['onDefense'] = 0

		# Computes distance to invaders we can see
		enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
		invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
		features['numInvaders'] = len(invaders)
		if len(invaders) > 0:
			dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
			features['invaderDistance'] = min(dists)

		if action == Directions.STOP: features['stop'] = 1
		rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
		if action == rev: features['reverse'] = 1

		return features

	def getWeights(self, gameState, action):
		return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -1010010, 'stop': -100, 'reverse': -2}

class RefinedDefensiveReflexAgent(ReflexCaptureAgent):
	"""
	A reflex agent that keeps its side Pacman-free. Again,
	this is to give you an idea of what a defensive agent
	could be like.  It is not the best or only way to make
	such an agent.
	"""
	def registerInitialState(self, gameState):
		ReflexCaptureAgent.registerInitialState(self, gameState)
		self.isOffensive = False

	def getFeatures(self, gameState, action):
		features = util.Counter()
		successor = self.getSuccessor(gameState, action)

		myState = successor.getAgentState(self.index)
		myPos = myState.getPosition()

		# Computes whether we're on defense (1) or offense (0)
		features['onDefense'] = 1
		if myState.isPacman: 
			features['onDefense'] = 0

		# Computes distance to invaders we can see
		enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
		invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
		#print a
		#print invaders


		features['numInvaders'] = len(invaders)
		if len(invaders) > 0:
			dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
			###########################################################
			#print self.getCurrentObservation()
			#distributions = [None]
			#self.displayDistributionsOverPositions(distributions)
			features['invaderDistance'] = min(dists)

		if action == Directions.STOP: features['stop'] = 1
		rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
		if action == rev: features['reverse'] = 1

		return features

	def getWeights(self, gameState, action):
		return {'numInvaders': -1000,\
						 'onDefense': 100,\
						 'invaderDistance': -1000,\
						 'stop': -100,\
						 'reverse': -2}




class DummyAgent(CaptureAgent):
	"""
	A Dummy agent to serve as an example of the necessary agent structure.
	You should look at baselineTeam.py for more details about how to
	create an agent as this is the bare minimum.
	"""

	def registerInitialState(self, gameState):
		"""
		This method handles the initial setup of the
		agent to populate useful fields (such as what team
		we're on).

		A distanceCalculator instance caches the maze distances
		between each pair of positions, so your agents can use:
		self.distancer.getDistance(p1, p2)

		IMPORTANT: This method may run for at most 15 seconds.
		"""

		'''
		Make sure you do not delete the following line. If you would like to
		use Manhattan distances instead of maze distances in order to save
		on initialization time, please take a look at
		CaptureAgent.registerInitialState in captureAgents.py.
		'''
		CaptureAgent.registerInitialState(self, gameState)

		'''
		Your initialization code goes here, if you need any.
		'''


	def chooseAction(self, gameState):
		"""
		Picks among actions randomly.
		"""
		actions = gameState.getLegalActions(self.index)

		'''
		You should change this in your own agent.
		'''

		return random.choice(actions)

