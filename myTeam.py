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
FIRST_AGENT = 'GreedyBustersAgent'
#FIRST_AGENT = 'MidfielderAgent'
#SECOND_AGENT = 'RefinedDefensiveReflexAgent'
SECOND_AGENT = 'MidfielderAgent'
#SECOND_AGENT = 'RefinedDefensiveReflexAgent'

def createTeam(firstIndex, secondIndex, isRed,
							 first = FIRST_AGENT, second = SECOND_AGENT):
	"""
	This function should return a list of two agents that will form the
	team, initialized using firstIndex and secondIndex as their agent
	index numbers.  isRed is True if the red team is being created, and
	will be False if the blue team is being created.

	As a potentially helpful development aid, this function can take
	additional string-valued keyword arguments ("first" and "second" are
	such arguments in the case of this function), which will come from
	the --redOpts and --blueOpts command-line arguments to capture.py.
	For the nightly contest, however, y
	our team will be created without
	any extra arguments, so you should make sure that the default
	behavior is what you want for the nightly contest.
	"""

	# The following line is an example only; feel free to change it.
	return [eval(first)(firstIndex), eval(second)(secondIndex)]

	# create a global/static wrapper and make the two instances of agents
	# share the wrapper?



class InferenceModule:
		def __init__(self, opponentAgentIndex, currentAgent):
				"Sets the ghost agent for later access"
				#self.opponentAgent = opponentAgent
				self.index = opponentAgentIndex
				self.obs = [] # most recent observation position
				self.beliefs = None
				self.agent = currentAgent
				self.observationDistributions = {}

		def observeState(self, gameState):
				"Collects the relevant noisy distance observation and pass it along."
				#distances = gameState.getNoisyGhostDistances()
				#successor = self.agent.getSuccessor(gameState, action)
				myState = gameState.getAgentState(self.agent.index)
				myPos = myState.getPosition()
				enemies = [gameState.getAgentState(i) for i in self.agent.getOpponents(gameState)]
				print self.agent.getOpponents(gameState)
				distances = list()

				for a in enemies:
					print a
					distances.append(self.agent.getMazeDistance(myPos, a.getPosition()))

				#distances = [self.agent.getMazeDistance(myPos, a.getPosition()) for a in enemies]

				if len(distances) >= self.index: # Check for missing observations
						obs = distances[self.index - 1]
						self.obs = obs
						self.observe(obs, gameState)

		def getObservationDistribution(self, noisyDistance):
				"""
				Returns the factor P( noisyDistance | TrueDistances ), the likelihood of the provided noisyDistance
				conditioned upon all the possible true distances that could have generated it.
				"""
				SONAR_NOISE_VALUES = capture.SONAR_NOISE_VALUES
				SONAR_NOISE_RANGE = capture.SONAR_NOISE_RANGE
				SONAR_MAX = (SONAR_NOISE_RANGE - 1)/2
				SONAR_DENOMINATOR = 2 ** SONAR_MAX  + 2 ** (SONAR_MAX + 1) - 2.0
				SONAR_NOISE_PROBS = [2 ** (SONAR_MAX-abs(v)) / SONAR_DENOMINATOR  for v in SONAR_NOISE_VALUES]

				observationDistributions = self.observationDistributions
				if noisyDistance == None:
						return util.Counter()
				if noisyDistance not in observationDistributions:
						distribution = util.Counter()
						for error , prob in zip(SONAR_NOISE_VALUES, SONAR_NOISE_PROBS):
								distribution[max(1, noisyDistance - error)] += prob
						observationDistributions[noisyDistance] = distribution
				return observationDistributions[noisyDistance]

		def getPositionDistribution(self, agentIndex, gameState):
				"""
				Returns a distribution over successor positions of the ghost from the
				given gameState.

				You must first place the ghost in the gameState, using setGhostPosition
				below.
				"""
				agentPosition = gameState.getPosition(agentIndex) # The position you set
				actionDist = self.ghostAgent.getDistribution(gameState)
				dist = util.Counter()
				for action, prob in actionDist.items():
						successorPosition = game.Actions.getSuccessor(agentPosition, action)
						dist[successorPosition] = prob
				return dist

		def setOpponentPosition(self, gameState, opponentPosition):
				
				conf = game.Configuration(opponentPosition, game.Directions.STOP)
				gameState.data.agentStates[self.index] = game.AgentState(conf, False)
				return gameState


		def initialize(self, gameState):
				"Initializes beliefs to a uniform distribution over all positions."
				# The legal positions do not include the ghost prison cells in the bottom left.
				self.legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
				self.initializeUniformly(gameState)

		def initializeUniformly(self, gameState):
				"Sets the belief state to a uniform prior belief over all positions."
				pass

		def observe(self, observation, gameState):
				"Updates beliefs based on the given distance observation and gameState."
				pass

		def elapseTime(self, gameState):
				"Updates beliefs for a time step elapsing from a gameState."
				pass

		def getBeliefDistribution(self):
				"""
				Returns the agent's current belief state, a distribution over ghost
				locations conditioned on all evidence so far.
				"""
				pass


class ExactInference(InferenceModule):

		def initializeUniformly(self, gameState):
				"Begin with a uniform distribution over ghost positions."
				self.beliefs = util.Counter()
				for p in self.legalPositions: self.beliefs[p] = 1.0
				self.beliefs.normalize()



		def observe(self, observation, gameState):
				noisyDistance = observation
				emissionModel = self.getObservationDistribution(noisyDistance)
				myPosition = gameState.getAgentState(self.agent.index).getPosition()

				allPossible = util.Counter()
				if noisyDistance is None: # HANDLE THIS LATER
						return

				else:
						for p in self.legalPositions:
								trueDistance = self.agent.getMazeDistance(p, myPosition)
								probNoisyDistanceGivenTrueDistance = emissionModel[trueDistance]
								
								if emissionModel[trueDistance] > 0:
										allPossible[p] = self.beliefs[p] * probNoisyDistanceGivenTrueDistance

				allPossible.normalize()
				self.beliefs = allPossible

		def getBeliefDistribution(self):

				return self.beliefs




##########
# Agents #
##########


class BustersAgent(CaptureAgent):
		def __init__( self, index, timeForComputing = .1 ):
				"""
				Lists several variables you can query:
				self.index = index for this agent
				self.red = true if you're on the red team, false otherwise
				self.agentsOnTeam = a list of agent objects that make up your team
				self.distancer = distance calculator (contest code provides this)
				self.observationHistory = list of GameState objects that correspond
						to the sequential order of states that have occurred so far this game
				self.timeForComputing = an amount of time to give each turn for computing maze distances
						(part of the provided distance calculator)
				"""
				# Agent index for querying state
				self.index = index

				# Whether or not you're on the red team
				self.red = None

				# Agent objects controlling you and your teammates
				self.agentsOnTeam = None

				# Maze distance calculator
				self.distancer = None

				# A history of observations
				self.observationHistory = []

				# Time to spend each turn on computing maze distances
				self.timeForComputing = timeForComputing

				# Access to the graphics
				self.display = None

		def registerInitialState(self, gameState):
				self.start = gameState.getAgentPosition(self.index)
				opponentAgents = self.getOpponents(gameState)
				self.inferenceModules = [ExactInference(idx, self) for idx in opponentAgents]
				for inference in self.inferenceModules:
						inference.initialize(gameState)
				self.opponentBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
				self.red = gameState.isOnRedTeam(self.index)
				self.distancer = distanceCalculator.Distancer(gameState.data.layout)
				self.distancer.getMazeDistances()
				self.observe_enabled = False

		def getAction(self, gameState): # without elapse time
				"Updates beliefs, then chooses an action based on updated beliefs."
				for index, inf in enumerate(self.inferenceModules):
						self.opponentBeliefs[index] = inf.getBeliefDistribution()
						if self.observe_enabled:
							inf.observeState(gameState)
							print self.opponentBeliefs[index]
							self.displayDistributionsOverPositions(self.opponentBeliefs)
						else:
							self.observe_enabled = True
				#self.displayDistributionsOverPositions(self.opponentBeliefs)
				return self.chooseAction(gameState)

		def observationFunction(self, gameState):
				"Removes the ghost states from the gameState"
				#agents = gameState.data.agentStates
				#gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
				return gameState

		def chooseAction(self, gameState):
				"By default, a BustersAgent just stops.  This should be overridden."
				return Directions.STOP


class GreedyBustersAgent(BustersAgent, CaptureAgent):
		"An agent that charges the closest ghost."

		def registerInitialState(self, gameState):
				BustersAgent.registerInitialState(self, gameState)
				

		def chooseAction(self, gameState):
				myPosition = gameState.getAgentPosition(self.index)
				legal = [a for a in gameState.getLegalActions()]
				#livingGhosts = gameState.getLivingGhosts()

				opponentPositionDistributions = \
							[beliefs for i, beliefs in enumerate(self.opponentBeliefs)]

				likeliestOpponentPositions = []
				# First, find the likeliest position of each ghost
				for candidateOpponentDistribution in opponentPositionDistributions:
						maxProb = 0
						maxPos = None
						for pos, prob in candidateOpponentDistribution.items():
								if prob >= maxProb:
										maxPos = pos
										maxProb = prob
						likeliestOpponentPositions.append(maxPos)

				# Second, find the nearest probable ghost
				nearestOpponentPosition = None
				nearestOpponentDistance = 9999999999 # ASSUMING THIS IS THE MAX
				for opponentPos in likeliestOpponentPositions:
						dist = self.getMazeDistance(myPosition, opponentPos)
						if dist < nearestOpponentDistance:
								nearestOpponentDistance = dist
								nearestOpponentPosition = opponentPos
				
				#print(pacmanPosition, nearestGhostPosition)
				minimalDistance = self.getMazeDistance(myPosition, nearestOpponentPosition)

				# Third, find the optimal action
				# Compute the NEW distance to the nearest ghost GIVEN an action
				# Among those distances, find the minimum.
				# I suspect that there can be another way, where
				# we choose the nearest ghost GIVEN an action and then compute the distance.
				optimalAction = random.choice(legal)
				print legal
				for candidateAction in legal:
						newMyPosition = Actions.getSuccessor(myPosition, candidateAction)            
						newDist = self.getMazeDistance(newMyPosition, nearestOpponentPosition)
						
						print "{} ... {} ...?".format(legal, newDist)
						if newDist < minimalDistance:
								minimalDistance = newDist
								optimalAction = candidateAction
								#print("UPDATE:", optimalAction, minimalDistance)
						

				return optimalAction



class ReflexCaptureAgent(CaptureAgent):
	"""
	A base class for reflex agents that chooses score-maximizing actions
	"""
 
	def registerInitialState(self, gameState):
		self.start = gameState.getAgentPosition(self.index)
		CaptureAgent.registerInitialState(self, gameState)


	def chooseAction(self, gameState):
		"""
		Picks among the actions with the highest Q(s,a).
		"""
		actions = gameState.getLegalActions(self.index)

		# You can profile your evaluation time by uncommenting these lines
		# start = time.time()
		values = [self.evaluate(gameState, a) for a in actions]
		# print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

		maxValue = max(values)
		bestActions = [a for a, v in zip(actions, values) if v == maxValue]

		foodLeft = len(self.getFood(gameState).asList())

		if foodLeft <= 2:
		#if foodLeft <= 10:	
			bestDist = 9999
			for action in actions:
				successor = self.getSuccessor(gameState, action)
				pos2 = successor.getAgentPosition(self.index)
				dist = self.getMazeDistance(self.start,pos2)
				if dist < bestDist:
					bestAction = action
					bestDist = dist
			return bestAction

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

class HomogenousReflexAgent(ReflexCaptureAgent):
	"""
	A reflex agent that seeks food. This is an agent
	we give you to get an idea of what an offensive agent might look like,
	but it is by no means the best or only way to build an offensive agent.
	"""
	def getFeatures(self, gameState, action):
		features = util.Counter()
		successor = self.getSuccessor(gameState, action)
		foodList = self.getFood(successor).asList()    
		features['successorScore'] = -len(foodList)#self.getScore(successor)

		myState = successor.getAgentState(self.index)
		myPos = myState.getPosition()
		# Computes distance to invaders we can see
		enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
		teams = [successor.getAgentState(i) for i in self.getTeam(successor) if i != self.index]
		myTeamPos = teams[0].getPosition()
		print "{} ... myTeamPos : {}".format(myPos, myTeamPos)
		features['teamDistance'] = self.getMazeDistance(myPos, myTeamPos)

		ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
		invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
		#print a
		#print invaders

		# Computes whether we're on defense (1) or offense (0)
		features['onDefense'] = 1
		if myState.isPacman: 
			features['onDefense'] = 0


		
		
		features['numGhosts'] = len(ghosts)
		if len(ghosts) > 0:
			dists = [self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]
			features['ghostDistance'] = min(dists)

		features['invaderDistance'] = len(invaders)
		if len(invaders) > 0:
			dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
			features['invaderDistance'] = min(dists)

		# Compute distance to the nearest food

		if len(foodList) > 0: # This should always be True,  but better safe than sorry
			myPos = successor.getAgentState(self.index).getPosition()
			minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
			features['distanceToFood'] = minDistance

		if action == Directions.STOP: features['stop'] = 1
		rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
		if action == rev: features['reverse'] = 1

		return features

	def getWeights(self, gameState, action):
		successor = self.getSuccessor(gameState, action)
		myState = successor.getAgentState(self.index)
		fight_or_flight_factor = 1 # Pacman -> positive (fight)
		if myState.isPacman: 
			fight_or_flight_factor = 100
		else:
			fight_or_flight_factor = -100

		if myState.isPacman:
			return {'successorScore': 100,\
						 'distanceToFood': -20000,\
						 'ghostDistance': 100000,\
						 'invaderDistance': 0,\
						 'numGhosts': 10,\
						 'numInvaders': 0,\
						 'stop': -100,\
						 'reverse': -2,\
						 'teamDistance': 1}
		else: # if ghost, you must catch the invader pacmans
			return {'successorScore': 100,\
						 'distanceToFood': -5,\
						 'ghostDistance': 0,\
						 'invaderDistance': -100000,\
						 'numGhosts': 0,\
						 'numInvaders': 0,\
						 'stop': -100,\
						 'reverse': -2,\
						 'teamDistance': 1}

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

		features['successorScore'] = -len(foodList)#self.getScore(successor)

		# Compute distance to the nearest food

		if len(foodList) > 0: # This should always be True,  but better safe than sorry
			minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
			features['distanceToFood'] = minDistance


		if action == Directions.STOP: features['stop'] = 1
		rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
		if action == rev: features['reverse'] = 1

		return features

	def getWeights(self, gameState, action):
		successor = self.getSuccessor(gameState, action)
		foodList = self.getFood(successor).asList()    
		ourFoodList = self.getFoodYouAreDefending(successor).asList()

		if len(ourFoodList) < len(foodList):
			defense_factor = 3
		else:
			defense_factor = 0

		return {'successorScore': 100,\
					 'distanceToFood': -5,\
					 'invaderDistance': -10,# * defense_factor,
					 'ghostDistance': 4,
					 'stop': -100,
					 'reverse': -100}



class MidfielderAgent_old(ReflexCaptureAgent):
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

		features['successorScore'] = -len(foodList)#self.getScore(successor)

		# Compute distance to the nearest food

		if len(foodList) > 0: # This should always be True,  but better safe than sorry
			minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
			features['distanceToFood'] = minDistance


		if action == Directions.STOP: features['stop'] = 1
		rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
		if action == rev: features['reverse'] = 1

		return features

	def getWeights(self, gameState, action):
		successor = self.getSuccessor(gameState, action)
		foodList = self.getFood(successor).asList()    
		ourFoodList = self.getFoodYouAreDefending(successor).asList()

		if len(ourFoodList) < len(foodList):
			defense_factor = 100
		else:
			defense_factor = 0

		

		return {'successorScore': 100,\
					 'distanceToFood': -5,\
					 'invaderDistance': -10,# * defense_factor,
					 'ghostDistance': 4,
					 'stop': -100,
					 'reverse': -100}

		return {'successorScore': 100,\
					 'distanceToFood': -7,\
					 'invaderDistance': -100 * defense_factor,
					 'ghostDistance': 1 * defense_factor,
					 'stop': -100,
					 'reverse': -100}


class OffensiveReflexAgent(ReflexCaptureAgent):
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
						 'invaderDistance': -100,\
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

