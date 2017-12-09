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
import math

##

import distanceCalculator
from util import nearestPoint



#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
							 first = 'RefinedDefensiveReflexAgent', second = 'TimidOffensiveReflexAgent'):
	
	
	#memberone = eval('HybridAgent')(firstIndex)
	#membertwo = eval('HybridAgent')(secondIndex, memberone)
	
	#return [memberone, membertwo]
	

	def pair_team(first_agent, second_agent):
		return [eval(first_agent)(firstIndex), eval(second_agent)(secondIndex)]

	#version_one = pair_team('TimidOffensiveReflexAgent', 'RefinedDefensiveReflexAgent')

	memberone = eval('HybridAgent')(firstIndex)
	membertwo = eval('SensorRefinedDefensiveReflexAgent')(secondIndex, memberone)

	return [memberone, membertwo]

	#version_two = pair_team('SensorRefinedDefensiveReflexAgent', 'HybridAgent')
	#version_two = pair_team('HybridAgent', 'HybridAgent')



	return version_two
	


##########
# Agents #
##########


class HybridAgent(CaptureAgent):
	"""
	A Dummy agent to serve as an example of the necessary agent structure.
	You should look at baselineTeam.py for more details about how to
	create an agent as this is the bare minimum.
	"""

	def __init__(self, index, teammates = None, timeForComputing=.1):
		CaptureAgent.__init__(self, index, timeForComputing)

		self.teammates = []
		if teammates is not None:
			self.teammates.append(teammates)
		for teammate in self.teammates:
			if len(teammate.teammates) == 0:
				teammate.teammates.append(self)



	def registerInitialState(self, gameState):
		print "HybridAgent"
		CaptureAgent.registerInitialState(self, gameState)
		self.numagents = gameState.getNumAgents()
		self.start = gameState.getAgentPosition(self.index)
		#self.myindex =
		#self.mycolor =
		self.distancer = distanceCalculator.Distancer(gameState.data.layout)
		self.distancer.getMazeDistances()

		self.walls = gameState.getWalls()
		self.legalpositions = self.getlegal(self.walls)

		self.oldfood = self.getFoodYouAreDefending(gameState)
		self.scared = False # gameState.data.agentStates[self.index].scaredTimer
		self.opponents = self.getOpponents(gameState)

		self.initializebeliefs(gameState)
		self.message = self.beliefs

		self.observeEnabled = False
		self.elapseEnabled = False

		# Feature-based
		self.isOffensive = True
		self.touchHome = False
		self.foodTotal = len(self.getFood(gameState).asList())
		self.foodToReturn = 0

	def getlegal(self, walls):
		legal = []
		for x in range(walls.width):
			for y in range(walls.height):
				if not walls[x][y]:
					legal.append((x,y))
		return legal

	def initializebeliefs(self, gameState):
		self.beliefs = [None] * self.numagents

		for i in range(self.numagents):
			if i in self.opponents:
				self.beliefs[i] = util.Counter()
				for pos in self.legalpositions:
					self.beliefs[i][pos] = 1
				self.beliefs[i].normalize()

	def observe(self, gameState, agent):
		allpossible = util.Counter()
		read = self.distances[agent]
		for pos in self.legalpositions:
			true = util.manhattanDistance(pos, self.currentpos)
			#true = self.getMazeDistance(pos, self.currentpos)
			if gameState.getDistanceProb(true, read) > 0:
				allpossible[pos] = gameState.getDistanceProb(true, read) * self.beliefs[agent][pos]
		allpossible.normalize()
		self.beliefs[agent] = allpossible

	def elapse(self, gameState, agent):
		allpossible = util.Counter()
		for pos in self.legalpositions:
			total = 1
			possiblenew = [pos]
			if (list(pos)[0]+1,list(pos)[1]) in self.legalpositions:
				total += 1
				possiblenew.append((list(pos)[0]+1, list(pos)[1]))
			if (list(pos)[0],list(pos)[1]+1) in self.legalpositions:
				total += 1
				possiblenew.append((list(pos)[0], list(pos)[1]+1))
			if (list(pos)[0]-1,list(pos)[1]) in self.legalpositions:
				total += 1
				possiblenew.append((list(pos)[0]-1, list(pos)[1]))
			if (list(pos)[0],list(pos)[1]-1) in self.legalpositions:
				total += 1
				possiblenew.append((list(pos)[0], list(pos)[1]-1))

			#print "POSSIBLE NEW: {}".format(possiblenew)

			for new in possiblenew:
				if new in allpossible:
					allpossible[new] = allpossible[new] + (1/float(total)) * self.beliefs[agent][pos]
				else:
					allpossible[new] = (1/float(total)) * self.beliefs[agent][pos]
		allpossible.normalize()
		self.beliefs[agent] = allpossible

	def checkeat(self, gameState):
		cord = None
		newfood = self.getFoodYouAreDefending(gameState)
		if newfood != self.oldfood:
			print(newfood.packBits())
		self.oldfood = newfood
		#print cord
		return cord

	def getfeats(self, gameState):
		feats = util.Counter()
		#feats['scared']
		pass

	def communicate(self, teammates):
		for member in teammates:
			member.message = self.beliefs

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

	#################################################
	# FEATURE-BASED PART
	#################################################

	def opponentProbabilityAt(self, pos):
		prob = 0
		for enemy in self.opponents:
			prob += self.beliefs[enemy][pos]
		return prob 

	def getFeatures(self, gameState, action):
		#print "getFeatures ... scaredTimer: {}".format(gameState.data.agentStates[self.index].scaredTimer)
		features = util.Counter()
		successor = self.getSuccessor(gameState, action)
		foodList = self.getFood(successor).asList()    
		ourFoodList = self.getFoodYouAreDefending(successor).asList()
		# next state and position
		myState = successor.getAgentState(self.index)
		myPos = myState.getPosition()

		enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
		ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]

		# Computes distance to invaders we can see
		invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
		features['numInvaders'] = len(invaders)

		features['successorScore'] = -len(foodList) #self.getScore(successor)

		# Compute distance to the nearest food
		if len(foodList) > 0: 
			myPos = successor.getAgentState(self.index).getPosition()
			minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
			features['distanceToFood'] = minDistance

		# Compute the number and distance to ghosts we can see
		features['numGhosts'] = len(ghosts)
		if len(ghosts) > 0:
			dists = [self.getMazeDistance(myPos, a.getPosition()) for a in ghosts]
			features['ghostDistance'] = min(dists)

		# get the sum probability of whether enemies will be there in the next state
		features['enemyProbability'] = self.opponentProbabilityAt(myPos)

		if action == Directions.STOP: features['stop'] = 1
		rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
		if action == rev: features['reverse'] = 1

		# If currently on defense, chase the ghost; if on offense, do the opposite
		
		#if gameState.getAgentState(self.index).isPacman:
		if myState.isPacman:
			#print "DOES THIS GET CALLED"
			features['ghostDistance'] *= (-1) 
			features['enemyProbability'] *= (-100.0)
		else:
			pass
			#features['numInvaders'] = 0
		
		
		return features

	def getWeights(self, gameState, action):

		return {'successorScore': 10, 
				'numInvaders': -10,
				'distanceToFood': -0.1,
				'ghostDistance': 2,
				'enemyProbability': 10.0,
				'stop': -100,
				'reverse': -10}
		"""
		return {'successorScore': 100, 
				'numInvaders': -1000,
				'distanceToFood': -1,
				'ghostDistance': -2,
				'enemyProbability': -2,
				'stop': -100,
				'reverse': -10}
		"""

	def evaluate(self, gameState, action):
		"""
		Computes a linear combination of features and feature weights
		"""
		features = self.getFeatures(gameState, action)
		weights = self.getWeights(gameState, action)
		#print "evaluate : {}".format(features * weights)
		#print features['enemyProbability'] * (-1000.0)
		return features * weights

	def update_belief_from_communication(self):
		new_beliefs = []
		#print len(self.beliefs)
		#print len(self.message)
		#print zip(self.beliefs, self.message)
		for belief_a, belief_b in zip(self.beliefs, self.message):
			if not belief_a or not belief_b:
				new_beliefs.append(None)
				continue

			new_belief = belief_a + belief_b
			new_belief.normalize()
			new_beliefs.append(new_belief)

		self.beliefs = new_beliefs


	def chooseAction(self, gameState):
		actions = gameState.getLegalActions(self.index)
		self.currentstate = gameState.getAgentState(self.index)
		self.currentpos = self.currentstate.getPosition()
		self.distances = gameState.getAgentDistances()

		for enemy in self.opponents:
			# print "FOR {} in {}".format(enemy, self.opponents)
			# check if we can observe any opponents and update beliefs accordingly
			if gameState.getAgentPosition(enemy) is not None:
			#	print "DOESN'T THIS HAPPEN EVERY TIME?"
				self.beliefs[enemy] = util.Counter()
				self.beliefs[enemy][gameState.getAgentPosition(enemy)] = 1.0
			if self.observeEnabled and self.elapseEnabled:
				# elapse time for last agent (if not observable)
				#if enemy == (self.index - 1) % 4:
				self.elapse(gameState, enemy)
				# update based on reading for each agent (if not observable)
				self.observe(gameState, enemy)
			else:
				self.observeEnabled = True
				self.elapseEnabled = True

		# communicate with the teammate and update beliefs from the new message
		self.communicate(self.teammates)
		self.update_belief_from_communication()
		
		for enemy in self.opponents:
			opponent_distribution = self.beliefs[enemy]

			if opponent_distribution.totalCount() == 0:
				self.initializebeliefs(gameState)


		self.displayDistributionsOverPositions(self.beliefs)

		####################################################
		def chooseOffensiveAction(gameState):
			values = [self.evaluate(gameState, a) for a in actions]
			
			# print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

			foodLeft = len(self.getFood(gameState).asList())
			ourFoodLeft = len(self.getFoodYouAreDefending(gameState).asList())
			foodToReturn = self.foodTotal - self.getScore(gameState) - foodLeft
			
			# As the game proceeds, the agent gets more conservative and timid
			# and tries to fetch a fewer number of pellets
			winning_condition = foodLeft <= 2
			carrying_food_threshold = 4 * (float(foodLeft) / self.foodTotal)
			carry_food_back_condition = self.isOffensive and (foodToReturn > carrying_food_threshold)
			#time_running_out_condition = None 	

			home_returning_condition = winning_condition \
									or carry_food_back_condition\
									

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
				bestAction = random.choice(bestActions)

				return bestAction


		def chooseDefensiveAction(gameState):
			scared_condition = gameState.data.agentStates[self.index].scaredTimer > 25
			#print scared_condition

			if scared_condition: # go kamikaze
				return chooseOffensiveAction(gameState)


			opponent_to_go = random.choice(self.opponents)
			for i in self.getOpponents(gameState):
				enemy = gameState.getAgentState(i)
				if enemy.isPacman:
					opponent_to_go = i

			opponent_distribution = self.beliefs[opponent_to_go]
			max_probability = 0
			max_position = self.currentpos # or None?

			for candidate_position, candidate_probability in opponent_distribution.items():
				if candidate_probability > max_probability:
					max_probability = candidate_probability
					max_position = candidate_position

			bestDist = 9999

			target_enemy_position = max_position
			for action in actions:
				successorState = self.getSuccessor(gameState, action)
				myNextPosition = successorState.getAgentPosition(self.index)
				dist = self.getMazeDistance(myNextPosition, target_enemy_position)
				#print "Action {} :: dist from {} to {} is {}".format(action, myNextPosition, target_enemy_position, dist)
				if dist < bestDist:
					bestAction = action
					bestDist = dist

			#print bestAction
			return bestAction


		#####################################################
		self.isOffensive = gameState.getAgentState(self.index).isPacman

		
		if self.isOffensive:
			return chooseOffensiveAction(gameState)
		else:
			return chooseDefensiveAction(gameState)

		return random.choice(actions)



# REFLEX AGENTS

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
		carrying_food_threshold = max(1, int(5 * (float(foodLeft) / self.foodTotal)))
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
				'ghostDistance': 20}

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


class SensorRefinedDefensiveReflexAgent(ReflexCaptureAgent):
	"""
	A reflex agent that keeps its side Pacman-free. Again,
	this is to give you an idea of what a defensive agent
	could be like.  It is not the best or only way to make
	such an agent.
	"""
	def __init__(self, index, teammates = None, timeForComputing=.1):
		CaptureAgent.__init__(self, index, timeForComputing)

		self.teammates = []
		if teammates is not None:
			self.teammates.append(teammates)
		for teammate in self.teammates:
			if len(teammate.teammates) == 0:
				teammate.teammates.append(self)


	def registerInitialState(self, gameState):
		print "SensorRefinedDefensiveReflexAgent"
		ReflexCaptureAgent.registerInitialState(self, gameState)
		self.isOffensive = False

		CaptureAgent.registerInitialState(self, gameState)
		self.numagents = gameState.getNumAgents()
		self.start = gameState.getAgentPosition(self.index)
		#self.myindex =
		#self.mycolor =
		self.distancer = distanceCalculator.Distancer(gameState.data.layout)
		self.distancer.getMazeDistances()

		self.walls = gameState.getWalls()
		self.legalpositions = self.getlegal(self.walls)

		self.oldfood = self.getFoodYouAreDefending(gameState)
		self.scared = False
		self.opponents = self.getOpponents(gameState)

		self.initializebeliefs(gameState)
		self.message = self.beliefs

		self.observeEnabled = False
		self.elapseEnabled = False

		# Feature-based
		self.touchHome = False
		self.foodTotal = len(self.getFood(gameState).asList())
		self.foodToReturn = 0


	def getlegal(self, walls):
		legal = []
		for x in range(walls.width):
			for y in range(walls.height):
				if not walls[x][y]:
					legal.append((x,y))
		return legal

	def initializebeliefs(self, gameState):
		self.beliefs = [None] * self.numagents

		for i in range(self.numagents):
			if i in self.opponents:
				self.beliefs[i] = util.Counter()
				for pos in self.legalpositions:
					self.beliefs[i][pos] = 1
				self.beliefs[i].normalize()

	def observe(self, gameState, agent):
		allpossible = util.Counter()
		read = self.distances[agent]
		for pos in self.legalpositions:
			true = util.manhattanDistance(pos, self.currentpos)
			#true = self.getMazeDistance(pos, self.currentpos)
			if gameState.getDistanceProb(true, read) > 0:
				allpossible[pos] = gameState.getDistanceProb(true, read) * self.beliefs[agent][pos]
		allpossible.normalize()
		self.beliefs[agent] = allpossible


	def update_belief_from_communication(self):
		new_beliefs = []
		#print len(self.beliefs)
		#print len(self.message)
		#print zip(self.beliefs, self.message)
		for belief_a, belief_b in zip(self.beliefs, self.message):
			if not belief_a or not belief_b:
				new_beliefs.append(None)
				continue

			new_belief = belief_a + belief_b
			new_belief.normalize()
			new_beliefs.append(new_belief)

		self.beliefs = new_beliefs


	def elapse(self, gameState, agent):
		allpossible = util.Counter()
		for pos in self.legalpositions:
			total = 1
			possiblenew = [pos]
			if (list(pos)[0]+1,list(pos)[1]) in self.legalpositions:
				total += 1
				possiblenew.append((list(pos)[0]+1, list(pos)[1]))
			if (list(pos)[0],list(pos)[1]+1) in self.legalpositions:
				total += 1
				possiblenew.append((list(pos)[0], list(pos)[1]+1))
			if (list(pos)[0]-1,list(pos)[1]) in self.legalpositions:
				total += 1
				possiblenew.append((list(pos)[0]-1, list(pos)[1]))
			if (list(pos)[0],list(pos)[1]-1) in self.legalpositions:
				total += 1
				possiblenew.append((list(pos)[0], list(pos)[1]-1))

			#print "POSSIBLE NEW: {}".format(possiblenew)

			for new in possiblenew:
				if new in allpossible:
					allpossible[new] = allpossible[new] + (1/float(total)) * self.beliefs[agent][pos]
				else:
					allpossible[new] = (1/float(total)) * self.beliefs[agent][pos]
		allpossible.normalize()
		self.beliefs[agent] = allpossible

	def checkeat(self, gameState):
		cord = None
		newfood = self.getFoodYouAreDefending(gameState)
		if newfood != self.oldfood:
			print(newfood.packBits())
		self.oldfood = newfood
		#print cord
		return cord

	def getfeats(self, gameState):
		feats = util.Counter()
		#feats['scared']
		pass

	def communicate(self, teammates):
		for member in teammates:
			member.message = self.beliefs

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

	def pacmanProbabilityAt(self, gameState, pos):
		prob = 0
		for enemy in self.opponents:
			if gameState.getAgentState(enemy).isPacman:
				prob += self.beliefs[enemy][pos] 

		return prob 

	def ghostProbabilityAt(self, gameState, pos):
		prob = 0
		for enemy in self.opponents:
			if not gameState.getAgentState(enemy).isPacman:
				prob += self.beliefs[enemy][pos] 
		return prob 

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


		features['numInvaders'] = len(invaders)
		if len(invaders) > 0:
			dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
			features['invaderDistance'] = min(dists)


		features['pacmanProbability'] = self.pacmanProbabilityAt(gameState, myPos)

		if action == Directions.STOP: features['stop'] = 1
		rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
		if action == rev: features['reverse'] = 1

		return features

	def getWeights(self, gameState, action):
		a = {'numInvaders': -10,\
						 'onDefense': 100,\
						 'invaderDistance': -0.1,\
						 'pacmanProbability': 100.0,
						 'stop': -10,\
						 'reverse': -2}

		b = {'numInvaders': -1000,\
						 'onDefense': 100000,\
						 'invaderDistance': -10,\
						 'stop': -10,\
						 'reverse': -2}

		return a

	def evaluate(self, gameState, action):
		"""
		Computes a linear combination of features and feature weights
		"""
		features = self.getFeatures(gameState, action)
		weights = self.getWeights(gameState, action)
		#print "evaluate : {}".format(features * weights)
		return features * weights

	def chooseAction(self, gameState):

		actions = gameState.getLegalActions(self.index)
		self.currentstate = gameState.getAgentState(self.index)
		self.currentpos = self.currentstate.getPosition()
		self.distances = gameState.getAgentDistances()

		for enemy in self.opponents:
			# check if we can observe any opponents and update beliefs accordingly
			if gameState.getAgentPosition(enemy) is not None:
				self.beliefs[enemy] = util.Counter()
				self.beliefs[enemy][gameState.getAgentPosition(enemy)] = 1.0
			if self.observeEnabled and self.elapseEnabled:
				self.elapse(gameState, enemy)
				self.observe(gameState, enemy)
			else:
				self.observeEnabled = True
				self.elapseEnabled = True


		self.communicate(self.teammates)
		self.update_belief_from_communication()
		
		for enemy in self.opponents:
			opponent_distribution = self.beliefs[enemy]

			if opponent_distribution.totalCount() == 0:
				self.initializebeliefs(gameState)

		opponent_to_go = None
		for i in self.getOpponents(gameState):
			enemy = gameState.getAgentState(i)
			if enemy.isPacman:
				opponent_to_go = i


		if opponent_to_go is None:
			values = [self.evaluate(gameState, a) for a in actions]

			maxValue = max(values)
			bestActions = [a for a, v in zip(actions, values) if v == maxValue]
			bestAction = random.choice(bestActions)

			return bestAction


		else:
			opponent_distribution = self.beliefs[opponent_to_go]

			max_probability = -1
			max_position = self.currentpos # or None?

			for candidate_position, candidate_probability in opponent_distribution.items():
				if candidate_probability > max_probability:
					max_probability = candidate_probability
					max_position = candidate_position

			bestDist = float('inf')

			target_enemy_position = max_position
			for action in actions:
				successorState = self.getSuccessor(gameState, action)
				myNextPosition = successorState.getAgentPosition(self.index)
				dist = self.getMazeDistance(myNextPosition, target_enemy_position)
				if dist < bestDist:
					bestAction = action
					bestDist = dist

			
			return bestAction
		


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

