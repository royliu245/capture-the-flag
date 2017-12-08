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
import distanceCalculator

from util import nearestPoint
from util import Counter


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='ParentAgent', second='ParentAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    # The following line is an example only; feel free to change it.

    memberone = eval(first)(firstIndex)
    membertwo = eval(second)(secondIndex, memberone)
    return [memberone, membertwo]

    #return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ParentAgent(CaptureAgent):
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
        CaptureAgent.registerInitialState(self, gameState)
        self.numagents = gameState.getNumAgents()
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

        # for enemy in self.getOpponents(gameState):

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


    def chooseAction(self, gameState):

        actions = gameState.getLegalActions(self.index)
        self.currentstate = gameState.getAgentState(self.index)
        self.currentpos = self.currentstate.getPosition()
        self.distances = gameState.getAgentDistances()


        #update beliefs from message
        #self.beliefs = self.message

        for enemy in self.opponents:
            print "FOR {} in {}".format(enemy, self.opponents)
            # check if we can observe any opponents and update beliefs accordingly
            if gameState.getAgentPosition(enemy) is not None:
                print "DOESN'T THIS HAPPEN EVERY TIME?"
                self.beliefs[enemy] = util.Counter()
                self.beliefs[enemy][gameState.getAgentPosition(enemy)] = 1.0
            else:
                # elapse time for last agent (if not observable)
                #if enemy == (self.index - 1) % 4:
                self.elapse(gameState, enemy)
                # update based on reading for each agent (if not observable)
                self.observe(gameState, enemy)


        self.communicate(self.teammates)

        #self.debugDraw(self.walls.asList(),[1,0,0])
        """
        print "I am agent:", self.index, "I am:", self
        print "My Teammates Are:", self.teammates
        print "Am i red?", self.red
        print "My state:", self.currentstate
        print "My location is:", self.currentpos
        print "Distance Readings:", self.distances
        print "Agent #0 pos", gameState.getAgentPosition(0)
        print "Agent #1 pos", gameState.getAgentPosition(1)
        print "Agent #2 pos", gameState.getAgentPosition(2)
        print "Agent #3 pos", gameState.getAgentPosition(3)
        print "The time is:", time.clock()
        """
        #cord = self.checkeat(gameState)
        #if cord is not None:
            #print "YOLO", cord

        for ghost in self.beliefs:
            if ghost is not None:
                print ghost.totalCount()

        self.displayDistributionsOverPositions(self.beliefs)



        #return random.choice(actions)

        opponent_to_go = random.choice(self.opponents)
        #print "Opponent to go: {}".format(opponent_to_go)
        #opponent_to_go = self.opponents[0]

        opponent_distribution = self.beliefs[opponent_to_go]
        #print opponent_distribution
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
            print "Action {} :: dist from {} to {} is {}".format(action, myNextPosition, target_enemy_position, dist)
            if dist < bestDist:
                bestAction = action
                bestDist = dist

        print bestAction
        return bestAction

        return random.choice(actions)