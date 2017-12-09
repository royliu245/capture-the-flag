class InferenceModule:
    def __init__(self, opponentAgent):
        "Sets the ghost agent for later access"
        self.opponentAgent = opponentAgent
        self.index = opponentAgent.index
        self.obs = [] # most recent observation position

class ExactInference(InferenceModule):
    def __init__(self):
        self.beliefs = None
        

    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()


    def getObservationDistribution(noisyDistance):
        """
        Returns the factor P( noisyDistance | TrueDistances ), the likelihood of the provided noisyDistance
        conditioned upon all the possible true distances that could have generated it.
        """
        global observationDistributions
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
            successorPosition = game.Actions.getSuccessor(ghostPosition, action)
            dist[successorPosition] = prob
        return dist

    def setGhostPosition(self, gameState, ghostPosition):
        """
        Sets the position of the ghost for this inference module to the
        specified position in the supplied gameState.

        Note that calling setGhostPosition does not change the position of the
        ghost in the GameState object used for tracking the true progression of
        the game.  The code in inference.py only ever receives a deep copy of
        the GameState object which is responsible for maintaining game state,
        not a reference to the original object.  Note also that the ghost
        distance observations are stored at the time the GameState object is
        created, so changing the position of the ghost will not affect the
        functioning of observeState.
        """
        conf = game.Configuration(ghostPosition, game.Directions.STOP)
        gameState.data.agentStates[self.index] = game.AgentState(conf, False)
        return gameState

    def observe(self, observation, gameState):

        noisyDistance = observation
        emissionModel = self.getObservationDistribution(noisyDistance)
        myPosition = gameState.getPosition()

        allPossible = util.Counter()
        if noisyDistance is None: # HANDLE THIS LATER
            return

        else:
            for p in self.legalPositions:
                trueDistance = self.getMazeDistance(p, myPosition)
                probNoisyDistanceGivenTrueDistance = emissionModel[trueDistance]
                
                if emissionModel[trueDistance] > 0:
                    allPossible[p] = self.beliefs[p] * probNoisyDistanceGivenTrueDistance

        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        updatedBeliefs = util.Counter()
        for oldPos in self.legalPositions:
            newPosDist = self.getPositionDistribution(\
                        self.setGhostPosition(gameState, oldPos))
            for newPos, prob in newPosDist.items(): 
                updatedBeliefs[newPos] += self.beliefs[oldPos] * prob
                # P(ghost at newPos @ t+1 after being at oldPos @ t) 
                #  = P(ghost at oldPos @ t) * P(ghost at newPos @ t+1|at oldPos @ t)

        updatedBeliefs.normalize()
        self.beliefs = updatedBeliefs


    def getBeliefDistribution(self):
        return self.beliefs

