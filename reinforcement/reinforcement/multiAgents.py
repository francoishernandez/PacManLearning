# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util
from graphicsDisplay import saveData

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        foodScore = currentGameState.getNumFood() - successorGameState.getNumFood()
        safetyScore = 0
        ghostScore = 0
        distanceScore = 0

        ppos = successorGameState.getPacmanPosition()
        for i in range(0, len(successorGameState.getGhostPositions())):
            gpos = successorGameState.getGhostPositions()[i]
            isScared = newScaredTimes[i] > 0
            if manhattanDistance(gpos, ppos) < 2 and not isScared:
                safetyScore += float('-infinity')
            elif manhattanDistance(gpos, ppos) < 2 and isScared:
                ghostScore += 100

        return foodScore + safetyScore + ghostScore + distanceScore * 0.5

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    def minimaxSearch(self, currentGameState, depth, agent):
        if currentGameState.isWin() or currentGameState.isLose() or depth == self.depth:
            return None, self.evaluationFunction(currentGameState)
        else:
            bestAction = None
            if agent == 0:
                bestValue = float('-inf')
            else:
                bestValue = float('inf')

            for action in currentGameState.getLegalActions(agent):
                gameState = currentGameState.generateSuccessor(agent, action)
                nextAgent = (agent + 1) % currentGameState.getNumAgents()

                if agent == 0:
                    newValue = self.minimaxSearch(gameState, depth, nextAgent)[1]
                    if newValue > bestValue:
                        bestAction, bestValue = action, newValue

                elif nextAgent == 0:
                    newValue = self.minimaxSearch(gameState, depth+1, nextAgent)[1]
                    if newValue < bestValue:
                        bestAction, bestValue = action, newValue

                else:
                    newValue = self.minimaxSearch(gameState, depth, nextAgent)[1]
                    if newValue < bestValue:
                        bestAction, bestValue = action, newValue

            return bestAction, bestValue


    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        agent = 0
        depth = 0
        bestAction, bestValue = self.minimaxSearch(gameState, depth, agent)
        return bestAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def alphabetaSearch(self, currentGameState, depth, agent, alpha, beta):
        if currentGameState.isWin() or currentGameState.isLose() or depth == self.depth:
            return None, self.evaluationFunction(currentGameState)
        else:
            bestAction = None
            if agent == 0:
                bestValue = float('-inf')
            else:
                bestValue = float('inf')

            for action in currentGameState.getLegalActions(agent):
                gameState = currentGameState.generateSuccessor(agent, action)
                nextAgent = (agent + 1) % currentGameState.getNumAgents()

                if agent == 0:
                    a, newValue = self.alphabetaSearch(gameState, depth, nextAgent, alpha, beta)
                    if newValue >= bestValue:
                        bestAction, bestValue = action, newValue
                    if bestValue > beta:
                        return bestAction, bestValue
                    alpha = max(alpha, bestValue)

                elif nextAgent == 0:
                    a, newValue = self.alphabetaSearch(gameState, depth+1, nextAgent, alpha, beta)
                    if newValue <= bestValue:
                        bestAction, bestValue = action, newValue
                    if bestValue < alpha:
                        return bestAction, bestValue
                    beta = min(beta, bestValue)

                else:
                    a, newValue = self.alphabetaSearch(gameState, depth, nextAgent, alpha, beta)
                    if newValue <= bestValue:
                        bestAction, bestValue = action, newValue
                    if bestValue < alpha:
                        return bestAction, bestValue
                    beta = min(beta, bestValue)

            return bestAction, bestValue


    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        agent = 0
        depth = 0
        alpha = float('-inf')
        beta = float('inf')
        bestAction, bestValue = self.alphabetaSearch(gameState, depth, agent, alpha, beta)
        return bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def expectimaxSearch(self, currentGameState, depth, agent):
        if currentGameState.isWin() or currentGameState.isLose() or depth == self.depth:
            return None, self.evaluationFunction(currentGameState)
        else:
            bestAction = None
            if agent == 0:
                bestValue = float('-inf')
            else:
                bestValue = 0

            p = 1/ len(currentGameState.getLegalActions(agent))
            for action in currentGameState.getLegalActions(agent):
                gameState = currentGameState.generateSuccessor(agent, action)
                nextAgent = (agent + 1) % currentGameState.getNumAgents()

                if agent == 0:
                    a, newValue = self.expectimaxSearch(gameState, depth, nextAgent)
                    if newValue >= bestValue:
                        bestAction, bestValue = action, newValue

                elif nextAgent == 0:
                    bestValue += p * self.expectimaxSearch(gameState, depth+1, nextAgent)[1]

                else:
                    bestValue += p * self.expectimaxSearch(gameState, depth, nextAgent)[1]

            return bestAction, bestValue


    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        agent = 0
        depth = 0
        bestAction, bestValue = self.expectimaxSearch(gameState, depth, agent)

        # Call to the saveData method to add moves generated with expectimax to the training set
        saveData(bestAction, self.getExpectedReward(gameState, bestAction))

        return bestAction

    def getExpectedReward(self, gameState, action):
        nextGameState = gameState.generateSuccessor(0, action)
        return nextGameState.getScore() - gameState.getScore()


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    return currentGameState.getScore()

# Abbreviation
better = betterEvaluationFunction

