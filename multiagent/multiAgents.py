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

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        foodList = newFood.asList()
        fooddistance = [manhattanDistance(newPos, food) for food in foodList]
        ghostdistance = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        
        score = successorGameState.getScore()

        if fooddistance:
            score += 1 / (min(fooddistance) + 1)

        if ghostdistance:
            if min(ghostdistance) <= 1:
                score -= 1000
            else:
                score -= 1 / (sum(ghostdistance) + 1)
        return score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        number = gameState.getNumAgents()
        def MINIMAX(state,agentIndex,depth):
            v_max=float('-inf')
            v_min =float('inf')
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            nextagent = (agentIndex + 1) % number
            nextdepth = depth + 1 if nextagent == 0 else depth
            if agentIndex ==0:
                for i in state.getLegalActions(agentIndex):
                    v_max= max(v_max,MINIMAX(state.generateSuccessor(agentIndex,i),nextagent,nextdepth))
            else:
                for i in state.getLegalActions(agentIndex):
                    v_min = min(v_min,MINIMAX(state.generateSuccessor(agentIndex,i),nextagent,nextdepth))
            return v_max if agentIndex==0 else v_min
        score = float('-inf')
        bestaction = None
        for i in gameState.getLegalActions(0):
            actionscore = MINIMAX(gameState.generateSuccessor(0,i),1,0)
            score = max(score,actionscore)
            if score == actionscore:
                bestaction = i
        return bestaction
            
        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        number = gameState.getNumAgents()
        def MINIMAX(state,agentIndex,depth,alpha=float('-inf'),beta=float('inf')):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            nextagent = (agentIndex + 1) % number
            nextdepth = depth + 1 if nextagent == 0 else depth
            if agentIndex ==0:
                value = float('-inf')
                for i in state.getLegalActions(agentIndex):
                    value= max(value,MINIMAX(state.generateSuccessor(agentIndex,i),nextagent,nextdepth,alpha,beta))
                    if value>beta:
                        return value
                    alpha = max(alpha,value)
            else:
                value = float('inf')
                for i in state.getLegalActions(agentIndex):
                    value = min(value,MINIMAX(state.generateSuccessor(agentIndex,i),nextagent,nextdepth,alpha,beta))
                    if value<alpha:
                        return value
                    beta = min(beta,value)
            return value
        alpha, score, bestaction = float('-inf'), float('-inf'), None
        for i in gameState.getLegalActions(0):
            actionscore = MINIMAX(gameState.generateSuccessor(0, i), 1, 0, alpha, float('inf'))
            if actionscore > score:                   # 严格大于，保留第一个最优动作
               score, bestaction = actionscore, i
            alpha = max(alpha, actionscore)
        return bestaction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        number = gameState.getNumAgents()
        def expectimax(state,agentIndex,depth):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            nextagent = (agentIndex + 1) % number
            nextdepth = depth + 1 if nextagent == 0 else depth
            if agentIndex ==0:
                value = float('-inf')
                for i in state.getLegalActions(0):
                    value = max(value,expectimax(state.generateSuccessor(agentIndex,i),nextagent,nextdepth))
                return value
            else:
                value = 0
                count = len(state.getLegalActions(agentIndex))
                for i in state.getLegalActions(agentIndex):
                    value += expectimax(state.generateSuccessor(agentIndex,i),nextagent,nextdepth)/count
                return value
        score = float('-inf')   
        bestaction = None
        for i in gameState.getLegalActions(0):
            actionscore = expectimax(gameState.generateSuccessor(0,i),1,0)
            score = max(score,actionscore)
            if score == actionscore:
                bestaction = i
        return bestaction
        

        

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return float('-inf')

    score = currentGameState.getScore()
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood().asList()
    capsules = currentGameState.getCapsules()
    ghosts = currentGameState.getGhostStates()

    # 尽量吃完食物，同时向最近的食物移动
    score -= 4 * len(food)
    if food:
        nearestFood = min(manhattanDistance(pos, f) for f in food)
        score += 10 / (nearestFood + 1)

    # 鼓励消耗能量豆
    score -= 15 * len(capsules)

    dangerNearby = False
    for ghost in ghosts:
        distance = manhattanDistance(pos, ghost.getPosition())

        if ghost.scaredTimer > 0:
            # 恐惧时间足够时，鼓励靠近幽灵
            if distance < ghost.scaredTimer:
                score += 30 / (distance + 1)
        else:
            if distance <= 1:
                score -= 200
            elif distance < 4:
                score -= 20 / distance

            if distance < 4:
                dangerNearby = True

    # 附近有危险幽灵时，优先靠近能量豆
    if dangerNearby and capsules:
        nearestCapsule = min(
            manhattanDistance(pos, capsule)
            for capsule in capsules
        )
        score += 30 / (nearestCapsule + 1)

    return score
    

# Abbreviation
better = betterEvaluationFunction
