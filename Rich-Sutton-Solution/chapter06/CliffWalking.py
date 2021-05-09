import numpy as np
import matplotlib.pyplot as plt

# world height
WORLD_HEIGHT = 4

# world width
WORLD_WIDTH = 12

# probability for exploration
EPSILON = 0.1

# step size
ALPHA = 0.5

# gamma for Q-Learning and Expected Sarsa
GAMMA = 1

# all possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
actions = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]

# initial state action pair values
stateActionValues = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
startState = [3, 0]
goalState = [3, 11]

# reward for each action in each state
actionRewards = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
actionRewards[:, :, :] = -1.0
actionRewards[2, 1:11, ACTION_DOWN] = -100.0
actionRewards[3, 0, ACTION_RIGHT] = -100.0

# set up destinations for each action in each state
actionDestination = []
for i in range(0, WORLD_HEIGHT):
    actionDestination.append([])
    for j in range(0, WORLD_WIDTH):
        destinaion = dict()
        destinaion[ACTION_UP] = [max(i - 1, 0), j]
        destinaion[ACTION_LEFT] = [i, max(j - 1, 0)]
        destinaion[ACTION_RIGHT] = [i, min(j + 1, WORLD_WIDTH - 1)]
        if i == 2 and 1 <= j <= 10:
            destinaion[ACTION_DOWN] = startState
        else:
            destinaion[ACTION_DOWN] = [min(i + 1, WORLD_HEIGHT - 1), j]
        actionDestination[-1].append(destinaion)
actionDestination[3][0][ACTION_RIGHT] = startState
#print(actionDestination)
"""
[[{0: [0, 0], 1: [1, 0], 2: [0, 0], 3: [0, 1]}, {0: [0, 1], 1: [1, 1], 2: [0, 0], 3: [0, 2]}, {0: [0, 2], 1: [1, 2], 2: [0, 1], 3: [0, 3]}, {0: [0, 3], 1: [1, 3], 2: [0, 2], 3: [0, 4]}, {0: [0, 4], 1: [1, 4], 2: [0, 3], 3: [0, 5]}, {0: [0, 5], 1: [1, 5], 2: [0, 4], 3: [0, 6]}, {0: [0, 6], 1: [1, 6], 2: [0, 5], 3: [0, 7]}, {0: [0, 7], 1: [1, 7], 2: [0, 6], 3: [0, 8]}, {0: [0, 8], 1: [1, 8], 2: [0, 7], 3: [0, 9]}, {0: [0, 9], 1: [1, 9], 2: [0, 8], 3: [0, 10]}, {0: [0, 10], 1: [1, 10], 2: [0, 9], 3: [0, 11]}, {0: [0, 11], 1: [1, 11], 2: [0, 10], 3: [0, 11]}], [{0: [0, 0], 1: [2, 0], 2: [1, 0], 3: [1, 1]}, {0: [0, 1], 1: [2, 1], 2: [1, 0], 3: [1, 2]}, {0: [0, 2], 1: [2, 2], 2: [1, 1], 3: [1, 3]}, {0: [0, 3], 1: [2, 3], 2: [1, 2], 3: [1, 4]}, {0: [0, 4], 1: [2, 4], 2: [1, 3], 3: [1, 5]}, {0: [0, 5], 1: [2, 5], 2: [1, 4], 3: [1, 6]}, {0: [0, 6], 1: [2, 6], 2: [1, 5], 3: [1, 7]}, {0: [0, 7], 1: [2, 7], 2: [1, 6], 3: [1, 8]}, {0: [0, 8], 1: [2, 8], 2: [1, 7], 3: [1, 9]}, {0: [0, 9], 1: [2, 9], 2: [1, 8], 3: [1, 10]}, {0: [0, 10], 1: [2, 10], 2: [1, 9], 3: [1, 11]}, {0: [0, 11], 1: [2, 11], 2: [1, 10], 3: [1, 11]}], [{0: [1, 0], 1: [3, 0], 2: [2, 0], 3: [2, 1]}, {0: [1, 1], 1: [3, 0], 2: [2, 0], 3: [2, 2]}, {0: [1, 2], 1: [3, 0], 2: [2, 1], 3: [2, 3]}, {0: [1, 3], 1: [3, 0], 2: [2, 2], 3: [2, 4]}, {0: [1, 4], 1: [3, 0], 2: [2, 3], 3: [2, 5]}, {0: [1, 5], 1: [3, 0], 2: [2, 4], 3: [2, 6]}, {0: [1, 6], 1: [3, 0], 2: [2, 5], 3: [2, 7]}, {0: [1, 7], 1: [3, 0], 2: [2, 6], 3: [2, 8]}, {0: [1, 8], 1: [3, 0], 2: [2, 7], 3: [2, 9]}, {0: [1, 9], 1: [3, 0], 2: [2, 8], 3: [2, 10]}, {0: [1, 10], 1: [3, 0], 2: [2, 9], 3: [2, 11]}, {0: [1, 11], 1: [3, 11], 2: [2, 10], 3: [2, 11]}], [{0: [2, 0], 1: [3, 0], 2: [3, 0], 3: [3, 0]}, {0: [2, 1], 1: [3, 1], 2: [3, 0], 3: [3, 2]}, {0: [2, 2], 1: [3, 2], 2: [3, 1], 3: [3, 3]}, {0: [2, 3], 1: [3, 3], 2: [3, 2], 3: [3, 4]}, {0: [2, 4], 1: [3, 4], 2: [3, 3], 3: [3, 5]}, {0: [2, 5], 1: [3, 5], 2: [3, 4], 3: [3, 6]}, {0: [2, 6], 1: [3, 6], 2: [3, 5], 3: [3, 7]}, {0: [2, 7], 1: [3, 7], 2: [3, 6], 3: [3, 8]}, {0: [2, 8], 1: [3, 8], 2: [3, 7], 3: [3, 9]}, {0: [2, 9], 1: [3, 9], 2: [3, 8], 3: [3, 10]}, {0: [2, 10], 1: [3, 10], 2: [3, 9], 3: [3, 11]}, {0: [2, 11], 1: [3, 11], 2: [3, 10], 3: [3, 11]}]]
"""

# choose an action based on epsilon greedy algorithm
def chooseAction(state, stateActionValues):
    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(actions)
    else:
        values_ = stateActionValues[state[0], state[1], :]
        return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])
        # if all stateActionValues are zero then it return all the actions i.e. [0,1,2,3]

# an episode with Sarsa
# @stateActionValues: values for state action pair, will be updated
# @expected: if True, will use expected Sarsa algorithm
# @stepSize: step size for updating
# @return: total rewards within this episode
def sarsa(stateActionValues, expected=False, stepSize=ALPHA):
    currentState = startState
    currentAction = chooseAction(currentState, stateActionValues)
    rewards = 0.0
    while currentState != goalState:
        newState = actionDestination[currentState[0]][currentState[1]][currentAction]
        newAction = chooseAction(newState, stateActionValues)
        reward = actionRewards[currentState[0], currentState[1], currentAction]
        rewards += reward
        if not expected:
            valueTarget = stateActionValues[newState[0], newState[1], newAction]
        else:
            # calculate the expected value of new state
            valueTarget = 0.0
            actionValues = stateActionValues[newState[0], newState[1], :]
            bestActions = np.argwhere(actionValues == np.max(actionValues))
            #print(bestActions)
            for action in actions:
                if action in bestActions:
                    valueTarget += ((1.0 - EPSILON) / len(bestActions) + EPSILON / len(actions)) * stateActionValues[newState[0], newState[1], action]
                else:
                    valueTarget += EPSILON / len(actions) * stateActionValues[newState[0], newState[1], action]
        valueTarget *= GAMMA
        # Sarsa update
        stateActionValues[currentState[0], currentState[1], currentAction] += stepSize * (reward +
            valueTarget - stateActionValues[currentState[0], currentState[1], currentAction])
        currentState = newState
        currentAction = newAction
    return rewards

# an episode with Q-Learning
# @stateActionValues: values for state action pair, will be updated
# @expected: if True, will use expected Sarsa algorithm
# @stepSize: step size for updating
# @return: total rewards within this episode
def qLearning(stateActionValues, stepSize=ALPHA):
    currentState = startState
    rewards = 0.0
    while currentState != goalState:
        currentAction = chooseAction(currentState, stateActionValues)
        reward = actionRewards[currentState[0], currentState[1], currentAction]
        rewards += reward
        newState = actionDestination[currentState[0]][currentState[1]][currentAction]
        # Q-Learning update
        stateActionValues[currentState[0], currentState[1], currentAction] += stepSize * (
            reward + GAMMA * np.max(stateActionValues[newState[0], newState[1], :]) -
            stateActionValues[currentState[0], currentState[1], currentAction])
        currentState = newState
    return rewards

# print optimal policy
def printOptimalPolicy(stateActionValues):
    optimalPolicy = []
    for i in range(0, WORLD_HEIGHT):
        optimalPolicy.append([])
        for j in range(0, WORLD_WIDTH):
            if [i, j] == goalState:
                optimalPolicy[-1].append('G')
                continue
            bestAction = np.argmax(stateActionValues[i, j, :])
            if bestAction == ACTION_UP:
                optimalPolicy[-1].append('U')
            elif bestAction == ACTION_DOWN:
                optimalPolicy[-1].append('D')
            elif bestAction == ACTION_LEFT:
                optimalPolicy[-1].append('L')
            elif bestAction == ACTION_RIGHT:
                optimalPolicy[-1].append('R')
    for row in optimalPolicy:
        print(row)

# figure 6.5
# Use 20 independent runs instead of a single run to draw the figure
# Actually with a single run I failed to present a smooth curve
# However the optimal policy converges well with a single run
# Sarsa converges to the safe path, while Q-Learning converges to the optimal path
def figure6_5():
    # averaging the reward sums from 10 successive episodes
    averageRange = 10

    # episodes of each run
    nEpisodes = 500

    # perform 20 independent runs
    runs = 20

    rewardsSarsa = np.zeros(nEpisodes)
    rewardsQLearning = np.zeros(nEpisodes)
    for run in range(0, runs):
        stateActionValuesSarsa = np.copy(stateActionValues)
        stateActionValuesQLearning = np.copy(stateActionValues)
        for i in range(0, nEpisodes):
            # cut off the value by -100 to draw the figure more elegantly
            rewardsSarsa[i] += max(sarsa(stateActionValuesSarsa), -100)
            rewardsQLearning[i] += max(qLearning(stateActionValuesQLearning), -100)

    # averaging over independt runs
    rewardsSarsa /= runs
    rewardsQLearning /= runs

    # averaging over successive episodes
    smoothedRewardsSarsa = np.copy(rewardsSarsa)
    smoothedRewardsQLearning = np.copy(rewardsQLearning)
    for i in range(averageRange, nEpisodes):
        smoothedRewardsSarsa[i] = np.mean(rewardsSarsa[i - averageRange: i + 1])
        smoothedRewardsQLearning[i] = np.mean(rewardsQLearning[i - averageRange: i + 1])

    # display optimal policy
    print('Sarsa Optimal Policy:')
    printOptimalPolicy(stateActionValuesSarsa)
    print('Q-Learning Optimal Policy:')
    printOptimalPolicy(stateActionValuesQLearning)

    # draw reward curves
    plt.figure(1)
    plt.plot(smoothedRewardsSarsa, label='Sarsa')
    plt.plot(smoothedRewardsQLearning, label='Q-Learning')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.legend()

# Due to limited capacity of calculation of my machine, I can't complete this experiment
# with 100,000 episodes and 50,000 runs to get the fully averaged performance
# However even I only play for 1,000 episodes and 10 runs, the curves looks still good.
def figure6_7():
    stepSizes = np.arange(0.1, 1.1, 0.1)
    nEpisodes = 1000
    runs = 10

    ASY_SARSA = 0
    ASY_EXPECTED_SARSA = 1
    ASY_QLEARNING = 2
    INT_SARSA = 3
    INT_EXPECTED_SARSA = 4
    INT_QLEARNING = 5
    methods = range(0, 6)

    performace = np.zeros((6, len(stepSizes)))
    for run in range(0, runs):
        for ind, stepSize in zip(range(0, len(stepSizes)), stepSizes):
            stateActionValuesSarsa = np.copy(stateActionValues)
            stateActionValuesExpectedSarsa = np.copy(stateActionValues)
            stateActionValuesQLearning = np.copy(stateActionValues)
            for ep in range(0, nEpisodes):
                print('run:', run, 'step size:', stepSize, 'episode:', ep)
                sarsaReward = sarsa(stateActionValuesSarsa, expected=False, stepSize=stepSize)
                expectedSarsaReward = sarsa(stateActionValuesExpectedSarsa, expected=True, stepSize=stepSize)
                qLearningReward = qLearning(stateActionValuesQLearning, stepSize=stepSize)
                performace[ASY_SARSA, ind] += sarsaReward
                performace[ASY_EXPECTED_SARSA, ind] += expectedSarsaReward
                performace[ASY_QLEARNING, ind] += qLearningReward

                if ep < 100:
                    performace[INT_SARSA, ind] += sarsaReward
                    performace[INT_EXPECTED_SARSA, ind] += expectedSarsaReward
                    performace[INT_QLEARNING, ind] += qLearningReward

    performace[:3, :] /= nEpisodes * runs
    performace[3:, :] /= runs * 100
    labels = ['Asymptotic Sarsa', 'Asymptotic Expected Sarsa', 'Asymptotic Q-Learning',
              'Interim Sarsa', 'Interim Expected Sarsa', 'Interim Q-Learning']
    plt.figure(2)
    for method, label in zip(methods, labels):
        plt.plot(stepSizes, performace[method, :], label=label)
    plt.xlabel('alpha')
    plt.ylabel('reward per episode')
    plt.legend()

# Drawing figure 6.7 may take a while
#figure6_7()

figure6_5()
plt.show()
"""
Sarsa Optimal Policy:
['R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'D']
['R', 'U', 'U', 'R', 'U', 'R', 'U', 'U', 'R', 'U', 'R', 'D']
['U', 'U', 'U', 'U', 'R', 'U', 'L', 'R', 'U', 'U', 'R', 'D']
['U', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'G']
Q-Learning Optimal Policy:
['U', 'R', 'R', 'R', 'R', 'R', 'R', 'D', 'R', 'R', 'R', 'D']
['U', 'R', 'R', 'R', 'R', 'D', 'D', 'R', 'R', 'R', 'R', 'D']
['R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'D']
['U', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'U', 'G']
After an initial transient, Q-learning
learns values for the optimal policy, that which travels right along the edge of the
cliff. Unfortunately, this results in its occasionally falling off the cliff because of
the ε-greedy action selection. Sarsa, on the other hand, takes the action selection
into account and learns the longer but safer path through the upper part of the
grid. Although Q-learning actually learns the values of the optimal policy, its on-
line performance is worse than that of Sarsa, which learns the roundabout policy.
Of course, if ε were gradually reduced, then both methods would asymptotically
converge to the optimal policy.
"""
