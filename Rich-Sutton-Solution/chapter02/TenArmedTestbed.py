"""
 To roughly assess the relative effectiveness of the greedy and ε-greedy methods, we compared them numerically on a suite of test problems.
 This was a set of 2000 randomly generated k-armed bandit problems with k = 10. For each bandit problem, such as that shown in Figure 2.1,
 the action values, q∗(a), a = 1, . . . , 10, were selected according to a normal (Gaussian) distribution with mean 0 and variance 1.
 Then, when a learning method applied to that problem selected action A(t) at time t, the actual reward R(t) was selected from a normal
 distribution with mean q∗(A(t)) and variance 1. It is these distributions which are shown as gray in Figure 2.1. We call this suite of
 test tasks the 10-armed testbed. For any learning method, we can measure its performance and behavior as it improves with experience over
 1000 steps interacting with one of the bandit problem. This makes up one run. Repeating this for 2000 independent runs with a different
 bandit problem, we obtained measures of the learning algorithm’s average behavior.
"""

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class Bandit:
    # @kArm: # of arms
    # @epsilon: probability for exploration in epsilon-greedy algorithm
    # @initial: initial estimation for each action
    # @stepSize: constant step size for updating estimations
    # @sampleAverages: if True, use sample averages to update estimations instead of constant step size
    # @UCB: if not None, use UCB algorithm to select action
    # @gradient: if True, use gradient based bandit algorithm
    # @gradientBaseline: if True, use average reward as baseline for gradient based bandit algorithm
    def __init__(self, kArm=10, epsilon=0., initial=0., stepSize=0.1, sampleAverages=False, UCBParam=None,
                 gradient=False, gradientBaseline=False, trueReward=0.):
        self.k = kArm
        self.stepSize = stepSize
        self.sampleAverages = sampleAverages
        self.indices = np.arange(self.k)
        self.time = 0
        self.UCBParam = UCBParam
        self.gradient = gradient
        self.gradientBaseline = gradientBaseline
        self.averageReward = 0
        self.trueReward = trueReward

        # real reward for each action
        self.qTrue = []

        # estimation for each action
        self.qEst = np.zeros(self.k)

        # # of chosen times for each action
        self.actionCount = []

        self.epsilon = epsilon

        # initialize real rewards with N(0,1) distribution and estimations with desired initial value
        for i in range(0, self.k):
            self.qTrue.append(np.random.randn() + trueReward)
            self.qEst[i] = initial
            self.actionCount.append(0)

        self.bestAction = np.argmax(self.qTrue)

    # get an action for this bandit, explore or exploit?
    def getAction(self):
        # explore
        if self.epsilon > 0:
            if np.random.binomial(1, self.epsilon) == 1:
                return np.random.choice(self.indices)

        # exploit
        if self.UCBParam is not None:
            UCBEst = self.qEst + \
                     self.UCBParam * np.sqrt(np.log(self.time + 1) / (np.asarray(self.actionCount) + 1))
            print(UCBEst)
            return np.argmax(UCBEst)
        if self.gradient:
            expEst = np.exp(self.qEst)
            self.actionProb = expEst / np.sum(expEst)
            return np.random.choice(self.indices, p=self.actionProb)
        return np.argmax(self.qEst)

    # take an action, update estimation for this action
    def takeAction(self, action):
        # generate the reward under N(real reward, 1)
        reward = np.random.randn() + self.qTrue[action]
        self.time += 1
        self.averageReward = (self.time - 1.0) / self.time * self.averageReward + reward / self.time #(4) / 5 * 2 + (4 / 5)
        self.actionCount[action] += 1

        if self.sampleAverages:
            # update estimation using sample averages
            self.qEst[action] += 1.0 / self.actionCount[action] * (reward - self.qEst[action])
        elif self.gradient:
            oneHot = np.zeros(self.k)
            oneHot[action] = 1
            if self.gradientBaseline:
                baseline = self.averageReward
            else:
                baseline = 0
            self.qEst = self.qEst + self.stepSize * (reward - baseline) * (oneHot - self.actionProb)
            #self.qEst[action] += self.stepSize * (reward - baseline) * (oneHot[action] - self.actionProb[action])
        else:
            # update estimation with constant step size
            self.qEst[action] += self.stepSize * (reward - self.qEst[action])
        return reward

figureIndex = 0

# for figure 2.1
def figure2_1():
    global figureIndex
    plt.figure(figureIndex)
    figureIndex += 1
    sns.violinplot(data=np.random.randn(200,10) + np.random.randn(10))
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")

def banditSimulation(nBandits, time, bandits):
    bestActionCounts = [np.zeros(time, dtype='float') for _ in range(0, len(bandits))]
    averageRewards = [np.zeros(time, dtype='float') for _ in range(0, len(bandits))]
    for banditInd, bandit in enumerate(bandits):
        for i in range(0, nBandits):
            for t in range(0, time):
                action = bandit[i].getAction()
                reward = bandit[i].takeAction(action)
                averageRewards[banditInd][t] += reward
                if action == bandit[i].bestAction:
                    bestActionCounts[banditInd][t] += 1
        bestActionCounts[banditInd] /= nBandits
        averageRewards[banditInd] /= nBandits
    return bestActionCounts, averageRewards


# for figure 2.2
def epsilonGreedy(nBandits, time):
    epsilons = [0, 0.1, 0.01]
    bandits = []
    for epsInd, eps in enumerate(epsilons):
        bandits.append([Bandit(epsilon=eps, sampleAverages=True) for _ in range(0, nBandits)])
    bestActionCounts, averageRewards = banditSimulation(nBandits, time, bandits)
    global figureIndex
    plt.figure(figureIndex)
    figureIndex += 1
    for eps, counts in zip(epsilons, bestActionCounts):
        plt.plot(counts, label='epsilon = '+str(eps))
    plt.xlabel('Steps')
    plt.ylabel('% optimal action')
    plt.legend()
    plt.figure(figureIndex)
    figureIndex += 1
    for eps, rewards in zip(epsilons, averageRewards):
        plt.plot(rewards, label='epsilon = '+str(eps))
    plt.xlabel('Steps')
    plt.ylabel('average reward')
    plt.legend()


# for figure 2.3
def optimisticInitialValues(nBandits, time):
    bandits = [[], []]
    bandits[0] = [Bandit(epsilon=0, initial=5, stepSize=0.1) for _ in range(0, nBandits)]
    bandits[1] = [Bandit(epsilon=0.1, initial=0, stepSize=0.1) for _ in range(0, nBandits)]
    bestActionCounts, _ = banditSimulation(nBandits, time, bandits)
    global figureIndex
    plt.figure(figureIndex)
    figureIndex += 1
    plt.plot(bestActionCounts[0], label='epsilon = 0, q = 5')
    plt.plot(bestActionCounts[1], label='epsilon = 0.1, q = 0')
    plt.xlabel('Steps')
    plt.ylabel('% optimal action')
    plt.legend()


# for figure 2.4
def ucb(nBandits, time):
    bandits = [[], []]
    bandits[0] = [Bandit(epsilon=0, stepSize=0.1, UCBParam=2) for _ in range(0, nBandits)]
    bandits[1] = [Bandit(epsilon=0.1, stepSize=0.1) for _ in range(0, nBandits)]
    _, averageRewards = banditSimulation(nBandits, time, bandits)
    global figureIndex
    plt.figure(figureIndex)
    figureIndex += 1
    plt.plot(averageRewards[0], label='UCB c = 2')
    plt.plot(averageRewards[1], label='epsilon greedy epsilon = 0.1')
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend()


# for figure 2.5
def gradientBandit(nBandits, time):
    bandits =[[], [], [], []]
    bandits[0] = [Bandit(gradient=True, stepSize=0.1, gradientBaseline=True, trueReward=4) for _ in range(0, nBandits)]
    bandits[1] = [Bandit(gradient=True, stepSize=0.1, gradientBaseline=False, trueReward=4) for _ in range(0, nBandits)]
    bandits[2] = [Bandit(gradient=True, stepSize=0.4, gradientBaseline=True, trueReward=4) for _ in range(0, nBandits)]
    bandits[3] = [Bandit(gradient=True, stepSize=0.4, gradientBaseline=False, trueReward=4) for _ in range(0, nBandits)]
    bestActionCounts, _ = banditSimulation(nBandits, time, bandits)
    labels = ['alpha = 0.1, with baseline',
              'alpha = 0.1, without baseline',
              'alpha = 0.4, with baseline',
              'alpha = 0.4, without baseline']
    global figureIndex
    plt.figure(figureIndex)
    figureIndex += 1
    for i in range(0, len(bandits)):
        plt.plot(bestActionCounts[i], label=labels[i])
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()

# Figure 2.6
def figure2_6(nBandits, time):
    labels = ['epsilon-greedy', 'gradient bandit',
              'UCB', 'optimistic initialization']
    generators = [lambda epsilon: Bandit(epsilon=epsilon, sampleAverages=True),
                  lambda alpha: Bandit(gradient=True, stepSize=alpha, gradientBaseline=True),
                  lambda coef: Bandit(epsilon=0, stepSize=0.1, UCBParam=coef),
                  lambda initial: Bandit(epsilon=0, initial=initial, stepSize=0.1)]
    parameters = [np.arange(-7, -1, dtype=np.float),
                  np.arange(-5, 2, dtype=np.float),
                  np.arange(-4, 3, dtype=np.float),
                  np.arange(-2, 3, dtype=np.float)]

    bandits = [[generator(pow(2, param)) for _ in range(0, nBandits)] for generator, parameter in zip(generators, parameters) for param in parameter]
    _, averageRewards = banditSimulation(nBandits, time, bandits)
    rewards = np.sum(averageRewards, axis=1)/time

    global figureIndex
    plt.figure(figureIndex)
    figureIndex += 1
    i = 0
    for label, parameter in zip(labels, parameters):
        l = len(parameter)
        plt.plot(parameter, rewards[i:i+l], label=label)
        i += l
    plt.xlabel('Parameter(2^x)')
    plt.ylabel('Average reward')
    plt.legend()


figure2_1()
epsilonGreedy(2000, 1000)
optimisticInitialValues(2000, 1000)
ucb(2000, 1000)
gradientBandit(2000, 1000)

# This will take somehow a long time
figure2_6(2000, 1000)

plt.show()


"""
UCB uncertainity analysys
[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
[ 1.19459581  1.66510922  1.66510922  1.66510922  1.66510922  1.66510922 2.53727248  2.53727248  2.53727248  2.53727248]
[ 1.91020426  1.82212986  1.78556602  1.94850287  1.66196544  2.6771324 2.6771324   2.6771324   2.6771324   2.6771324 ]
[ 1.98995549  1.90188109  1.86531725  2.0282541   1.74171667  2.03733577 2.78991767  2.78991767  2.78991767  2.78991767]
[ 2.05651977  1.96844537  1.93188153  2.09481838  1.80828095  2.10390004 1.83791506  2.88405377  2.88405377  2.88405377]
[ 2.11347993  2.02540553  1.98884169  2.15177855  1.86524111  2.16086021 1.89487523  2.07126834  2.96460761  2.96460761]
[ 2.16315181  2.07507741  2.03851357  2.20145043  1.91491299  2.21053209 1.94454711  2.12094022  2.18568365  3.03485426]
[ 2.20711513  2.11904073  2.08247689  2.24541375  1.95887631  2.25449541 1.98851043  2.16490354  2.22964697  2.12973361]
[ 2.24649359  2.15841919  2.12185535  2.28479221  1.99825477  1.90071054 2.02788889  2.204282    2.26902543  2.16911207]
[ 2.28211376  2.19403936  2.15747552  1.69783388  2.03387494  1.92979429 2.06350905  2.23990216  2.30464559  2.20473223]
[ 2.31460056  2.22652616  2.18996232  1.72435924  2.06636174  1.95631965 2.09599586  2.27238897  1.92841473  2.23721904]
[ 1.86349683  2.25636307  2.21979923  1.74872098  2.09619865  1.98068138 2.12583277  2.30222587  1.95277646  2.26705595]
[ 1.8860063   2.28393143  2.24736759  1.77123045  2.12376701  2.00319086 2.15340113  1.75002179  1.97528594  2.29462431]
[ 1.90691326  2.30953712  2.27297328  1.79213741  2.1493727   2.02409782 2.17900682  1.77092875  1.9961929   1.65264105]
[ 1.92642095  1.81876141  2.29686523  1.8116451   2.17326465  2.04360551 2.20289877  1.79043644  2.01570059  1.67214875]
[ 1.94469689  1.83703735  1.71113719  1.82992104  2.19564801  2.06188145 2.22528213  1.80871238  2.03397653  1.69042468]
[ 1.9618807   1.85422116  1.728321    1.84710485  2.2166938   2.07906526 1.45898155  1.82589619  2.05116034  1.7076085 ]
[ 1.97808994  1.8704304   1.74453024  1.86331409  1.59447253  2.0952745 1.47519079  1.84210543  2.06736958  1.72381773]
[ 1.99342445  1.88576491  1.75986476  1.8786486   1.60980705  1.71425592 1.4905253   1.85743994  2.08270409  1.73915225]
[ 2.00796976  1.90031022  1.77441006  1.89319391  1.62435235  1.72685252 1.50507061  1.87198525  1.916697    1.75369755]
[ 1.70932807  1.91414008  1.78823992  1.90702377  1.63818221  1.73882953 1.51890047  1.88581511  1.92867401  1.76752741]
[ 1.72074096  1.92731855  1.80141839  1.92020224  1.65136068  1.75024242 1.53207894  1.89899358  1.96039699  1.78070589]
[ 1.7316382   1.9399016   1.81400144  1.93278529  1.66394373  1.76113966 1.54466198  1.91157663  2.17485991  1.79328893]
[ 1.74206237  1.95193839  1.82603823  1.94482208  1.67598052  1.77156383 1.55669878  1.92361342  1.92851884  1.80532573]
[ 1.75205107  1.54219556  1.8375722   1.95635605  1.68751449  1.78155253 1.56823274  1.93514738  1.9360696   1.81685969]
"""
