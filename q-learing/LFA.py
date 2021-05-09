import numpy as np
import gym

lr_r = 0.01
eps = 0.01



for i in range(1000):
    env = gym.make("MountainCar-v0")
    s = env.reset()

    for j in range(10000):
        env.render()
        if np.random.uniform(0,1) < eps:
            action = np.random.choice(env.action_space.n)
        else:
            action =

        s_ ,r ,done ,_ = env.step(action)

        if done == True:
            break

        else:

        s = s_

# a wrapper class for polynomial / Fourier -based value function
POLYNOMIAL_BASES = 0
class BasesValueFunction:
    # @order: # of bases, each function also has one more constant parameter (called bias in machine learning)
    # @type: polynomial bases or Fourier bases
    def __init__(self, order, type):
        self.order = order
        self.weights = np.zeros()

        # set up bases function
        self.bases = []
        for i in range(0, order + 1):
            self.bases.append(lambda s, i=i: pow(s, i))

    # get the value of @state
    def value(self, state):
        # map the state space into [0, 1]
        state /= float(N_STATES)
        # get the feature vector
        feature = np.asarray([func(state) for func in self.bases])
        return np.dot(self.weights, feature)

    def update(self, delta, state):
        # map the state space into [0, 1]
        state /= float(N_STATES)
        # get derivative value
        derivativeValue = np.asarray([func(state) for func in self.bases])
        self.weights += delta * derivativeValue

def gradientMonteCarlo(valueFunction, alpha, distribution=None):
    currentState = START_STATE
    trajectory = [currentState]

    # We assume gamma = 1, so return is just the same as the latest reward
    reward = 0.0
    while currentState not in END_STATES:
        action = getAction()
        newState, reward = takeAction(currentState, action)
        trajectory.append(newState)
        currentState = newState

    # Gradient update for each state in this trajectory
    for state in trajectory[:-1]:
        delta = alpha * (reward - valueFunction.value(state))
        valueFunction.update(delta, state)
        if distribution is not None:
            distribution[state] += 1

"""
import numpy as np
bases = []
for i in range(0, 3):
    bases.append(lambda s, i=i: pow(s, i))
state = float(input("Enter a number"))
a = np.asarray([func(state) for func in bases]) # np.array([i(state) for i in bases])
print(a)
"""
