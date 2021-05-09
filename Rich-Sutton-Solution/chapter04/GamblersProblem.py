"""
Gambler’s Problem A gambler has the opportunity to make bets on the outcomes of a sequence of coin flips.
If the coin comes up heads, he wins as many dollars as he has staked on that flip; if it is tails,
he loses his stake. The game ends when the gambler wins by reaching his goal of $100, or loses by running
out of money. On each flip, the gambler must decide what portion of his capital to stake, in integer numbers
of dollars. This problem can be formulated as an undiscounted, episodic, finite MDP. The state is the
gambler’s capital, s ∈ {1, 2, . . . , 99} and the actions are stakes, a ∈ {0, 1, . . . , min(s, 100−s)}.
The reward is zero on all transitions except those on which the gambler reaches his goal, when it is +1.
The state-value function then gives the probability of winning from each state. A policy is a mapping
from levels of capital to stakes. The optimal policy maximizes the probability of reaching the goal.
Let p(h) denote the probability of the coin coming up heads. If p(h) is known, then the entire problem is
known and it can be solved, for instance, by value iteration. Figure 4.3 shows the change in the value function
over successive sweeps of value iteration, and the final policy found, for the case of p(h) = 0.4. This
policy is optimal, but not unique. In fact, there is a whole family of optimal policies, all corresponding to
ties for the argmax action selection with respect to the optimal value function. Can you guess what the entire family looks like?
"""
import numpy as np
import matplotlib.pyplot as plt
# goal
GOAL = 100

# all states, including state 0 and state 100
states = np.arange(GOAL + 1)
# state value
stateValue = np.zeros(GOAL + 1)
stateValue[GOAL] = 1.0 #reward
# optimal policy is mapping from state(gambler’s capital) to best actions(stakes) i.e. in particular state what action(how much invest) to
# take to get as much reward as possible.
policy = np.zeros(GOAL + 1)

# probability of head
headProb = 0.4
"""
Initialize array V arbitrarily (e.g., V(s) = 0 for all s ∈ S + )
Repeat
 ∆ ← 0
 For each s ∈ S:
   actionValue = []
   For each action a ∈ A(s):
    V (s) ← max(a)sigma(p(s',r|s, a)[r + γV(s')])
    ∆ ← max(∆, |v − V (s)|)
    v ← V (s)
 until ∆ < θ (a small positive number)
"""
# value iteration
while True:
    delta = 0.0
    #a = 0
    for state in states[1:GOAL]:
        # get possilbe actions for current state
        actions = np.arange(min(state, GOAL - state) + 1) # The last element is not included thats why we add 1
        actionReturns = []
        for action in actions:
            actionReturns.append(headProb * stateValue[state + action] + (1 - headProb) * stateValue[state - action])
            # in state take action and go into next state i.e. capital increases means win and the probability of winning is 0.4
            # if the next state in which capital decreases i.e. we loose and the probability of loosing is (1-0.4)
        newValue = np.max(actionReturns)
        delta = np.maximum(delta,np.abs(stateValue[state] - newValue))
        #print(delta)
        # update state value
        stateValue[state] = newValue
        #a += 1
    #print(stateValue)
    if delta < 1e-9:
        #print(a) 99 times while loop run
        break
#Now we have updated our stateValue so we can easily find better policy
# calculate the optimal policy
for state in states[1:GOAL]:
    actions = np.arange(min(state, GOAL - state) + 1) # The last element is not included thats why we add 1
    actionReturns = []
    for action in actions:
        actionReturns.append(headProb * stateValue[state + action] + (1 - headProb) * stateValue[state - action])
    # due to tie, can't reproduce the optimal policy in book
    policy[state] = actions[np.argmax(actionReturns)]

# figure 4.3
plt.figure(1)
plt.xlabel('Capital')
plt.ylabel('Value estimates')
plt.plot(stateValue)
plt.figure(2)
plt.scatter(states, policy)
plt.xlabel('Capital')
plt.ylabel('Final policy (stake)')
plt.show()

"""
Why does the optimal policy for the gambler’s problem have such a
curious form? In particular, for capital of 50 it bets it all on one flip, but for capital
of 51 it does not. Why is this a good policy?
"""
