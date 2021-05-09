import gym
import numpy as np
np.random.seed(2)
env = gym.make('FrozenLake-v0')
lr = 0.8
y = 0.95
num_episodes = 20000
rlist = []

Q = np.zeros([env.observation_space.n,env.action_space.n])
for i in range(num_episodes):
    s = env.reset()
    rall = 0
    d = False
    j = 0
    while j < 100:
        j += 1
        a = np.argmax(Q[s,:]+np.random.randn(1,env.action_space.n)*(1.0/(i+1)))
        s1,r,d,_ = env.step(a)
        Q[s,a] += lr*(r + y*np.max(Q[s1,:]) - Q[s,a])
        rall += r
        s = s1
        if d == True:
            break
    rlist.append(rall)

print('Score over time:' + str(sum(rlist)/num_episodes))
print('Final Q-Table value')
print(Q)
