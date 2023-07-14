import gym
from MCTS import MCTS

game = 'FrozenLake-v1'
env = gym.make(game)

obs,_ = env.reset()
print(env.action_space)


root = env.reset()
done = False
mcts_tree = MCTS(root, game, 1)

while not done:

    env.render()
    action = mcts_tree.run_iter(10)
    new_state, reward, observation,_ = env.step(action)