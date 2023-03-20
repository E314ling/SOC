import gym
from MCTS import MCTS

game = 'FrozenLake-v1'
env = gym.make(game)


root = env.reset()
done = False
mcts_tree = MCTS(state = root, game = game)
while not done:

    env.render()
    action = mcts_tree.run_iter(10)
    new_state, reward, observation,_ = env.step(action)