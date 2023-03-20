import numpy as np

from random import randint
import gym


class Node():

    def __init__(self, state, parent=None, statistics = {}):

        self.state = state
        self.parent = parent
        self.children = {}
        self.statistics = statistics
    
    def expand(self, action, next_state):
        child = Node(next_state, parent=self)
        self.children[action] = child

        return child
    
class MCTS():


    def __init__(self, state, game,num_players, C=1):

        self.game = gym.make(game)
        self.actions = self.game.action_space
        
        self.num_players = num_players
        self.C = C
        self.root = Node(state, statistics={"visits":0, "reward": np.zeros(self.num_players)})

    def is_fully_expanded(self, node:Node):
        return len(self.game.get_actions(node.state)) == len(list(node.children))
    
    def best_action(self, node:Node):

        children = list(node.children.values())
        visits = np.array([child.statistics["visits"] for child in children])
        rewards = np.array([child.statistics["reward"] for child in children])

        total_rollouts = node.statistics["visits"]

        pid = self.game.get_current_player_id(node.state)

        ucb =  ( rewards [: , pid ]/ visits + self.C *np. sqrt (2* np. log ( total_rollouts )/ visits ))
        best_ucb_id = np.random.choice(np.flatnonzero(ucb == ucb.max()))
        return list(node.children.keys())[best_ucb_id]
    
    def tree_policy(self, node:Node):

        while not self.game.is_terminal(node.state):
            if not self.is_fully_expanded(node):
                act_set = np.setdiff1d(self.game.get_actions(node.state), list(node.children.keys()))
                action = act_set[randint(0, len(act_set)-1)]
                new_state = self.game.perform_action(action, node.state)

                childnode = node.expand(action, new_state)
                childnode.statistics = {"visits":0, "reward": np.zeros(self.game.num_players())}

                return childnode
            else:
                node = node.children[self.best_action(node)]

        return node
    
    def rollout(self, node:Node):
        roll_state = node.state.copy()

        while not self.game.is_terminal(roll_state):
            act_set = self.game.get_actions(roll_state)
            action = act_set[randint(0, len(act_set)-1)]
            roll_state = self.game.perform_action(action, roll_state)

        return self.game.reward(roll_state)

    def backup(self,node:Node, reward):
        while not node is None:
            node.statistics["visits"] +=1
            node.statistics["reward"] += reward
            node = node.parent

    def run_iter(self, iterations):
        for i in range(iterations):
            selected_node = self.tree_policy(self.root)
            reward = self.rollout(selected_node)
            self.backup(selected_node, reward)
        return self.best_action(self.root)



        