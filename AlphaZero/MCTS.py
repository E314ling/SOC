import numpy as np

from random import randint
import gym

class SDE():

    def __init__(self) -> None:

        self.terminal_time = 1
        self.N = 20
        self.dt = self.terminal_time / self.N

        self.action_bound = 5
        self.actions_n = 10

        self.state_bound = 5
        self.state_n = 10

        self.actions = np.linspace(-self.actions,self.actions,self.actions_n)
        self.states = np.linspace(-self.state_bound, self.state_bound, self.state_n)
    
    def reward(self, X, a,n):
        if n == self.N-1:
            return X**2
        else:
            return X**2 + a**2
    
    def check_if_done(self,X,n):
        if n == self.N-1:
            return True
        else:
            return False
        
    def step(self, X, action,n):

        X_new = X + self.dt*action + np.sqrt(self.dt)*np.random.rand(size = 1)

        done = self.check_if_done(X_new,n)
        reward = self.reward(X,action)

        return X_new, reward, done
    def start_state(self):
        return 2*self.state_bound*np.random.rand(1) - self.state_bound
    
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
        self.num_actions = self.game.action_space.n
        self.actions = np.linspace(0,self.num_actions,self.num_actions, dtype= np.int32)
        self.C = C
        state = self.game.reset()
        self.root = Node(state, statistics={"visits":0, "reward": 0, "is_terminal":False})

    def is_fully_expanded(self, node:Node):
        return len(self.actions) == len(list(node.children))
    
    def best_action(self, node:Node):

        children = list(node.children.values())
        visits = np.array([child.statistics["visits"] for child in children])
        rewards = np.array([child.statistics["reward"] for child in children])

        total_rollouts = node.statistics["visits"]

        ucb =  ( rewards[:]/ visits + self.C *np. sqrt (2* np. log ( total_rollouts )/ visits ))
        best_ucb_id = np.random.choice(np.flatnonzero(ucb == ucb.max()))
        return list(node.children.keys())[best_ucb_id]
    
    def tree_policy(self, node:Node):

        while not node.statistics['is_terminal']:
            if not self.is_fully_expanded(node):
                act_set = np.setdiff1d(self.actions, list(node.children.keys()))
                action = act_set[randint(0, len(act_set)-1)]
                new_state, reward, done,_ ,__= self.game.step(action)

                childnode = node.expand(action, new_state)
                childnode.statistics = {"visits":0, "reward": reward, "is_terminal":done}

                return childnode
            else:
                node = node.children[self.best_action(node)]

        return node
    
    def rollout(self, node:Node):
        roll_state:Node
        roll_state = node.state

        while not roll_state.statistics['is_terminal']:
            act_set = self.actions
            action = act_set[randint(0, len(act_set)-1)]
            roll_state,reward, done,_,__ = self.game.step(action)

        return reward

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



        