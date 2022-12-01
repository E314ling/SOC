import tensorflow as tf
import keras
from keras import layers, losses
import numpy as np
import matplotlib.pyplot as plt
import scipy.special
import itertools
class MultivariantPolynomial():
    '''
    inputs:
    (i) the degree of the multivariant polynomial
    (ii) dimension of the underlining space
    '''
    def __init__(self, deg, dim):
        self.deg = deg
        self.dim = dim
        self.num_weights = int(scipy.special.binom(self.dim+self.deg, self.deg))

        self.weigths = np.zeros(self.num_weights, dtype = np.float32)
        
        for i in range(self.num_weights):
            b = 0.003
            self.weigths[i] = 2*b*np.random.rand() - b
        self.exps = self.get_exponentials()

    def get_exponentials(self):
        
        rangeStr = ''

        for i in range(self.deg+1):
            rangeStr += str(i)

        combinations = []
        print(rangeStr,self.dim)
        for x in itertools.product(rangeStr, repeat = self.dim):
            s = 0
            for e in  x:
                s += int(e)
            if (s <= self.deg):
                combinations.append(np.array(x)) 
        print(combinations)
        return combinations

    def evaluate(self, x):
        '''
        creates the vector [1,x1,x2, x1x2, ...]
        for x with dimension n
        '''

        vals = np.ones(self.num_weights, dtype=np.float32)

        counter = 0
        for expo in self.exps:
            for n in range(self.dim):
                e = int(expo[n])
                vals[counter] *= x[n]**e
            counter += 1
        
        return vals

    
    def predict(self, vals):
        
        return np.dot(vals, self.weigths)
        
        
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
       
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
       
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)

class experience_memory():

    def __init__(self, buffer_capacity, batch_size, state_dim, action_dim):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, state_dim), dtype=np.float32)
        self.action_buffer = np.zeros((self.buffer_capacity, action_dim), dtype=np.float32)
        self.reward_buffer = np.zeros((self.buffer_capacity, 1), dtype=np.float32)
        self.next_state_buffer = np.zeros((self.buffer_capacity, state_dim), dtype=np.float32)
        self.done_buffer = np.zeros((self.buffer_capacity, 1), dtype=np.float32)

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.done_buffer[index] = obs_tuple[4]


        self.buffer_counter += 1

class ActorCritic():

    def __init__(self, state_dim, action_dim):

        

        self.batch_size = 64
        self.max_memory_size = 10000

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = 0.99
        self.tau = 0.05
        self.lower_action_bound = -1
        self.upper_action_bound = 1

        self.action_space = np.linspace(self.lower_action_bound,self.upper_action_bound,11)
        self.num_a = len(self.action_space)

        self.critic = MultivariantPolynomial(3, self.action_dim + self.state_dim)
        self.target_critic = MultivariantPolynomial(3, self.action_dim + self.state_dim)
        
        self.buffer = experience_memory(self.max_memory_size, self.batch_size, self.state_dim, self.action_dim)
        self.mean = np.zeros(1)
        self.std_dev = 0.2*np.ones(1)

        self.noice_Obj = OUActionNoise(self.mean, self.std_dev)
        # init the neural nets

        self.alpha = 0.001
        self.eps = 1

    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
      
        with tf.GradientTape() as tape:
            target_actions = self.target_actor.evaluate(next_state_batch)
            
            y = reward_batch + (tf.ones_like(done_batch)-done_batch)* self.gamma*self.target_critic([next_state_batch, target_actions])
            
            critic_value = self.critic([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.math.square(y- critic_value))
        
        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            actions = self.actor(state_batch)
            critic_value = self.critic([state_batch, actions])

            actor_loss = tf.math.reduce_mean(critic_value)
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )

    
    def update_without_replay(self, state, action,reward, next_state, done):
        
        
        vals = np.zeros(self.num_a)
        for i in range(self.num_a):
            a = self.action_space[i]
            critic_grad = self.critic.evaluate([state,a])
            vals[i] = self.critic.predict(critic_grad)

        a_ind = np.argmin(vals)
        action = self.action_space[a_ind]

        target_critic_grad = self.target_critic.evaluate([next_state, action])
        target_q = self.target_critic.predict(target_critic_grad)

        y = tf.convert_to_tensor(reward) + tf.convert_to_tensor((1-done)* self.gamma)*target_q
        
        critic_grad = self.critic.evaluate([state, tf.reshape(tf.convert_to_tensor(action),[1,1])])
        critic_value = self.critic.predict(critic_grad)

        critic_TD = y - critic_value
        
        self.critic.weigths = self.critic.weigths + self.alpha * critic_TD * critic_grad
        
    def learn_without_replay(self, state, action,reward, next_state, done):
        if done:
            done_num = 1
        else:
            done_num = 0
        self.update_without_replay(state, action,reward, next_state, done_num)
    
        
    def learn(self):
        # get sample

        record_range = min(self.buffer.buffer_counter, self.buffer.buffer_capacity)

        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = tf.convert_to_tensor(self.buffer.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.buffer.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.buffer.reward_buffer[batch_indices])
        next_state_batch = tf.convert_to_tensor(self.buffer.next_state_buffer[batch_indices])

        done_batch = tf.convert_to_tensor(self.buffer.done_buffer[batch_indices])
        
        self.update(state_batch, action_batch, reward_batch, next_state_batch, done_batch)


    @tf.function
    def update_target(self, target_weights, weights):
        for (a,b) in zip(target_weights, weights):
            a.assign(self.tau *b + (1-self.tau) *a)


    def epsilon_greedy(self, state):
        
        if( self.eps < np.random.rand()):
            rand_ind  = np.random.choice(self.num_a)
            rand_a = self.action_space[rand_ind]
            
            return  rand_a
        else:
            vals = np.zeros(self.num_a)
            for i in range(self.num_a):
                a = self.action_space[i]
                critic_grad = self.critic.evaluate([state,a])
                vals[i] = self.critic.predict(critic_grad)

            a_ind = np.argmin(vals)
           
            return self.action_space[a_ind]
        
class CaseOne():

    def __init__(self):
        # dX_t = (A X_t + B u_t) dt + sig * dB_t
        self.A = 1
        self.B = 1
        self.sig = 1

        # f(x,u) = f_A ||x||^2 + f_B ||u||^2
        self.f_A = 1
        self.f_B = 1

        # g(x) = D * ||x||^2
        self.D = 0

        self.num_episodes = 500
        self.state_dim = 1
        self.action_dim = 1
        self.AC = ActorCritic(self.state_dim, self.action_dim)

        self.T = 1
        self.N = 20
        self.dt = self.T/self.N

        self.r = 0

    def f(self, n,x,a):
        return np.float32(self.B *np.linalg.norm(a)**2 + self.A*np.linalg.norm(x)**2)
        

    def g(self, n,x):
        return self.D * np.linalg.norm(x)**2
    
    def has_exit(self,x):
        if self.r == 0:
            return False
        else:
            if (np.linalg.norm(x) < self.r):
                return False
            else:
                return True

    def check_if_done(self,n,x):
        if n == self.N-2:
            return True
        else:
            if self.r == 0:
                return False
            else:
                if (np.linalg.norm(x) < self.r):
                    return False
                else:
                    return True
    
    def run_episodes(self):
        ep_reward_list = []
        # To store average reward history of last few episodes
        avg_reward_list = []
        for ep in range(self.num_episodes):
            X = np.zeros(self.N, dtype= np.float32)
            n=0
            episodic_reward = 0
            while(True):
                a = 10

                self.AC.alpha = 0.0001
                self.AC.eps = 0.7 * (a / (a + ep))
                state = tf.expand_dims(tf.convert_to_tensor(X[n]),0)

                action = self.AC.epsilon_greedy(state)

                reward = self.f(n,state, action)

                X[n+1] = X[n] + (X[n] + action)* self.dt + self.sig * np.sqrt(self.dt) * np.random.normal()

                new_state = tf.expand_dims(tf.convert_to_tensor(X[n+1]),0)

                done = self.check_if_done(n,new_state)
                
                episodic_reward += reward

                self.AC.buffer.record((state,action,reward, new_state, done))

                self.AC.learn_without_replay(state, action, reward, new_state, done)
                #self.AC.learn()

                
                #self.AC.update_target(self.AC.target_critic.weigths, self.AC.critic.weigths)
                self.AC.target_critic.weigths = self.AC.tau* self.AC.critic.weigths + (1-self.AC.tau)* self.AC.target_critic.weigths

                if (done):
                    break
                
                if (n == self.N-2):
                    break
                n += 1

            ep_reward_list.append(episodic_reward)
            # Mean of last 40 episodes
            avg_reward = np.mean(ep_reward_list[-50:])
            print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
            avg_reward_list.append(avg_reward)
        # Plotting graph
        # Episodes versus Avg. Rewards
        plt.plot(avg_reward_list)
        plt.xlabel("Episode")
        plt.ylabel("Avg. Epsiodic Reward")
        plt.show()

    def plots(self):
        n_x = 20
        n_a = 20

        state_space = np.linspace(-2,2, n_x)
        
        fig, ax = plt.subplots(3)
 
        V = np.zeros(n_x)

        P = np.zeros(n_x)

        for ix in range(n_x):
            
            vals = np.zeros(self.AC.num_a)
            for i in range(self.AC.num_a):
                a = self.AC.action_space[i]
                q_grad = self.AC.critic.evaluate([state_space[ix],a])
                vals[i] = self.AC.critic.predict(q_grad)
            
            action = self.AC.action_space[np.argmin(vals)]
            print(action)
            P[ix] = action

            print(vals)
            v = np.min(vals)
            print(v)
            V[ix] = v
        
        ax[0].plot(state_space, V)
        ax[0].set_title('value function')

        ax[1].plot(state_space, P)
        ax[1].set_title('policy function')
        ax[1].set_ylim([self.AC.lower_action_bound -0.5, self.AC.upper_action_bound+0.5])

        X = np.zeros((self.N,1))

        for i in range(self.N-1):
            vals = np.zeros(self.AC.num_a)
            for j in range(self.AC.num_a):
                a = self.AC.action_space[j]
                q_grad = self.AC.critic.evaluate([X[i],a])
                vals[j] = self.AC.critic.predict(q_grad)
            
            
            action = self.AC.action_space[np.argmin(vals)]
            print(action)
            print(X[i])
            X[i+1] = X[i] + (X[i] + action)*self.dt + self.sig * np.sqrt(self.dt) * np.random.normal()

        ax[2].plot(np.linspace(0, self.T, self.N), X)
        ax[2].set_ylim([-5,5])
        ax[2].fill_between(np.linspace(0, self.T, self.N),-self.r,self.r , color = 'green', alpha = 0.3)
        plt.show()

if __name__ == "__main__":

    LQR = CaseOne()

    LQR.run_episodes()
    LQR.plots()
   