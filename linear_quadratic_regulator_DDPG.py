import tensorflow as tf
import keras
from keras import layers, losses
import numpy as np
import matplotlib.pyplot as plt

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

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

class ActorCritic():

    def __init__(self, state_dim, action_dim):

        self.batch_size = 32
        self.max_memory_size = 1000

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = 0.99
        self.tau = 0.005
        self.lower_action_bound = -1
        self.upper_action_bound = 1

        self.buffer = experience_memory(self.max_memory_size, self.batch_size, self.state_dim, self.action_dim)

        self.mean = np.zeros(1)
        self.std_dev = 0.5*np.ones(1)

        self.noice_Obj = OUActionNoise(self.mean, self.std_dev)
        # init the neural nets
        self.critic = self.get_critic_NN()
        self.target_critic = self.get_critic_NN()
        self.critic_optimizer = tf.keras.optimizers.Adam(0.002)

        self.actor = self.get_actor_NN()
        self.target_actor = self.get_actor_NN()
        self.actor_optimizer = tf.keras.optimizers.Adam(0.001)

    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch)
           
            y = reward_batch + self.gamma*self.target_critic([next_state_batch, target_actions])
            critic_value = self.critic([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.math.square(y- critic_value))
            
        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            actions = self.actor(state_batch)
            critic_value = self.critic([state_batch, actions])

            actor_loss = -tf.math.reduce_mean(critic_value)
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )
        
    def learn(self):
        # get sample

        record_range = min(self.buffer.buffer_counter, self.buffer.buffer_capacity)

        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = tf.convert_to_tensor(self.buffer.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.buffer.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.buffer.reward_buffer[batch_indices])
        next_state_batch = tf.convert_to_tensor(self.buffer.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)

    @tf.function
    def update_target(self, target_weights, weights):
        for (a,b) in zip(target_weights, weights):
            a.assign(self.tau *b + (1-self.tau) *a)

        


    def get_critic_NN(self):
        # input [state, action]

        state_input = layers.Input(shape =(self.state_dim,))

        state_out = layers.BatchNormalization()(state_input)
        state_out = layers.Dense(16, activation='relu')(state_out)
        state_out = layers.Dense(32, activation ='relu')(state_out)

        action_input = layers.Input(shape = (self.action_dim,))
        action_out = layers.BatchNormalization()(action_input)
        action_out = layers.Dense(32, activation = 'relu')(action_out)

        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(256, activation = 'relu')(concat)
        out = layers.Dense(256, activation = 'relu')(out)
        out = layers.Dense(1)(out)

        model = keras.Model(inputs = [state_input, action_input], outputs = out)

        return model

    def get_actor_NN(self):
        
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(self.state_dim,))
        out = layers.BatchNormalization()(inputs)
        out = layers.Dense(256, activation="relu")(out)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

        # Our upper bound is 2.0 .
        outputs = outputs * self.upper_action_bound
        model = tf.keras.Model(inputs, outputs)
        return model

    def policy(self, state):
        sampled_actions = tf.squeeze(self.actor(state))
        
        noice = self.noice_Obj()
      
        sampled_actions = sampled_actions + noice

        legal_action = np.clip(sampled_actions, self.lower_action_bound, self.upper_action_bound)

        return [np.squeeze(legal_action)]
        
class CaseOne():

    def __init__(self):
        
        self.num_episodes = 200
        self.state_dim = 1
        self.action_dim = 1
        self.AC = ActorCritic(self.state_dim, self.action_dim)

        self.T = 5
        self.N = 50
        self.dt = self.T/self.N

        self.r = 2

        self.sig = 0.1

    def f(self, n,x,a):
        return np.linalg.norm(a)**2
        

    def g(self, n,x):
        return np.zeros(self.AC.action_dim)
    
    def has_exit(self,x):
        if (np.linalg.norm(x) < self.r):
            return False
        else:
            return True

    def run_episodes(self):
        ep_reward_list = []
        # To store average reward history of last few episodes
        avg_reward_list = []
        for ep in range(self.num_episodes):
            X = np.zeros((self.N,1), dtype= np.float32)
            n=0
            episodic_reward = 0
            while(True):

                state = tf.expand_dims(tf.convert_to_tensor(X[n]),0)

                action = self.AC.policy(state)

                reward = self.f(n,state, action)

                X[n+1] = X[n] + (X[n] + action)* self.dt + self.sig * np.sqrt(self.dt) * np.random.normal()

                new_state = tf.expand_dims(tf.convert_to_tensor(X[n+1]),0)

                done = self.has_exit(new_state)
                if (done):
                    reward = reward + self.g(n,new_state)

                episodic_reward += reward
                self.AC.buffer.record((state,action,reward, new_state))

                self.AC.learn()

                self.AC.update_target(self.AC.target_critic.variables, self.AC.critic.variables)
                self.AC.update_target(self.AC.target_actor.variables, self.AC.actor.variables)

                if (done):
                    break
                
                if (n == self.N-2):
                    break
                n += 1

            ep_reward_list.append(episodic_reward)
            # Mean of last 40 episodes
            avg_reward = np.mean(ep_reward_list[-40:])
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
           
           action = self.AC.actor(tf.expand_dims(tf.convert_to_tensor(state_space[ix]),0))
          
          
           P[ix] = action

           v = self.AC.critic([tf.expand_dims(tf.convert_to_tensor(state_space[ix]),0),action])
           
           V[ix] = v
        
        ax[0].plot(state_space, V)
        ax[0].set_title('value function')

        ax[1].plot(state_space, P)
        ax[1].set_title('policy function')
        ax[1].set_ylim([self.AC.lower_action_bound -0.5, self.AC.upper_action_bound+0.5])

        X = np.zeros((self.N,1))

        for i in range(self.N-1):
            X[i+1] = X[i] + (X[i] + self.AC.actor(tf.expand_dims(tf.convert_to_tensor(X[i]),0))[0][0])*self.dt + self.sig * np.sqrt(self.dt) * np.random.normal()
        ax[2].plot(np.linspace(0, self.T, self.N), X)
        ax[2].set_ylim([-5,5])
        ax[2].fill_between(np.linspace(0, self.T, self.N),-self.r,self.r , color = 'green', alpha = 0.3)
        plt.show()

if __name__ == "__main__":

    LQR = CaseOne()

    LQR.run_episodes()
    LQR.plots()