import tensorflow as tf
import keras
from keras import layers, losses
import numpy as np
import matplotlib.pyplot as plt
import linear_quadratic_regulator_DP as LQR
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.1, dt=1e-2, x_initial=None):
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
        self.state_buffer = np.zeros((self.buffer_capacity, state_dim+1), dtype=np.float32)
        self.action_buffer = np.zeros((self.buffer_capacity, action_dim), dtype=np.float32)
        self.reward_buffer = np.zeros((self.buffer_capacity, 1), dtype=np.float32)
        self.next_state_buffer = np.zeros((self.buffer_capacity, state_dim+1), dtype=np.float32)
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

    def __init__(self, state_dim, action_dim, load_model):

        self.batch_size = 512
        self.max_memory_size = 1000000

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = 1
        self.tau = 0.001
        self.lower_action_bound = -4
        self.upper_action_bound = 4

        self.buffer = experience_memory(self.max_memory_size, self.batch_size, self.state_dim, self.action_dim)

        
        # init the neural nets
        if (load_model):
            self.critic = tf.keras.models.load_model('./Models/LQR_2D_DDPG_critic.h5')
            self.target_critic = tf.keras.models.load_model('./Models/LQR_2D_DDPG_critic.h5')

            

            self.actor = tf.keras.models.load_model('./Models/LQR_2D_DDPG_Actor.h5')
            self.target_actor = tf.keras.models.load_model('./Models/LQR_2D_DDPG_Actor.h5')
        else:
            self.critic = self.get_critic_NN()
            self.target_critic = self.get_critic_NN()
            #self.target_critic_1.set_weights(self.critic_1.get_weights())

            self.actor = self.get_actor_NN()
            self.target_actor = self.get_actor_NN()
            self.target_actor.set_weights(self.actor.get_weights())


        self.critic_lr = 0.0001
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        
       
        self.actor_lr = 0.0001
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)
      
        self.var = 1
        self.var_decay = 1
        self.lr_decay = 1

        self.update_frames = 2

       
    def save_model(self):
        self.critic.save('./Models/LQR_2D_DDPG_critic.h5')
        
        
        self.actor.save('./Models/LQR_2D_DDPG_actor.h5')
    def update_lr(self):
        self.critic_lr = self.critic_lr * self.lr_decay
        self.critic_optimizer= tf.keras.optimizers.Adam(self.critic_lr)
        
        self.actor_lr = self.actor_lr * self.lr_decay
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)

    def update_var(self):
        self.var = np.max([0.1,self.var * self.var_decay])
    
    @tf.function
    def update_critic(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        target_actions = self.target_policy(next_state_batch) 
        target_vals= self.target_critic([next_state_batch, target_actions])
        
        
        y = tf.stop_gradient(reward_batch + (1-done_batch)* self.gamma*target_vals)

        with tf.GradientTape() as tape:
            
            critic_value = self.critic([state_batch, action_batch])
            
            critic_loss = losses.MSE(y, critic_value)

       
        
        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables) 
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))


    @tf.function
    def update_actor(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        
        with tf.GradientTape() as tape:
            tape.watch(self.actor.trainable_variables)
            actions = self.actor(state_batch)
            critic_value = self.critic([state_batch, actions])
            actor_loss = tf.math.reduce_mean(critic_value)
            
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )

   
    def learn(self,frame_num):
        # get sample

        record_range = min(self.buffer.buffer_counter, self.buffer.buffer_capacity)

        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = tf.convert_to_tensor(self.buffer.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.buffer.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.buffer.reward_buffer[batch_indices])
        next_state_batch = tf.convert_to_tensor(self.buffer.next_state_buffer[batch_indices])

        done_batch = tf.convert_to_tensor(self.buffer.done_buffer[batch_indices])
        
        self.update_critic(state_batch, action_batch, reward_batch, next_state_batch, done_batch)

        if (frame_num % self.update_frames == 0 and frame_num != 0):
            self.update_actor(state_batch, action_batch, reward_batch, next_state_batch, done_batch)


    @tf.function
    def update_target(self, target_weights, weights):
        for (a,b) in zip(target_weights, weights):
            a.assign(self.tau *b + (1-self.tau) *a)

        
    def get_critic_NN(self):
        # input [state, action]
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        state_input = layers.Input(shape =(self.state_dim+1,))
        action_input = layers.Input(shape =(self.action_dim,))

        input = tf.concat([state_input, action_input],1)
       
        out_1 = layers.Dense(128, activation = 'relu')(input)
        out_1 = layers.BatchNormalization()(out_1)
        out_1 = layers.Dense(64, activation = 'relu')(out_1)
        out_1 = layers.Dense(1, kernel_initializer= last_init)(out_1)

        model = keras.Model(inputs = [state_input, action_input], outputs = out_1)

        return model

    def get_actor_NN(self):
        
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(self.state_dim+1,))
       
        out = layers.Dense(64, activation="tanh")(inputs)
        out = layers.BatchNormalization()(out)
        out = layers.Dense(32, activation="tanh")(out)
        outputs = layers.Dense(self.action_dim, activation='tanh', kernel_initializer=last_init)(out)

        # Our upper bound is 2.0 .
        outputs = outputs * self.upper_action_bound
        model = tf.keras.Model(inputs, outputs)
        return model

    def policy(self, state):
        sampled_actions = tf.squeeze(self.actor(state))
        
        #noice = self.noice_Obj()
      
        sampled_actions = sampled_actions + np.random.normal(loc= 0, scale=self.var)

        legal_action = np.clip(sampled_actions, self.lower_action_bound, self.upper_action_bound)

        return [np.squeeze(legal_action)]
        #return [np.squeeze(sampled_actions)]
    
    @tf.function
    def target_policy(self, state):
        sampled_actions = self.target_actor(state)
        
        #noice = self.noice_Obj()
        noice = tf.random.normal(shape = sampled_actions.get_shape(), mean = 0.0, stddev = 0.2, dtype = tf.float32)
        noice = tf.clip_by_value(noice, -0.5, 0.5)
        sampled_actions = sampled_actions + noice
        
        legal_action = tf.clip_by_value(sampled_actions, clip_value_min = self.lower_action_bound, clip_value_max =self.upper_action_bound)

        return legal_action
        
class CaseOne():

    def __init__(self):
        # dX_t = (A X_t + B u_t) dt + sig * dB_t
        self.A = np.identity(2)
        self.B = np.identity(2)
        self.sig = 0.1

        # f(x,u) = f_A ||x||^2 + f_B ||u||^2
        self.f_A = np.identity(2)
        self.f_B = np.identity(2)

        # g(x) = D * ||x||^2
        self.D = np.identity(2)

        self.num_episodes = 6100
        self.state_dim = 2
        self.action_dim = 2
        self.AC = ActorCritic(self.state_dim, self.action_dim, False)

        self.T = 1
        self.N = 20
        self.dt = self.T/self.N


        self.r = 0
        self.dashboard_num = 1000
        self.mean_abs_error_v1 = []
        self.mean_abs_error_v2 = []
        self.mean_abs_error_P_x_0 = []
        self.mean_abs_error_P_y_0 = []

    def f(self, n,x,a):
        
       
        y1= np.dot(np.transpose(x), self.f_A)
        y2 = np.dot(np.transpose(a), self.f_B)
      
        return np.float32(np.dot(y1,x) + np.dot(y2,a))
        
    def g(self, n,x):
        y = np.dot(np.transpose(x), self.D)
        return np.dot(y,x)
    
    def has_exit(self,x):
        if self.r == 0:
            return False
        else:
            if (np.linalg.norm(x) < self.r):
                return False
            else:
                return True

    def check_if_done(self,n,x):
        if n == self.N-1:
            return True
        else:
            if self.r == 0:
                return False
            else:
                if (np.linalg.norm(x) < self.r):
                    return False
                else:
                    return True
    
    def run_episodes(self, n_x,V_t,A_t, base):
        ep_reward_list = []
        # To store average reward history of last few episodes
        avg_reward_list = []
        X = np.zeros((self.N,2), dtype= np.float32)
        X[0] = 2.5*np.random.rand(self.state_dim) - 2.5

        frame_num = 0
        for ep in range(self.num_episodes):
            
            n=0
            episodic_reward = 0
            while(True):
                
                state = np.array([n,X[n][0],X[n][1]], np.float32)
               
                state = tf.expand_dims(tf.convert_to_tensor(state),0)
                done = self.check_if_done(n,state)

                if (done):
                    reward = self.g(n,X[n])
                    action = self.AC.policy(state)[0]
                    X = np.zeros((self.N,2), dtype= np.float32)
                    X[0] = 2.5*np.random.rand(self.state_dim) - 2.5
                    new_state = np.array([0,X[0][0],X[0][1]], np.float32)
                    new_state = tf.expand_dims(tf.convert_to_tensor(new_state),0)
                         
                else:

                    action = self.AC.policy(state)[0]

                    reward = self.f(n,X[n], action)
                    #X[n+1] =  X[n] + (X[n] + action)*self.dt + np.sqrt(self.sig*self.dt)  * np.random.normal()
                    
                    X[n+1] =  (X[n] + action) + self.sig*np.random.normal(2)

                    new_state = np.array([n+1,X[n+1][0],X[n+1][1]], np.float32)
                    new_state = tf.expand_dims(tf.convert_to_tensor(new_state),0)

                self.AC.buffer.record((state,action,reward, new_state, done))
                 
                episodic_reward += reward
                # warm up
                if (ep >= 100):
                    self.AC.learn(n)
                    self.AC.update_target(self.AC.target_critic.variables, self.AC.critic.variables)
                    
                    if (n % self.AC.update_frames == 0 and n != 0):
                        
                        self.AC.update_target(self.AC.target_actor.variables, self.AC.actor.variables)
                        self.AC.update_lr()
                        self.AC.update_var()
                frame_num += 1
                if(done):
                    break
                else:
                    n += 1

            if (ep % self.dashboard_num == 0 and ep >100):
                self.dashboard(n_x,V_t,A_t,avg_reward_list,base)
            
            if (ep >= 100):
                self.AC.var = 0.1
                ep_reward_list.append(episodic_reward)
                # Mean of last 40 episodes
                avg_reward = np.mean(ep_reward_list[-100:])
                print("Episode * {} * Avg Reward is ==> {}, var ==> {}, actor_lr ==> {}".format(ep, avg_reward, self.AC.var, self.AC.actor_lr))
                avg_reward_list.append(avg_reward)
        # Plotting graph
        # Episodes versus Avg. Rewards
        self.AC.save_model()
        self.dashboard(n_x,V_t,A_t,avg_reward_list,base)
        
    def dashboard(self,n_x,V_t,A_t,avg_reward_list,base):

        x_space = np.linspace(-2,2, n_x)
        y_space = np.linspace(-2,2, n_x)

        fig = plt.figure()
        
        V1 = np.zeros((n_x,n_x))
        P = np.zeros((n_x,n_x,2))
        
        t0 = 1

        ax = fig.add_subplot(2, 4, 1)
        ax.plot(avg_reward_list)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Avg. Epsiodic Reward')
        ax.hlines(base,xmin = 0, xmax = len(avg_reward_list), color = 'black', label = 'baseline: {}'.format(np.round(base,)))
        ax.set_title('baseline value: {} \n Avg 100 Episodes: {}'.format(np.round(base,2), np.round(avg_reward_list[-1],2)))

        # for y axis poilcy
        policy_x_0 = np.zeros(n_x)
        policy_x_0_true = np.zeros(n_x)
        # for x axis poilcy
        policy_y_0 = np.zeros(n_x)
        policy_y_0_true = np.zeros(n_x)

        for ix in range(n_x):
            for iy in range(n_x):
                state = np.array([t0,x_space[ix],y_space[iy]])
                action = self.AC.actor(tf.expand_dims(tf.convert_to_tensor(state),0))
                
                P[ix][iy] = action[0]

                v1= self.AC.critic([tf.expand_dims(tf.convert_to_tensor(state),0),action])
                
                V1[ix][iy] = v1
                
               
               
                if (ix == 0):
                    policy_x_0[iy] = action[0][1]
                    
                    policy_x_0_true[iy] = A_t[t0][ix][iy][1]
                if (iy == 0):
                    policy_y_0[ix] = action[0][0]
                    policy_y_0_true[ix] = A_t[t0][ix][iy][0]

        error_v_1 = np.abs(V_t[t0] - V1)
        
        self.mean_abs_error_v1.append(np.mean(np.abs(V_t[t0] - V1)))
        

        error_P_x_0 = np.abs(policy_x_0 - policy_x_0_true)
        self.mean_abs_error_P_x_0.append(np.mean(np.abs(policy_x_0 - policy_x_0_true)))

        error_P_y_0 = np.abs(policy_y_0 - policy_y_0_true)
        self.mean_abs_error_P_y_0.append(np.mean(np.abs(policy_y_0 - policy_y_0_true)))

        X,Y = np.meshgrid(x_space, y_space)

        ax = fig.add_subplot(2, 4, 2, projection = '3d')

        ax.plot_surface(X,Y, V1, label = 'approx value function 1')
        
        ax.plot_surface(X,Y, V_t[t0], label = 'true value function', color = 'black', alpha = 0.4)
        
        ax.set_title('value function t = {}'.format(t0))

        ax = fig.add_subplot(2, 4, 3)
        ax.plot(x_space, policy_x_0, label = 'policy function approximation x = 0')
        ax.plot(x_space, policy_x_0_true,label = 'true policy function x = 0', color = 'black')
        
        ax.set_title('policy function x = 0 t = {}'.format(t0))

        ax = fig.add_subplot(2, 4, 4)
        ax.plot(x_space, policy_y_0, label = 'policy function approximation y = 0')
        ax.plot(x_space, policy_y_0_true,label = 'true policy function y= 0', color = 'black')
        
        ax.set_title('policy function y = 0 t = {}'.format(t0))

        ###########################################################
        episode_ax = self.dashboard_num * np.linspace(1,len(self.mean_abs_error_P_x_0), len(self.mean_abs_error_P_x_0))
        ax = fig.add_subplot(2, 4, 6)
        ax.scatter(episode_ax,  self.mean_abs_error_v1, label = 'Mean Abs Error 1: {}'.format(np.round(np.mean(error_v_1),2)))
        
        ax.plot(episode_ax,  self.mean_abs_error_v1)
       

        ax.set_title('abs error value function t = {} '.format(t0))
        ax.legend()
        ax.set_xlim([np.min(episode_ax)-1, np.max(episode_ax)+1])

        ax = fig.add_subplot(2, 4, 7)
        
        ax.scatter(episode_ax, self.mean_abs_error_P_x_0, label = 'Mean Abs Error x = 0: {}'.format(np.round(np.mean(error_P_x_0),2)))
        ax.plot(episode_ax, self.mean_abs_error_P_x_0)

        ax.set_title('abs error policy function x = 0 t = {} '.format(t0))
        ax.legend()
        ax.set_xlim([np.min(episode_ax)-1, np.max(episode_ax)+1])

        ax = fig.add_subplot(2, 4, 8)
        
        ax.scatter(episode_ax, self.mean_abs_error_P_y_0, label = 'Mean Abs Error y = 0: {}'.format(np.round(np.mean(error_P_y_0),2)))
        ax.plot(episode_ax, self.mean_abs_error_P_y_0)

        ax.set_title('abs error policy function y = 0 t = {} '.format(t0))
        ax.legend()
        ax.set_xlim([np.min(episode_ax)-1, np.max(episode_ax)+1])

        #terminal value function
        for ix in range(n_x):
            for iy in range(n_x):
                state = np.array([self.N-1,x_space[ix], y_space[iy]])
                action = self.AC.actor(tf.expand_dims(tf.convert_to_tensor(state),0))
                
                P[ix] = action

                v1 = self.AC.critic([tf.expand_dims(tf.convert_to_tensor(state),0),action])
                
                V1[ix] = v1
                
               
        
        ax = fig.add_subplot(2, 4, 5, projection = '3d')
        ax.plot_surface(X,Y, V1, label = 'approx value function 1')
        
        ax.plot_surface(X,Y, V_t[self.N-1], label = 'true value function', color = 'black', alpha = 0.4)
        
        ax.set_title('terminal value function t = {}'.format(self.N-1))
        
        fig.set_size_inches(w = 15, h= 7.5)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.15, hspace=0.3)
        #fig.savefig('.\Bilder_Episoden\TD3_Dashboard_Episode_{}'.format(len(avg_reward_list)))
        plt.show()
    def plots(self,n_x,V_t,A_t):
        
        n_a = 20
        
        state_space = np.linspace(-2,2, n_x)
        
        fig, ax = plt.subplots(3)
 
        V1 = np.zeros(n_x)

        P = np.zeros(n_x)

        V2 = np.zeros(n_x)

       
        t0 = self.N-1
        for ix in range(n_x):
           state = np.array([t0,state_space[ix]])
           action = self.AC.actor(tf.expand_dims(tf.convert_to_tensor(state),0))
          
           P[ix] = action

           v1,v2 = self.AC.critic([tf.expand_dims(tf.convert_to_tensor(state),0),action])
           
           V1[ix] = v1
          
           V2[ix] = v2
        
        ax[0].plot(state_space, V1, label = 'critic_1')
        ax[0].plot(state_space, V2,label = 'critic_2')
        ax[0].plot(state_space, V_t[t0], label = 'true_solution')
        ax[0].legend()
        ax[0].set_title('value function')

        ax[1].plot(state_space, P, label = 'approximation')
        ax[1].plot(state_space, A_t[t0],label = 'true_solution')
        ax[1].legend()
        ax[1].set_title('policy function')
    

        X = np.zeros(self.N)
        
        for i in range(self.N-1):
            state = np.array([i,X[i]])
            B_t = np.random.normal()
            X[i+1] = X[i] + (X[i] + self.AC.actor(tf.expand_dims(tf.convert_to_tensor(state),0)))*self.dt + self.sig * np.sqrt(self.dt) * B_t
            
        ax[2].plot(np.linspace(0, self.T, self.N), X, label = 'trajectorie')
       
        ax[2].set_ylim([-4,4])
        ax[2].fill_between(np.linspace(0, self.T, self.N),-self.r,self.r , color = 'green', alpha = 0.3)
        
        
        plt.show()

if __name__ == "__main__":

    lqr = CaseOne()
    n_x = 30
    
    V_t,A_t,base = LQR.Solution_2_D(lqr).create_solution(n_x)
    #V_t, A_t, base = LQR.Solution(lqr).dynamic_programming(n_x)
    
    lqr.run_episodes(n_x,V_t,A_t, base)
    