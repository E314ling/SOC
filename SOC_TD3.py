import tensorflow as tf
import keras
from keras import layers, losses
import numpy as np
import matplotlib.pyplot as plt
#import linear_quadratic_regulator_DP as LQR



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

    def __init__(self, state_dim, action_dim, load_model):

        self.batch_size = 128
        self.max_memory_size = 10000

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = 1
        self.tau = 0.001
        self.lower_action_bound = -1
        self.upper_action_bound = 1

        self.buffer = experience_memory(self.max_memory_size, self.batch_size, self.state_dim, self.action_dim)

        
        # init the neural nets
        if (load_model):
            self.critic_1 = tf.keras.models.load_model('./Models/LQR_2D_TD3_critic_1.h5')
            self.target_critic_1 = tf.keras.models.load_model('./Models/LQR_2D_TD3_critic_1.h5')

            self.critic_2 = tf.keras.models.load_model('./Models/LQR_2D_TD3_critic_2.h5')
            self.target_critic_2 = tf.keras.models.load_model('./Models/LQR_2D_TD3_critic_2.h5')


            self.actor = tf.keras.models.load_model('./Models/LQR_2D_TD3_Actor.h5')
            self.target_actor = tf.keras.models.load_model('./Models/LQR_2D_TD3_Actor.h5')
        else:
            self.critic_1 = self.get_critic_NN()
            self.target_critic_1 = self.get_critic_NN()
            #self.target_critic_1.set_weights(self.critic_1.get_weights())

            self.critic_2 = self.get_critic_NN()
            self.target_critic_2 = self.get_critic_NN()
            #self.target_critic_2.set_weights(self.critic_2.get_weights())

           
            
            self.actor = self.get_actor_NN()
            self.target_actor = self.get_actor_NN()
            self.target_actor.set_weights(self.actor.get_weights())


        self.critic_lr = 0.0001
        self.critic_optimizer_1 = tf.keras.optimizers.Adam(self.critic_lr)
        self.critic_optimizer_2 = tf.keras.optimizers.Adam(self.critic_lr)
       
        self.actor_lr = 0.0001
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)
      
        self.var = 2
        self.var_decay = 0.999
        self.lr_decay = 1

        self.update_frames = 2
        self.actor_loss = []
       
    def save_model(self):
        self.critic_1.save('./Models/LQR_2D_TD3_critic_1.h5')
        self.critic_2.save('./Models/LQR_2D_TD3_critic_2.h5')
        
        self.actor.save('./Models/LQR_2D_TD3_actor.h5')
    def update_lr(self):
        self.critic_lr = self.critic_lr * self.lr_decay
        self.critic_optimizer_1 = tf.keras.optimizers.Adam(self.critic_lr)
        self.critic_optimizer_2 = tf.keras.optimizers.Adam(self.critic_lr)
        
        self.actor_lr = self.actor_lr * self.lr_decay
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)

    def update_var(self):
        self.var = np.max([0.1,self.var * self.var_decay])
    
    @tf.function
    def update_critic(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        target_actions = self.target_policy(next_state_batch) 
        target_1, target_2 = self.target_critic_1([next_state_batch, target_actions]),self.target_critic_2([next_state_batch, target_actions])
        
        target_vals =  tf.minimum(target_1, target_2)

        y = tf.stop_gradient(reward_batch + (1-done_batch)* self.gamma*target_vals)

        with tf.GradientTape() as tape1:
            tape1.watch(self.critic_1.trainable_variables)
            
            critic_value_1 = self.critic_1([state_batch, action_batch])
            
            critic_loss_1 = tf.reduce_mean((y-critic_value_1)**2)

        with tf.GradientTape() as tape2:
            tape2.watch(self.critic_2.trainable_variables)
            
            critic_value_2 = self.critic_2([state_batch, action_batch])
            
            critic_loss_2 = tf.reduce_mean((y-critic_value_2)**2)

        
        critic_grad_1 = tape1.gradient(critic_loss_1, self.critic_1.trainable_variables) 
        self.critic_optimizer_1.apply_gradients(zip(critic_grad_1, self.critic_1.trainable_variables))

        critic_grad_2 = tape2.gradient(critic_loss_2, self.critic_2.trainable_variables)
        self.critic_optimizer_2.apply_gradients(zip(critic_grad_2, self.critic_2.trainable_variables))

        return critic_loss_1

    @tf.function
    def update_actor(self, state_batch):
        
        with tf.GradientTape() as tape:
            
            actions = self.actor(state_batch)
            critic_value_1,critic_value_2 = self.critic_1([state_batch, actions]), self.critic_2([state_batch, actions])
            actor_loss = tf.math.reduce_mean(tf.minimum(critic_value_1,critic_value_2))
            
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )
        return actor_loss

   
    def learn(self,frame_num):
        # get sample

        record_range = min(self.buffer.buffer_counter, self.buffer.buffer_capacity)

        batch_indices = np.random.choice(record_range, self.batch_size)

        state_batch = tf.convert_to_tensor(self.buffer.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.buffer.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.buffer.reward_buffer[batch_indices])
        next_state_batch = tf.convert_to_tensor(self.buffer.next_state_buffer[batch_indices])

        done_batch = tf.convert_to_tensor(self.buffer.done_buffer[batch_indices])
        
        c_loss_1 = self.update_critic(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
        
        if (frame_num % self.update_frames == 0 and frame_num != 0):
            actor_loss = self.update_actor(state_batch)
            self.actor_loss.append(actor_loss)
           

    @tf.function
    def update_target(self, target_weights, weights):
        for (a,b) in zip(target_weights, weights):
            a.assign(self.tau *b + (1-self.tau) *a)

        
    def get_critic_NN(self):
        # input [state, action]
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        state_input = layers.Input(shape =(self.state_dim,))
        action_input = layers.Input(shape =(self.action_dim,))

        input = tf.concat([state_input, action_input],1)
       
        out_1 = layers.Dense(100, activation = 'relu')(input)
        out_1 = layers.BatchNormalization()(out_1)
        out_1 = layers.Dense(50, activation = 'relu')(out_1)
        out_1 = layers.Dense(25, activation = 'relu')(out_1)
        out_1 = layers.Dense(1, kernel_initializer= last_init)(out_1)

        model = keras.Model(inputs = [state_input, action_input], outputs = out_1)

        return model

    def get_actor_NN(self):
        
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(self.state_dim,))
       
        out = layers.Dense(100, activation="relu")(inputs)
        out = layers.BatchNormalization()(out)
        out = layers.Dense(50, activation="relu")(out)
        out = layers.Dense(25, activation="relu")(out)
        outputs = layers.Dense(self.action_dim, activation='tanh', kernel_initializer=last_init)(out)

        # Our upper bound is 2.0 .
        outputs = outputs * self.upper_action_bound
        model = tf.keras.Model(inputs, outputs)
        return model

    def policy(self, state):
        sampled_actions = self.actor(state)
      
        sampled_actions = sampled_actions + tf.random.normal(shape = sampled_actions.get_shape(), mean = 0.0, stddev = self.var, dtype = tf.float32)

        legal_action = tf.clip_by_value(sampled_actions, clip_value_min= self.lower_action_bound, clip_value_max =self.upper_action_bound)


        return legal_action
        
    
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
        self.sig = 0

        # f(x,u) = f_A ||x||^2 + f_B ||u||^2
        self.f_A = np.identity(2)
        self.f_B = np.identity(2)

        # g(x) = D * ||x||^2
        self.D = 0*np.identity(2)

        self.num_episodes = 12000
        self.state_dim = 2
        self.action_dim = 2
        self.AC = ActorCritic(self.state_dim, self.action_dim, False)

        self.T = 1
        self.N = 40
        self.dt = self.T / self.N

        self.r1 = 1

        self.r2 = 3

        self.dashboard_num = 4000
      

    def f(self, n,x,a):

        y1= np.dot(np.transpose(x), self.f_A)
        
        y2 = np.dot(np.transpose(a), self.f_B)
      
        return np.float32(np.dot(y1,x) + np.dot(y2,a))
        
    def g(self, n,x):
        y = np.dot(np.transpose(x), self.D)
        return np.dot(y,x)
    
   

    def start_state(self):
        r1 = self.r1 + 0.1
        r2 = self.r2 - 0.1
        start_r = (r2 -r1)* np.random.rand() + r1
        random_pi = 2*np.pi *np.random.rand()
        
        X = np.array([start_r*np.cos(random_pi),start_r*np.sin(random_pi)])

        return X
        

    def check_if_done(self,n,x):
      
        if n == self.N-1:
            return True
        else:
            norm = np.linalg.norm(x)**2
          
            if ( norm <= self.r1**2) or (self.r2**2 <= norm):
                if ( norm <= self.r1**2):
                    print('exit at 1', x)
                return True
            
            else:
                return False
    
    def run_episodes(self, n_x):
        ep_reward_list = []
        stopping_time_list = []
        # To store average reward history of last few episodes
        avg_reward_list = []
        avg_stopping_list = []
        X = np.zeros((self.N,2), dtype= np.float32)
        
        X[0] = self.start_state()
        
        frame_num = 0
        for ep in range(self.num_episodes):
            
            n=0
            episodic_reward = 0
            while(True):
                
                state = np.array([X[n][0],X[n][1]], np.float32)
               
                done = self.check_if_done(n,state)
                state = tf.expand_dims(tf.convert_to_tensor(state),0)
                
                action = self.AC.policy(state)[0]
                
                if (done):
                    reward = self.g(n,X[n])
                    
                    X = np.zeros((self.N,2), dtype= np.float32)
                    X[0] = self.start_state()
                    
                    new_state = np.array([X[0][0],X[0][1]], np.float32)
                    new_state = tf.expand_dims(tf.convert_to_tensor(new_state),0)
                         
                else:
                    
                    reward = self.f(n,X[n], action)
                    #print('X[n], action,reward ', X[n], action,reward )
                    
                    #X[n+1] =  X[n] + (X[n] - action)*self.dt + np.sqrt(self.sig*self.dt)  * np.random.normal(size = 2)
                    
                    X[n+1] =  (X[n] + action) + self.sig*np.random.normal(2)
                  
                    new_state = np.array([X[n+1][0],X[n+1][1]], np.float32)
                   
                    new_state = tf.expand_dims(tf.convert_to_tensor(new_state),0)
                
                
                self.AC.buffer.record((state.numpy()[0],action,reward, new_state.numpy()[0], done))
                
                episodic_reward += reward
                # warm up
                if (ep >= 100):
                    self.AC.learn(n)
                    
                    self.AC.update_target(self.AC.target_critic_1.variables, self.AC.critic_1.variables)
                    self.AC.update_target(self.AC.target_critic_2.variables, self.AC.critic_2.variables)
                 
                    if (n % self.AC.update_frames == 0 and n != 0):
                     
                        self.AC.update_target(self.AC.target_actor.variables, self.AC.actor.variables)
                        self.AC.update_lr()
                        self.AC.update_var()
                frame_num += 1

                if(done):
                    stopping_time_list.append(n)
                    avg_stopping_time = np.mean(stopping_time_list[-100:])
                    avg_stopping_list.append(avg_stopping_time)
                    break
                else:
                    n += 1

            if (ep % self.dashboard_num == 0 and ep >100):
                self.dashboard(n_x,avg_reward_list,avg_stopping_list,self.AC.actor_loss)
            
            if (ep >= 100):
                
                ep_reward_list.append(episodic_reward)
                # Mean of last 40 episodes
                avg_reward = np.mean(ep_reward_list[-100:])
                print("Episode * {} * Avg Reward is ==> {} * Avg Stopping Time is ==> {}, var ==> {}, actor_lr ==> {}".format(ep, avg_reward,avg_stopping_time, self.AC.var, self.AC.actor_lr))
                avg_reward_list.append(avg_reward)
        # Plotting graph
        # Episodes versus Avg. Rewards
        self.AC.save_model()
        self.dashboard(n_x,avg_reward_list,avg_stopping_list,self.AC.actor_loss)
    
    def run_simulation(self, num_sim):
        fig = plt.figure()
        for sim in range(num_sim):
            X = np.zeros((self.N,2), dtype= np.float32)
            x,y = np.zeros(self.N, dtype= np.float32), np.zeros(self.N, dtype= np.float32)
            X[0] = self.start_state()
            n = 0
            while(True):
                x[n] = X[n][0]
                y[n] = X[n][1]
                state = np.array([X[n][0],X[n][1]], np.float32)
                done = self.check_if_done(n,state)
                state = tf.expand_dims(tf.convert_to_tensor(state),0)
                
                action = self.AC.actor(state).numpy()[0]

                if (done):
                    print('done')      
                else:

                 
                    #X[n+1] =  X[n] + (X[n] - action)*self.dt + np.sqrt(self.sig*self.dt)  * np.random.normal(size = 2)
                    X[n+1] =  (X[n] + action) + self.sig*np.random.normal(2)
                    print('X[n], action,X[n] + action, X[n+1]', X[n], action,X[n] + action, X[n+1])
                if(done):
                    x[n:] = x[n]
                    y[n:] = y[n]
                    break
                else:
                    n += 1
               
            plt.plot(x, y)
        time = np.linspace(0,2*np.pi,self.N)
        circ_x_1 = self.r1*np.cos(time)
        circ_y_1 = self.r1*np.sin(time)
        plt.plot(circ_x_1, circ_y_1, color = 'black')

        circ_x_2 = self.r2*np.cos(time)
        circ_y_2 = self.r2*np.sin(time)
        plt.plot(circ_x_2, circ_y_2, color = 'black')
        plt.show()
            
    def dashboard(self,n_x,avg_reward_list, avg_stopping_list,actor_loss):
        self.run_simulation(10)
        x_space = np.linspace(self.r1,self.r2, n_x)
        y_space = np.linspace(self.r1,self.r2, n_x)

        fig = plt.figure()
        
        V1 = np.zeros((n_x,n_x))
        P = np.zeros((n_x,n_x,2))
        V2 = np.zeros((n_x,n_x))
        t0 = 1

        ax = fig.add_subplot(2, 3, 1)
        ax.plot(avg_reward_list)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Avg. Epsiodic Reward')

        ax = fig.add_subplot(2, 3, 2)
        ax.plot(avg_stopping_list)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Avg. Stopping Time')

        ax = fig.add_subplot(2, 3, 3)
        ax.plot(actor_loss)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Actor Loss')
        

        # for y axis poilcy
        policy_x_0 = np.zeros(n_x)
       
        # for x axis poilcy
        policy_y_0 = np.zeros(n_x)
        

        for ix in range(n_x):
            for iy in range(n_x):
                state = np.array([x_space[ix],y_space[iy]])
                action = self.AC.actor(tf.expand_dims(tf.convert_to_tensor(state),0))
                
                P[ix][iy] = action[0]

                v1,v2 = self.AC.critic_1([tf.expand_dims(tf.convert_to_tensor(state),0),action]),self.AC.critic_2([tf.expand_dims(tf.convert_to_tensor(state),0),action])
                
                V1[ix][iy] = v1
                
                V2[ix][iy] = v2

               
                if (ix == 0):
                    policy_x_0[iy] = action[0][1]
                    
                    
                if (iy == 0):
                    policy_y_0[ix] = action[0][0]
                  

       

        X,Y = np.meshgrid(x_space, y_space)

        ax = fig.add_subplot(2, 3, 4, projection = '3d')

        ax.plot_surface(X,Y, V1, label = 'approx value function 1')
        ax.plot_surface(X,Y, V2, label = 'approx value function 2')
       
        
        ax.set_title('value function t = {}'.format(t0))

        ax = fig.add_subplot(2, 3, 5)
        ax.plot(x_space, policy_x_0, label = 'policy function approximation x = 0')
        
        ax.set_title('policy function x = 0 t = {}'.format(t0))

        ax = fig.add_subplot(2, 3, 6)
        ax.plot(x_space, policy_y_0, label = 'policy function approximation y = 0')
        
        ax.set_title('policy function y = 0 t = {}'.format(t0))

        
        fig.set_size_inches(w = 15, h= 7.5)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.15, hspace=0.3)
        #fig.savefig('.\Bilder_Episoden\TD3_Dashboard_Episode_{}'.format(len(avg_reward_list)))
        plt.show()
    

if __name__ == "__main__":

    lqr = CaseOne()
    n_x = 30
    
    #V_t, A_t, base = LQR.Solution(lqr).dynamic_programming(n_x)
    
    lqr.run_episodes(n_x)
    