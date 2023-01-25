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

        self.batch_size = 256
        self.max_memory_size = 100000

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = 1
        self.tau = 0.001
        self.tau_actor = 0.001
        self.lower_action_bound = -5
        self.upper_action_bound = 5

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
      
        self.var = 1
        self.var_decay = 0
        self.lr_decay = 1
        self.var_min = 0.2
        self.var_target = 0.2
        self.update_frames = 2

       
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
        self.var = np.max([self.var_min,self.var * self.var_decay])
    
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


    @tf.function
    def update_actor(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):
        
        with tf.GradientTape() as tape:
            tape.watch(self.actor.trainable_variables)
            actions = self.actor(state_batch)
            critic_value_1,critic_value_2 = self.critic_1([state_batch, actions]), self.critic_2([state_batch, actions])
            actor_loss = tf.math.reduce_mean(tf.minimum(critic_value_1,critic_value_2))
            
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
    def update_target_critic(self, target_weights, weights):
        for (a,b) in zip(target_weights, weights):
            a.assign(self.tau *b + (1-self.tau) *a)

    @tf.function
    def update_target_actor(self, target_weights, weights):
        for (a,b) in zip(target_weights, weights):
            a.assign(self.tau_actor *b + (1-self.tau_actor) *a)

    def get_critic_NN(self):
        # input [state, action]
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        state_input = layers.Input(shape =(self.state_dim+1,))
        action_input = layers.Input(shape =(self.action_dim,))

        input = tf.concat([state_input, action_input],1)
       
        out_1 = layers.Dense(256, activation = 'relu')(input)
        out_1 = layers.BatchNormalization()(out_1)
        out_1 = layers.Dense(256, activation = 'relu')(out_1)
       
        out_1 = layers.Dense(1, kernel_initializer= last_init)(out_1)

        model = keras.Model(inputs = [state_input, action_input], outputs = out_1)

        return model

    def get_actor_NN(self):
        
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(self.state_dim+1,))
        
        out = layers.Dense(256, activation="relu")(inputs)
        out = layers.BatchNormalization()(out)
        out = layers.Dense(256, activation="relu")(out)
        
        outputs = layers.Dense(self.action_dim, activation='tanh', kernel_initializer=last_init)(out)

        # Our upper bound is 2.0 .
        #outputs = outputs * self.upper_action_bound
        model = tf.keras.Model(inputs, outputs)
        return model

    def policy(self, state):
        sampled_actions = self.actor(state)
        
        #noice = self.noice_Obj()
      
        sampled_actions = sampled_actions + tf.random.normal(shape = sampled_actions.get_shape(),mean= 0, stddev=self.var, dtype= tf.float32)

        legal_action = tf.clip_by_value(sampled_actions, clip_value_min= -1, clip_value_max=1)
        return legal_action
        #return [np.squeeze(sampled_actions)]
    
    @tf.function
    def target_policy(self, state):
        sampled_actions = self.target_actor(state)
        
        #noice = self.noice_Obj()
        noice = tf.random.normal(shape = sampled_actions.get_shape(), mean = 0.0, stddev = self.var_target, dtype = tf.float32)
        noice = tf.clip_by_value(noice, -0.5, 0.5)
        sampled_actions = sampled_actions + noice
        
        legal_action = tf.clip_by_value(sampled_actions, clip_value_min = -1, clip_value_max =1)

        return legal_action
        
class CaseOne():

    def __init__(self):
        # dX_t = (A X_t + B u_t) dt + sig * dB_t
        self.A = np.identity(2)
        self.B = np.identity(2)
        self.sig = np.sqrt(2)

        self.discrete_problem = False
        # f(x,u) = f_A ||x||^2 + f_B ||u||^2
        self.f_A = np.identity(2)
        self.f_B = np.identity(2)

        # g(x) = D * ||x||^2
        self.D = np.identity(2)

        self.num_episodes = 6100
        self.warmup = 1000
        self.state_dim = 2
        self.action_dim = 2
        self.AC = ActorCritic(self.state_dim, self.action_dim, False)

        self.T = 1
        self.N = 50
        self.dt = self.T/self.N

        self.r = 0
        self.dashboard_num = 100
        self.mean_abs_error_v1 = []
        self.mean_abs_error_v2 = []
        self.mean_abs_error_P_x_0 = []
        self.mean_abs_error_P_y_0 = []

    def f(self, n,x,a):

        y1= np.dot(np.transpose(x), self.f_A)
        
        y2 = np.dot(np.transpose(a), self.f_B)
      
        if(self.discrete_problem):
            return np.float32(np.dot(y1,x) + np.dot(y2,a))
        else:
            return self.dt*(np.float32(np.dot(y1,x) + np.dot(y2,a)))
        
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
    

    def start_state(self):
        r1 = 0 + 0.1
        r2 = 2.1 - 0.1
        start_r = (r2 -r1)* np.random.rand() + r1
        random_pi = 2*np.pi *np.random.rand()
        
        X = np.array([start_r*np.cos(random_pi),start_r*np.sin(random_pi)])

        return X
    def run_episodes(self, n_x,V_t,A_t, base):
        ep_reward_list = []
        # To store average reward history of last few episodes
        avg_reward_list = []
        X = np.zeros((self.N,2), dtype= np.float32)
        #X[0] = 1*np.random.rand(self.state_dim) - 1
        X[0] = self.start_state()
        frame_num = 0
        for ep in range(self.num_episodes):
            
            n=0
            episodic_reward = 0
            while(True):
                #X[n] = np.clip(X[n], a_min= -3,a_max = 3)
                state = np.array([n,X[n][0],X[n][1]], np.float32)
               
                state = tf.expand_dims(tf.convert_to_tensor(state),0)
                done = self.check_if_done(n,state)

                if (ep <= self.warmup):
                    action = tf.convert_to_tensor(2* np.random.rand(2) -1)
                    action_env = self.AC.upper_action_bound * action
                else:
                    action = self.AC.policy(state)[0]
                    action_env = self.AC.upper_action_bound*action

                if (done):
                    reward = self.g(n,X[n])
                    
                    X = np.zeros((self.N,2), dtype= np.float32)
                    #X[0] = 1*np.random.rand(self.state_dim) - 1

                    #X[0] = np.clip(X[0], a_min= -1,a_max = 1)
                    X[0] = self.start_state()
                    new_state = np.array([0,X[0][0],X[0][1]], np.float32)
                    new_state = tf.expand_dims(tf.convert_to_tensor(new_state),0)
                         
                else:

                    reward = self.f(n,X[n], action_env)
                    if (self.discrete_problem):
                    
                        X[n+1] =  (X[n] + action_env) + self.sig*np.random.normal(size = 2)
                    else:
                        X[n+1] =  X[n] + (X[n] + action_env)*self.dt + self.sig*np.sqrt(self.dt)  * np.random.normal(size=2)
                    new_state = np.array([(n+1),X[n+1][0],X[n+1][1]], np.float32)
                    new_state = tf.expand_dims(tf.convert_to_tensor(new_state),0)
                
                self.AC.buffer.record((state.numpy()[0],action.numpy(),reward, new_state.numpy()[0], done))
               
                episodic_reward += reward
                # warm up
                if (ep >= self.warmup):
                    self.AC.learn(n)
                    
                    if (n % self.AC.update_frames == 0 and n != 0):
                        self.AC.update_target_critic(self.AC.target_critic_1.variables, self.AC.critic_1.variables)
                        self.AC.update_target_critic(self.AC.target_critic_2.variables, self.AC.critic_2.variables)
                        self.AC.update_target_actor(self.AC.target_actor.variables, self.AC.actor.variables)
                        self.AC.update_lr()
                        self.AC.update_var()
                frame_num += 1
                if(done):
                    break
                else:
                    n += 1

            if (ep % self.dashboard_num == 0):
                self.dashboard(n_x,V_t,A_t,avg_reward_list,self.AC)
            
            if (ep >= self.warmup):
                
                ep_reward_list.append(episodic_reward)
                # Mean of last 40 episodes
                avg_reward = np.mean(ep_reward_list[-500:])
                print("Episode * {} * Avg Reward is ==> {}, var ==> {}, actor_lr ==> {}".format(ep, avg_reward, self.AC.var, self.AC.actor_lr))
                avg_reward_list.append(avg_reward)
        # Plotting graph
        # Episodes versus Avg. Rewards
        self.AC.save_model()
        self.dashboard(n_x,V_t,A_t,avg_reward_list,self.AC)
        
    def dashboard(self,n_x,V_t,A_t,avg_reward_list,AC: ActorCritic):

        x_space = np.linspace(-2,2, n_x)
        y_space = np.linspace(-2,2, n_x)

        fig = plt.figure()
        
        V1 = np.zeros((n_x,n_x))
        P = np.zeros((n_x,n_x,2))
        V2 = np.zeros((n_x,n_x))
        t0 = 1

        ax = fig.add_subplot(2, 4, 1)
        ax.plot(avg_reward_list)
        ax.set_xlabel('Episode')
        ax.set_xlim([0,self.num_episodes])
        ax.set_ylabel('Avg. Epsiodic Reward')
        #ax.hlines(base,xmin = 0, xmax = self.num_episodes, color = 'black', label = 'baseline true solution: {}'.format(np.round(base,)))
        if(len(avg_reward_list) > 0):
            ax.set_title('Avg Cost 500 Episodes: {}'.format(np.round(avg_reward_list[-1],2)))

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
                
                P[ix][iy] = AC.upper_action_bound* action[0]

                v1,v2 = self.AC.critic_1([tf.expand_dims(tf.convert_to_tensor(state),0),action]),self.AC.critic_2([tf.expand_dims(tf.convert_to_tensor(state),0),action])
                
                V1[ix][iy] = v1
                
                V2[ix][iy] = v2

               
                if (ix == 0):
                    policy_x_0[iy] = AC.upper_action_bound*action[0][1]
                    
                    policy_x_0_true[iy] = A_t[t0][ix][iy][1]
                if (iy == 0):
                    policy_y_0[ix] = AC.upper_action_bound*action[0][0]
                    policy_y_0_true[ix] = A_t[t0][ix][iy][0]

        error_v_1 = (V_t[t0] - V1)**2
        error_v_2 = (V_t[t0] - V2)**2
        self.mean_abs_error_v1.append(np.mean((V_t[t0] - V1)**2))
        self.mean_abs_error_v2.append(np.mean((V_t[t0] - V2)**2))

        error_P_x_0 = (policy_x_0 - policy_x_0_true)**2
        self.mean_abs_error_P_x_0.append(np.mean((policy_x_0 - policy_x_0_true)**2))

        error_P_y_0 = (policy_y_0 - policy_y_0_true)**2
        self.mean_abs_error_P_y_0.append(np.mean((policy_y_0 - policy_y_0_true)**2))

        X,Y = np.meshgrid(x_space, y_space)

        ax = fig.add_subplot(2, 4, 2, projection = '3d')

        ax.plot_surface(X,Y, V1, label = 'approx value function 1')
        ax.plot_surface(X,Y, V2, label = 'approx value function 1')
        ax.plot_surface(X,Y, V_t[t0], label = 'true value function', color = 'black', alpha = 0.4)
        
        ax.set_title('value function n = {}'.format(t0))

        ax = fig.add_subplot(2, 4, 3)
        ax.plot(x_space, policy_x_0, label = 'policy function approximation x = 0')
        ax.plot(x_space, policy_x_0_true,label = 'true policy function x = 0', color = 'black')
        
        ax.set_title('policy function x = 0 n = {}'.format(t0))

        ax = fig.add_subplot(2, 4, 4)
        ax.plot(x_space, policy_y_0, label = 'policy function approximation y = 0')
        ax.plot(x_space, policy_y_0_true,label = 'true policy function y= 0', color = 'black')
        
        ax.set_title('policy function y = 0 n = {}'.format(t0))

        ###########################################################
        episode_ax = self.dashboard_num * np.linspace(1,len(self.mean_abs_error_P_x_0), len(self.mean_abs_error_P_x_0))
        ax = fig.add_subplot(2, 4, 6)
        #ax.scatter(episode_ax,  self.mean_abs_error_v1, label = 'MAE 1: {}'.format(np.round(np.mean(error_v_1),2)))
        #ax.scatter(episode_ax,  self.mean_abs_error_v2, label = 'MAE 2: {}'.format(np.round(np.mean(error_v_2),2)))
        ax.set_xlim([0,self.num_episodes])
        ax.plot(episode_ax,  self.mean_abs_error_v1, label = 'MSE 1: {}'.format(np.round(np.mean(error_v_1),2)))
        ax.plot(episode_ax,  self.mean_abs_error_v2, label = 'MSE 2: {}'.format(np.round(np.mean(error_v_2),2)))

        ax.set_title('MSE value functions n = {} '.format(t0))
        ax.legend()
        

        ax = fig.add_subplot(2, 4, 7)
        
        #ax.scatter(episode_ax, self.mean_abs_error_P_x_0, label = 'MAE policy approximation x = 0: {}'.format(np.round(np.mean(error_P_x_0),2)))
        ax.plot(episode_ax, self.mean_abs_error_P_x_0, label = 'MSE: {}'.format(np.round(np.mean(error_P_x_0),2)))
        ax.set_xlim([0,self.num_episodes])
        ax.set_title('MSE policy function x = 0 n = {} '.format(t0))
        ax.legend()
        

        ax = fig.add_subplot(2, 4, 8)
        
        #ax.scatter(episode_ax, self.mean_abs_error_P_y_0, label = 'MAE policy approximation y = 0: {}'.format(np.round(np.mean(error_P_y_0),2)))
        ax.plot(episode_ax, self.mean_abs_error_P_y_0, label = 'MAE: {}'.format(np.round(np.mean(error_P_y_0),2)))

        ax.set_title('abs error policy function y = 0 n = {} '.format(t0))
        ax.set_xlim([0,self.num_episodes])
        ax.legend()
 

        #terminal value function
        for ix in range(n_x):
            for iy in range(n_x):
                state = np.array([(self.N-1),x_space[ix], y_space[iy]])
                action = self.AC.actor(tf.expand_dims(tf.convert_to_tensor(state),0))
                
                P[ix] = action

                v1,v2 = self.AC.critic_1([tf.expand_dims(tf.convert_to_tensor(state),0),action]), self.AC.critic_2([tf.expand_dims(tf.convert_to_tensor(state),0),action])
                
                V1[ix][iy] = v1
                
                V2[ix][iy] = v2
        
        ax = fig.add_subplot(2, 4, 5, projection = '3d')
        ax.plot_surface(X,Y, V1, label = 'approx value function 1')
        ax.plot_surface(X,Y, V2,label = 'approx value function 1')
        ax.plot_surface(X,Y, V_t[self.N-1], label = 'true value function', color = 'black', alpha = 0.4)
        
        ax.set_title('terminal value function n = {}'.format(self.N-1))
        
        fig.set_size_inches(w = 18, h= 8)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.15, hspace=0.3)
        fig.savefig('.\Bilder_SOC\LQR_TD3_Episode_{}'.format(len(avg_reward_list)))
        #plt.show()
    
if __name__ == "__main__":

    lqr = CaseOne()
    n_x = 40
    
    V_t,A_t,base = LQR.Solution_2_D(lqr).create_solution(n_x)
    #V_t, A_t, base = LQR.Solution(lqr).dynamic_programming(n_x)
    
    lqr.run_episodes(n_x,V_t,A_t, base)
    