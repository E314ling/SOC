import tensorflow as tf
import keras
from keras import layers, losses
import numpy as np
import matplotlib.pyplot as plt


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

        self.batch_size = 512
        self.max_memory_size = 1000000

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = 0.9999
        self.tau = 0.001
        self.tau_actor = 0.001
        self.lower_action_bound = -15
        self.upper_action_bound = 15

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


        self.critic_lr = 0.0003
        self.critic_optimizer_1 = tf.keras.optimizers.Adam(self.critic_lr)
        self.critic_optimizer_2 = tf.keras.optimizers.Adam(self.critic_lr)
       
        self.actor_lr = 0.0003
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)
      
        self.var = 0.5
        self.var_decay = 0
        self.lr_decay = 1
        self.var_min = 0.1 #/ self.upper_action_bound
        self.var_target = 0.2 #/ self.upper_action_bound
        self.update_frames = 2

       
    def save_model(self):
        self.critic_1.save('./Models/pendulum_TD3_critic_1.h5')
        self.critic_2.save('./Models/pendulum_critic_2.h5')
        
        self.actor.save('./Models/pendulum_TD3_actor.h5')
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
            # for maximization put a minus infront
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

        state_input = layers.Input(shape =(self.state_dim,))
        action_input = layers.Input(shape =(self.action_dim,))

        input = tf.concat([state_input, action_input],1)
       
        out_1 = layers.Dense(128, activation = 'relu', kernel_regularizer='l2')(input)
      
        out_1 = layers.Dense(128, activation = 'relu', kernel_regularizer='l2')(out_1)
       
        out_1 = layers.Dense(1, kernel_initializer= last_init, kernel_regularizer='l2')(out_1)

        model = keras.Model(inputs = [state_input, action_input], outputs = out_1)

        return model

    def get_actor_NN(self):
        
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(self.state_dim,))
        
        out = layers.Dense(128, activation="relu", kernel_regularizer='l2')(inputs)
        
        out = layers.Dense(128, activation="relu", kernel_regularizer='l2')(out)
        
        outputs = layers.Dense(self.action_dim, activation='linear', kernel_initializer=last_init, kernel_regularizer='l2')(out)
        outputs = tf.clip_by_value(outputs, clip_value_max= 1, clip_value_min=-1)
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

    def __init__(self,run):
        # dX_t = (A X_t + B u_t) dt + sig * dB_t
        self.run = run
        self.sig = 0.5
        self.p, self.q = 1,1
        self.c1, self.c2 = 0.5,0.5
        self.eps = 0.25 

      

        self.num_episodes = 5000
        self.warmup = 0
        self.state_dim = 1
        self.action_dim = 1
        self.AC = ActorCritic(self.state_dim, self.action_dim, False)

        self.T = 1
        self.N = 20
        self.dt = self.T/self.N

        self.max_iterations = int(self.N/5) * 200

        
        self.dashboard_num = 100
        self.mean_abs_error_v1 = []
        self.mean_abs_error_v2 = []
        self.mean_abs_error_P_x_0 = []
        self.mean_abs_error_P_y_0 = []


    def fill_buffer(self, num_episodes):
        print('start warm up...')
        
        for ep in range(num_episodes):
            X = np.zeros(self.max_iterations, dtype= np.float32)
            X[0] = self.start_state()
            done = False
            n = 0
            while(True):

                state = np.array([X[n]], np.float32)
               
                state = tf.expand_dims(tf.convert_to_tensor(state),0)
                done = self.check_if_done(n,state)

                action = tf.convert_to_tensor(2* np.random.rand() -1)
                action_env = self.AC.upper_action_bound * action
                if (done):
                    reward = self.g(n,X[n],action_env)
                    
                    X = np.zeros(self.max_iterations, dtype= np.float32)
                   
                    X[0] = self.start_state()
                    new_state = np.array([X[0]], np.float32)
                    new_state = tf.expand_dims(tf.convert_to_tensor(new_state),0)
                         
                else:

                    reward = self.f(n,X[n], action_env)
                   
                    X[n+1] =  X[n] + (self.c1*np.sin(X[n]) - self.c2*np.cos(X[n])*action_env)*self.dt + self.sig*np.sqrt(self.dt)  * np.random.normal()


                    new_state = np.array([X[n+1]], np.float32)
                    new_state = tf.expand_dims(tf.convert_to_tensor(new_state),0)
                if done:
                    if n <= self.max_iterations-1:
                        self.AC.buffer.record((state.numpy()[0],action.numpy(),reward, new_state.numpy()[0], done))
                else:
                   self.AC.buffer.record((state.numpy()[0],action.numpy(),reward, new_state.numpy()[0], done))

                
               
                if(done):
                    break
                else:
                    n += 1
        print('warm up done')
        print('buffer samples: ', self.AC.buffer.buffer_counter)
                


    def f(self, n,x,a):

        return self.dt*(np.float32(self.p*a**2 + self.q))
        
    def g(self, n,x,a):

        return 0# (np.float32(n*self.q))
    
    def check_if_done(self,n,x):
        if n == self.max_iterations-1:
            print('not finished')
            return True
        else:
            if (x <= self.eps or x >= (2*np.pi -self.eps)):
                print('finished')
                return True
            else:
                return False
    

    def start_state(self):
        
        X = (2*np.pi -2*self.eps)*np.random.rand() + 2*self.eps
        #X = np.pi
        return X
    
    def run_episodes(self):
        # fill buffer with random samples
        warm_up = 50
        self.fill_buffer(warm_up)

        ep_reward_list = []
        stopping_time_list = []
        # To store average reward history of last few episodes
        avg_reward_list = []
        avg_stopping_list = []

        X = np.zeros(self.max_iterations, dtype= np.float32)
        #X[0] = 1*np.random.rand(self.state_dim) - 1
        X[0] = self.start_state()
        frame_num = 0
        for ep in range(self.num_episodes):
            
            n=0
            episodic_reward = 0
            while(True):
                #X[n] = np.clip(X[n], a_min= -3,a_max = 3)
                state = np.array([X[n]], np.float32)
               
                state = tf.expand_dims(tf.convert_to_tensor(state),0)
                done = self.check_if_done(n,state)
                #print('X[n],self.eps ,2*np.pi-self.eps, done', X[n],self.eps ,2*np.pi-self.eps, done)
                if (ep <= self.warmup):
                    action = tf.convert_to_tensor(2* np.random.rand() -1)
                    action_env = self.AC.upper_action_bound * action
                    
                else:
                    action = self.AC.policy(state)[0][0]
                    
                    action_env = self.AC.upper_action_bound*action

                if (done):
                    reward = self.g(n,X[n],action_env)
                    
                    X = np.zeros(self.max_iterations, dtype= np.float32)
                   
                    X[0] = self.start_state()
                    new_state = np.array([X[0]], np.float32)
                    new_state = tf.expand_dims(tf.convert_to_tensor(new_state),0)
                         
                else:

                    reward = self.f(n,X[n], action_env)
                   
                    X[n+1] =  X[n] + (self.c1*np.sin(X[n]) - self.c2*np.cos(X[n])*action_env)*self.dt + self.sig*np.sqrt(self.dt)  * np.random.normal()


                    new_state = np.array([X[n+1]], np.float32)
                    new_state = tf.expand_dims(tf.convert_to_tensor(new_state),0)
                
                if done:
                    if n <= self.max_iterations-1:
                        self.AC.buffer.record((state.numpy()[0],action.numpy(),reward, new_state.numpy()[0], done))
                else:
                   self.AC.buffer.record((state.numpy()[0],action.numpy(),reward, new_state.numpy()[0], done))

                
               
                episodic_reward += reward
                # warm up
                
                self.AC.learn(n)
                
                if (n % self.AC.update_frames == 0 and n != 0):
                    self.AC.update_target_critic(self.AC.target_critic_1.variables, self.AC.critic_1.variables)
                    self.AC.update_target_critic(self.AC.target_critic_2.variables, self.AC.critic_2.variables)
                    self.AC.update_target_actor(self.AC.target_actor.variables, self.AC.actor.variables)
                    self.AC.update_lr()
                    self.AC.update_var()
                frame_num += 1
                if(done):
                    stopping_time_list.append(self.dt*(n+0.5))
                    avg_stopping_time = np.mean(stopping_time_list[-1000:])
                    avg_stopping_list.append(avg_stopping_time)
                    break
                else:
                    n += 1

            if (ep % self.dashboard_num == 0):
                #self.dashboard(avg_reward_list,self.AC,avg_stopping_list)
                self.save_all(ep_reward_list,avg_reward_list,self.AC)
           
            
            ep_reward_list.append(episodic_reward)
            # Mean of last 40 episodes
            avg_reward = np.mean(ep_reward_list[-500:])
            print("Episode * {} * Avg Reward is ==> {}, var ==> {}, actor_lr ==> {}".format(ep, avg_reward, self.AC.var, self.AC.actor_lr))
            avg_reward_list.append(avg_reward)
        # Plotting graph
        # Episodes versus Avg. Rewards
        self.AC.save_model()
        self.save_all(ep_reward_list,avg_reward_list,self.AC)
        #self.dashboard(avg_reward_list,self.AC,avg_stopping_list)


    def save_all(self,ep_reward_list, avg_reward_list, AC:ActorCritic):
        steps = len(avg_reward_list)
        jacobi_V = np.load('./jacobi_value_fct_pendulum.npy')
        jacobi_u = np.load('./jacobi_policy_fct_pendulum.npy')
        n_x = len(jacobi_V)
        # save avg_reward and base
        np.save('./Saved_Runs/pendulum/ep_reward_list_{}_run_{}'.format(steps, self.run), np.array(ep_reward_list))
        np.save('./Saved_Runs/pendulum/avg_reward_step_{}_run_{}'.format(steps, self.run), np.array(avg_reward_list))


        x_space = np.linspace(self.eps,2*np.pi -self.eps, n_x)
        

        fig = plt.figure()
        
        V1 = np.zeros((n_x))
        V2 = np.zeros((n_x))

        P = np.zeros((n_x))
        

        True_P = jacobi_u
        True_V = jacobi_V
        
       
        for ix in range(n_x):
            
            state = np.array([x_space[ix]])
            
            action = self.AC.actor(tf.expand_dims(tf.convert_to_tensor(state),0))
            
            
            P[ix] = AC.upper_action_bound* action[0]
            

            v1,v2 = self.AC.critic_1([tf.expand_dims(tf.convert_to_tensor(state),0),action]),self.AC.critic_2([tf.expand_dims(tf.convert_to_tensor(state),0),action])
            
            V1[ix] = v1
            
            V2[ix] = v2
            
            
        #save value function and policy
        np.save('./Saved_Runs/pendulum/value_fct_1_{}_run_{}'.format(steps, self.run), V1)
        np.save('./Saved_Runs/pendulum/value_fct_2_{}_run_{}'.format(steps, self.run), V2)
        np.save('./Saved_Runs/pendulum/true_value_fct',True_V )

        np.save('./Saved_Runs/pendulum/policy_{}_run_{}'.format(steps, self.run), P)


        np.save('./Saved_Runs/pendulum/true_policy', True_P)



        # save x and y space

        np.save('./Saved_Runs/pendulum/X_space', x_space)


    def dashboard(self,avg_reward_list,AC: ActorCritic,avg_stopping_list):
        jacobi_V = np.load('./jacobi_value_fct_pendulum.npy')
        jacobi_u = np.load('./jacobi_policy_fct_pendulum.npy')
        n_x = len(jacobi_V)

        x_space = np.linspace(self.eps,2*np.pi -self.eps, n_x)
        

        fig = plt.figure()
        
        V1 = np.zeros(n_x)
        P = np.zeros(n_x)
        V2 = np.zeros(n_x)
        

        ax = fig.add_subplot(2, 3, 1)
        ax.plot(avg_reward_list)
        ax.set_xlabel('Episode')
        ax.set_xlim([0,self.num_episodes])
        ax.set_ylabel('Avg. Epsiodic Reward')
        
        ax.legend()
        if(len(avg_reward_list) > 0):
            ax.set_title('Avg Cost 500 Episodes: {}'.format(np.round(avg_reward_list[-1],2)))

        
        ax = fig.add_subplot(2, 3, 2)
        ax.plot(avg_stopping_list)
        ax.set_xlabel('Episode')
        ax.set_xlim([0,self.num_episodes])
        ax.set_ylabel('Avg. Epsiodic Stopping Time')
        
        ax.legend()
        if(len(avg_reward_list) > 0):
            ax.set_title('Avg Cost 500 Episodes: {}'.format(np.round(avg_stopping_list[-1],2)))

       

        for ix in range(n_x):
            
                state = np.array([x_space[ix]])

                
                action = self.AC.actor(tf.expand_dims(tf.convert_to_tensor(state),0))
                
                P[ix] = AC.upper_action_bound* action.numpy()[0]

                v1,v2 = self.AC.critic_1([tf.expand_dims(tf.convert_to_tensor(state),0),action]),self.AC.critic_2([tf.expand_dims(tf.convert_to_tensor(state),0),action])
                
                V1[ix] = v1
                
                V2[ix] = v2

               

        error_v_1 = (jacobi_V - V1)**2
        error_v_2 = (jacobi_V - V2)**2
        self.mean_abs_error_v1.append(np.mean((jacobi_V - V1)**2))
        self.mean_abs_error_v2.append(np.mean((jacobi_V - V2)**2))

        error_P_x_0 = (jacobi_u[1:-1] - P[1:-1])**2
        self.mean_abs_error_P_x_0.append(np.mean((jacobi_u[1:-1] - P[1:-1])**2))

        ax = fig.add_subplot(2, 3,3)

        ax.plot(x_space, V1, label = 'approx value function 1')
        ax.plot(x_space, V2, label = 'approx value function 1')
        ax.plot(x_space, jacobi_V, label = 'jacobi value function', color = 'black', alpha = 0.4)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('value function')

        ax = fig.add_subplot(2, 3,4)
        ax.plot(x_space[1:-1], P[1:-1], label = 'policy function approximation')
        ax.plot(x_space[1:-1], jacobi_u[1:-1],label = 'jacobi policy function', color = 'black')
        
        ax.set_xlabel('y')
        ax.set_title('policy functions')

       

        ###########################################################
        episode_ax = self.dashboard_num * np.linspace(1,len(self.mean_abs_error_P_x_0), len(self.mean_abs_error_P_x_0))
        ax = fig.add_subplot(2, 3,5)
        #ax.scatter(episode_ax,  self.mean_abs_error_v1, label = 'MAE 1: {}'.format(np.round(np.mean(error_v_1),2)))
        #ax.scatter(episode_ax,  self.mean_abs_error_v2, label = 'MAE 2: {}'.format(np.round(np.mean(error_v_2),2)))
        ax.set_xlim([0,self.num_episodes])
        ax.plot(episode_ax,  self.mean_abs_error_v1, label = 'MSE 1: {}'.format(np.round(np.mean(error_v_1),2)))
        ax.plot(episode_ax,  self.mean_abs_error_v2, label = 'MSE 2: {}'.format(np.round(np.mean(error_v_2),2)))
        ax.set_xlabel('Episode')
        
        ax.set_title('MSE value functions')
        ax.legend()
        

        ax = fig.add_subplot(2, 3,6)
        
        #ax.scatter(episode_ax, self.mean_abs_error_P_x_0, label = 'MAE policy approximation x = 0: {}'.format(np.round(np.mean(error_P_x_0),2)))
        ax.plot(episode_ax, self.mean_abs_error_P_x_0, label = 'MSE: {}'.format(np.round(np.mean(error_P_x_0),2)))
        ax.set_xlim([0,self.num_episodes])
        ax.set_title('MSE policy functions')
        ax.legend()
        ax.set_xlabel('Episode')
        

       
        
        fig.set_size_inches(w = 18, h= 8)
        fig.tight_layout()
        plt.subplots_adjust(wspace=0.15, hspace=0.3)
        fig.savefig('.\Bilder_SOC\pendulum_TD3_Episode_{}'.format(len(avg_reward_list)))
        #plt.show()
    
if __name__ == "__main__":

    runs = 5

    for i in range(runs):

        pendulum = CaseOne(i)
    
        pendulum.run_episodes()
    