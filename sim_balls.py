import tensorflow as tf
import keras
from keras import layers, losses
import numpy as np
import matplotlib.pyplot as plt
#import linear_quadratic_regulator_DP as LQR

     
class SimBalls():

    def __init__(self):
        # dX_t = (A X_t + B u_t) dt + sig * dB_t
        self.sig = 1

        # f(x,u) = f_A ||x||^2 + f_B ||u||^2
        self.discrete_problem = False
       

        self.state_dim = 2
        self.action_dim = 2
       
        self.T = 1
        self.N = 1000
        self.max_steps = 2000
        self.dt = self.T / self.N

        self.r1 = 1

        self.r2 = 3

        self.dashboard_num = 100


 
    def start_state(self):
        r1 = self.r1 + self.dt
        r2 = self.r2 - self.dt
        start_r = (r2 -r1)* np.random.rand() + r1
        random_pi = 2*np.pi *np.random.rand()
        
        X = np.array([start_r*np.cos(random_pi),start_r*np.sin(random_pi)])

        return X
        

    def check_if_done(self,n,x):
      
        if n == self.max_steps-1:
            return True, None
        else:
            norm = np.linalg.norm(x)
          
            if ( norm <= self.r1) or (self.r2 <= norm):
                if ( norm <= self.r1):
                    return True,'exit_A'
                if ( self.r2 <= norm):
                    return True, 'exit_C'
            else:
                return False, None

    def run_simulation(self, num_sim):
        fig = plt.figure(figsize= (6,6))
        for sim in range(num_sim):
            X = np.nan*np.ones((self.max_steps,2), dtype= np.float32)
            x,y = np.nan*np.ones(self.max_steps, dtype= np.float32), np.zeros(self.max_steps, dtype= np.float32)
            X[0] = self.start_state()
            n = 0
            while(True):
                x[n] = X[n][0]
                y[n] = X[n][1]
                state = np.array([X[n][0],X[n][1]], np.float32)
                done, exits = self.check_if_done(n,state)

             
                state = tf.expand_dims(tf.convert_to_tensor(state),0)
                
                if (done):
                    print('done')      
                else:

                    X[n+1] =  X[n] + self.sig*np.sqrt(self.dt)  * np.random.normal(size = 2)
                    
                if(done):
                    x[n:] = x[n]
                    y[n:] = y[n]
                    break
                else:
                    n += 1
                    if (n >= self.max_steps):
                        print(n,'Schritte')
               
            plt.plot(x, y)
        time = np.linspace(0,2*np.pi,100)
        circ_x_1 = self.r1*np.cos(time)
        circ_y_1 = self.r1*np.sin(time)
        plt.plot(circ_x_1, circ_y_1, color = 'black')
        
        circ_x_2 = self.r2*np.cos(time)
        circ_y_2 = self.r2*np.sin(time)
        plt.plot(circ_x_2, circ_y_2, color = 'black')
        fig.savefig('.\Sim_Balls_Committor')
        #plt.show()


if __name__ == "__main__":


    balls = SimBalls()

    balls.run_simulation(5)

     
        
    
    