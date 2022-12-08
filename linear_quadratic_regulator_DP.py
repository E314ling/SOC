import numpy as np
import matplotlib.pyplot as plt

from linear_quadratic_regulator_DQN import CaseOne
class Solution():

    def __init__(self,case_obj):
        
       # dX_t = (A X_t + B u_t) dt + sig * dB_t
        self.A = case_obj.A
        self.B = case_obj.B
        self.sig = case_obj.sig

        # f(x,u) = f_A ||x||^2 + f_B ||u||^2
        self.f_A = case_obj.f_A
        self.f_B = case_obj.f_B

        # g(x) = D * ||x||^2
        self.D = case_obj.D
        self.T = case_obj.T
        self.N = case_obj.N
        self.P_t = np.zeros(self.N)
        self.Q_t = np.zeros(self.N)
        self.P_t[-1] = self.D
        self.Q_t[-1] = 0

        self.dt = self.T/self.N

        self.r = 0
        self.V = {}
        self.poli = {}

    def create_solution(self,n_x):
        self.n_x = n_x
        self.x_upper = 2
        self.x_lower = -2
        x_space = np.linspace(-2,2, n_x)
        for n in reversed(range(self.N)):
            V_t = np.zeros(n_x)
            A_t = np.zeros(n_x)
            if (n == self.N-1):
                    P = self.P_t[n]
                    Q = self.Q_t[n]
            else:
                P_old = self.P_t[n+1]
                P = P_old + 1 - (P_old**2 / (P_old +1))
                Q_old = self.Q_t[n+1]
                Q = Q_old + self.sig * P_old
                self.Q_t[n] = Q
                self.P_t[n] =  P
           
            for i in range(n_x):
                x = x_space[i]
                  
                V_t[i] = P *x**2 + Q
                A_t[i] = -(P /(P+1)) *x
                
            self.V[n] = V_t
            self.poli[n] = A_t
        
        cum_cost = np.zeros(1000)

        for i in range(1000):
            X = np.zeros(self.N)
            X[0] = 2*np.random.rand() - 2

            c = np.zeros(self.N)
            for n in range(self.N-1):
                x_ind = self.state_to_dicrete(X[n])
                a = self.poli[n][x_ind]
                c[n] = self.f_A * np.linalg.norm(X[n])**2 + self.f_B * np.linalg.norm(a)**2

                X[n+1] = self.A * X[n] + self.B *a + self.sig * np.random.normal()
            
            cum_cost[i] = np.sum(c)
        
        return self.V, self.poli, np.mean(cum_cost)

    def state_to_dicrete(self, state):
        s_new = int(self.n_x * (state - self.x_lower) / (self.x_upper - self.x_lower)) 
        s_new = np.min([self.n_x-1, np.max([0, s_new])])

        return tuple([s_new])

    def dynamic_programming(self, n_x):
        self.n_x = n_x
        self.x_upper = 2
        self.x_lower = -2

        x_space = np.linspace(self.x_lower,self.x_upper,self.n_x)
        n_a = 80
        a_space = np.linspace(-4,4,n_a)

        V = {}
        actions = {}
        print('start DP')
        for t in reversed(range(self.N)):
            print('timestep: ', t)
            V_t = np.zeros(self.n_x)
            A_t = np.zeros(self.n_x)
            if (t == self.N-1):
                
                for i in range(self.n_x):
                    x = x_space[i]
                    V_t[i] = self.D * np.linalg.norm(x)**2
            else:
                
                
                for ix in range(self.n_x):
                    min_val = np.inf
                    min_action = 0
                    x = x_space[ix]
                    for ia in range(n_a):
                        a = a_space[ia]
                        cost = self.f_A * np.linalg.norm(x)**2 + self.f_B * np.linalg.norm(a)**2
                        
                        new_x = self.A * x + self.B *a
                        new_x_ind = self.state_to_dicrete(new_x)
                        value = cost + V[t+1][new_x_ind]
                        

                        if (value < min_val):
                            min_val = value
                            min_action = a
                    V_t[ix] = min_val
                    A_t[ix] = min_action

            V[t] = V_t
            actions[t] = A_t
        
        cum_cost = np.zeros(1000)

        for i in range(1000):
            X = np.zeros(self.N)
            X[0] = 2*np.random.rand() - 2

            c = np.zeros(self.N)
            for n in range(self.N-1):
                x_ind = self.state_to_dicrete(X[n])
                a = actions[n][x_ind]
                c[n] = self.f_A * np.linalg.norm(X[n])**2 + self.f_B * np.linalg.norm(a)**2

                X[n+1] = self.A * X[n] + self.B *a
            
            cum_cost[i] = np.sum(c)

        return V, actions, np.mean(cum_cost)