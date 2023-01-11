import numpy as np
import matplotlib.pyplot as plt


class Solution_2_D():

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
        self.P_t = np.zeros((self.N,2,2))
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
        y_space = np.linspace(-2,2, n_x)
        optimal_cost = 0
        for n in reversed(range(self.N)):
            V_t = np.zeros((n_x,n_x))
            A_t = np.zeros((n_x,n_x,2))

            if (n == self.N-1):
                    P = self.P_t[n]
                    Q = self.Q_t[n]
                    P_old = P
            else:
                P_old = self.P_t[n+1]

                inv = np.linalg.inv(np.identity(2) + P_old)
                y = np.dot(P_old, inv)
                P = P_old + np.identity(2) - np.dot(y, P_old)

                Q_old = self.Q_t[n+1]
                optimal_cost += np.trace(np.dot((self.sig**2)*np.identity(2), P_old))
                Q = Q_old + np.trace(np.dot((self.sig**2)*np.identity(2), P_old))
                self.Q_t[n] = Q
                self.P_t[n] =  P

            for i in range(n_x):
                for j in range(n_x):
                    x = x_space[i]
                    y = y_space[j]
                    z = np.array([x,y])
                    x_bar = np.dot(np.transpose(z),self.P_t[n])
                   
                    V_t[i][j] = np.dot(x_bar,z) + Q
                   
                    inv = np.linalg.inv(np.identity(2) + P_old)
                    
                    y_bar = np.dot(P_old,z)
                    
                    A_t[i][j] = -np.dot(inv, y_bar)
                
            self.V[n] = V_t
            self.poli[n] = A_t
        
        # fig, ax = plt.subplots(2,subplot_kw={"projection": "3d"})
        # X,Y = np.meshgrid(x_space, y_space)
        # ax[0].plot_surface(X,Y,self.V[self.N-1])
        # ax[1].plot_surface(X,Y,self.V[0])
        # plt.show()

        cum_cost = np.zeros(1000)

        for i in range(1000):
            X = np.zeros((self.N, 2))
            X[0] = 2*np.random.rand(2) - 2
            
            c = np.zeros(self.N)
            for n in range(self.N):
                X[n] = np.clip(X[n], a_min= -2,a_max = 2)
                
                
                if (n== self.N-1):
                    y = np.dot(np.transpose(X[n]), self.D)
                    c[n] = np.dot(y,X[n])
                else:
                    #a = self.poli[n][ind]
                    inv = np.linalg.inv(np.identity(2) + self.P_t[n+1])
                    
                    y_bar = np.dot(self.P_t[n+1],X[n])
                    
                    a = -np.dot(inv, y_bar)

                    y1= np.dot(np.transpose(X[n]), self.f_A)
                    
                    y2 = np.dot(np.transpose(a), self.f_B)
                    
                    c[n] = np.dot(y1,X[n]) + np.dot(y2,a)
                    
                    X[n+1] = X[n] + a + self.sig*np.random.normal(2)
                #X[n+1] = (1 + self.A * self.dt)* X[n] + self.B * self.dt* a + np.sqrt(self.sig * self.dt)*np.random.normal()
            cum_cost[i] = np.sum(c)
        
        print('optimal_cost', optimal_cost)
        print(np.mean(cum_cost))
        return self.V, self.poli, np.mean(cum_cost)

    def state_to_dicrete(self, state):
        x = state[0]
        y = state[1]
        x_new = int(self.n_x * (x - self.x_lower) / (self.x_upper - self.x_lower)) 
        x_new = np.min([self.n_x-1, np.max([0, x_new])])

        y_new = int(self.n_x * (y - self.x_lower) / (self.x_upper - self.x_lower)) 
        y_new = np.min([self.n_x-1, np.max([0, y_new])])

        return tuple([x_new, y_new])

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

                X[n+1] = np.dot(self.A,X[n]) + np.dot(self.B,a) +  np.random.normal(2)
            
            cum_cost[i] = np.sum(c)

        return V, actions, np.mean(cum_cost)

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
            for n in range(self.N):

                x_ind = self.state_to_dicrete(X[n])
                if (n== self.N-1):
                    c[n] = self.D * np.linalg.norm(X[n])**2
                else:
                    a = self.poli[n][x_ind]
                    c[n] = self.f_A * np.linalg.norm(X[n])**2 + self.f_B * np.linalg.norm(a)**2

                    X[n+1] = self.A * X[n] + self.B *a + self.sig * np.random.normal()
                #X[n+1] = (1 + self.A * self.dt)* X[n] + self.B * self.dt* a + np.sqrt(self.sig * self.dt)*np.random.normal()
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
                        
                        new_x = self.A * x + self.B *a +  np.random.normal(2)
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

                X[n+1] = self.A * X[n] + self.B *a + self.sig * np.random.normal()
            
            cum_cost[i] = np.sum(c)

        return V, actions, np.mean(cum_cost)

# Case = lqr_2D_TD3.CaseOne()

# sol = Solution_2_D(Case)
# sol.create_solution(20)