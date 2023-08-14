import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# file names

num_runs = 1
iteration_steps = 100

#\committor\avg_reward_step_0_run_0.npy

max_P_mean_total = 0
max_P_stddev_total = 0


max_V_mean_total = 0

max_V_stddev_total = 0


min_P_mean_total= 0
min_P_stddev_total = 0


min_V_mean_total = 0
min_V_mean_total = 0



for i in np.arange(3900,4000,100):
    X = np.load('./committor/X_space.npy')
    Y = np.load('./committor/Y_space.npy')

    X2 = np.load('./committor/X_space_VF.npy')
    Y2 = np.load('./committor/Y_space_VF.npy')

    
    n_x = 40

    

    rewards = np.zeros((num_runs,i))
    stopping = np.zeros((num_runs,i))

    bases = np.zeros(num_runs)
    V1 = np.zeros((num_runs,n_x,n_x))
    V2 = np.zeros((num_runs,n_x,n_x))

   
    

    P_x = np.zeros((num_runs,n_x,n_x))
    P_y = np.zeros((num_runs,n_x,n_x))


    t0 = 0
    t1 = 19

    V_true = np.load('./committor/true_value_fct.npy')
   

    P_x_true = np.load('./committor/true_policy_x.npy')
    P_y_true = np.load('./committor/true_policy_y.npy')
    
    true_f_e = np.load('./committor/true_free_energy.npy')
    n_e = len(true_f_e)
    x_space = np.linspace(1,3 ,n_e)

    f_e1 = np.zeros((num_runs,n_e))
    f_e2 = np.zeros((num_runs,n_e))
    

    true_x_VF = np.load('./committor/true_policy_x_VF.npy')
    true_y_VF = np.load('./committor/true_policy_y_VF.npy')
    P_x_VF = np.zeros((num_runs, 30,10))
    P_y_VF = np.zeros((num_runs, 30,10))
    

    fill_axis = np.linspace(0,i,i)
    for r in range(num_runs):
        print(r)
        rewards[r] = np.load('./committor/avg_reward_step_{}_run_{}.npy'.format(i,r))
        stopping[r] = np.load('./committor/avg_stopping_step_{}_run_{}.npy'.format(i,r))

        bases[r] = np.load('./committor/base_run_{}.npy'.format(r))

        V1[r] = np.load('./committor/value_fct_1__{}_run_{}.npy'.format(i,r))
        V2[r] = np.load('./committor/value_fct_2_{}_run_{}.npy'.format(i,r))

        
        P_x[r] = np.load('./committor/policy_X_{}_run_{}.npy'.format(i,r))
        P_y[r] = np.load('./committor/policy_Y_{}_run_{}.npy'.format(i,r))

        f_e1[r] = np.load('./committor/free_energy_approx_1_{}_run_{}.npy'.format(i,r))
        f_e2[r] = np.load('./committor/free_energy_approx_2_{}_run_{}.npy'.format(i,r))

        P_x_VF[r] = np.load('./committor/policy_X_{}_run_{}_VF.npy'.format(i,r))
        P_y_VF[r] = np.load('./committor/policy_Y_{}_run_{}_VF.npy'.format(i,r))

    reward_mean = np.mean(rewards, axis=0)
    reward_stddev = np.std(rewards, axis = 0)

    stopping_mean = np.mean(stopping, axis=0)
    stopping_stddev = np.std(stopping, axis = 0)
    
    base_mean = np.mean(bases)
    base_stddev = np.std(bases)

    V1_mean = np.mean(V1,axis=0)
    V2_mean = np.mean(V2,axis=0)
    
    max_V_mean = np.max([V1_mean,V2_mean])

    V1_stddev = np.std(V1,axis=0)
    V2_stddev = np.std(V2,axis=0)
  
    max_V_stddev = np.max([V1_stddev,V2_stddev])

    f_e1_mean = np.mean(f_e1, 0)
    f_e2_mean = np.mean(f_e2, 0)

    max_f_e_mean = np.max([f_e1_mean,f_e1_mean])

    P_x_VF_mean = np.mean(P_x_VF, 0)
    P_y_VF_mean = np.mean(P_y_VF, 0)

    max_P_VF_mean = np.max([P_x_VF_mean, P_y_VF_mean])
    


    f_e1_stddev = np.std(f_e1, 0)
    f_e2_stddev = np.std(f_e2, 0)

    max_f_e_stddev = np.max([f_e1_stddev,f_e1_stddev])


    P_x_mean = np.mean(P_x,axis=0)
    P_y_mean = np.mean(P_y,axis=0)
   

    max_P_mean = np.max([P_x_mean,P_y_mean,P_x_mean])

    P_x_stddev = np.std(P_x,axis=0)
    P_y_stddev = np.std(P_y,axis=0)
   

    max_P_stddev = np.max([P_y_stddev,P_x_stddev,P_y_stddev])

    max_P_mean_total = np.max([max_P_mean_total, np.max(P_x_mean),np.max(P_y_mean)])
    max_P_stddev_total = np.max([max_P_stddev_total,np.max(P_x_stddev), np.max(P_y_stddev)])
   
   

    max_V_mean_total = np.max([max_V_mean_total, np.max(V1_mean), np.max(V2_mean)])
    max_V_stddev_total = np.max([max_V_stddev_total, np.max(V1_stddev), np.max(V2_stddev)])
   

    min_P_mean_total = np.min([min_P_mean_total, np.min(P_x_mean),np.min(P_y_mean)])
    min_P_stddev_total= np.min([min_P_stddev_total,np.min(P_x_stddev), np.min(P_y_stddev)])
    
   

    min_V_mean_total = np.min([min_V_mean_total, np.min(V1_mean), np.min(V2_mean)])
    min_V_mean_total = np.min([min_V_mean_total, np.min(V1_mean), np.min(V2_mean)])
    

    fig1 = plt.figure()
    #fig1.set_tight_layout('tight')
    #fig1.set_figheight(5)
    #fig1.set_figwidth(10)

    ax1_1 = fig1.add_subplot()
    
    
    ax1_1.set_title('average cost over 500 episodes', fontsize = 9, fontweight='bold')
    
    ax1_1.set_xlim([0,5000])
    #ax1.set_ylim([0, np.max([np.max(base_mean), np.max(reward_mean)])+5])

   
    ax1_1.set_xlabel('episodes')
    ax1_1.set_ylabel('average cost')

    if(len(reward_mean)== 0):
        ax1_1.plot(reward_mean, label = 'mean: ', color='dodgerblue')
        ax1_1.fill_between(fill_axis,reward_mean,reward_mean + reward_stddev, alpha = 0.2, color='dodgerblue', label = 'standard deviation: ')
        ax1_1.fill_between(fill_axis,reward_mean,reward_mean - reward_stddev, alpha = 0.2, color='dodgerblue')
    else:    
        ax1_1.plot(reward_mean, label = 'mean: {}'.format(np.round(reward_mean[-1],2)), color='dodgerblue')
        ax1_1.fill_between(fill_axis,reward_mean,reward_mean + reward_stddev, alpha = 0.2, color='dodgerblue', label = 'standard deviation: {}'.format(np.round(reward_stddev[-1],2)))
        ax1_1.fill_between(fill_axis,reward_mean,reward_mean - reward_stddev, alpha = 0.2, color='dodgerblue')
    ax1_1.set_ylim([0, 15])
    ax1_1.legend()
   
    # ax1_2 = fig1.add_subplot(1,2,2)
    
    # ax1_2.set_title('average stopping time over 500 episodes', fontsize = 9, fontweight='bold')
    
    # ax1_2.set_xlim([0,5000])
    # #ax1.set_ylim([0, np.max([np.max(base_mean), np.max(reward_mean)])+5])

    # ax1_2.set_ylim([0, np.max([0, np.max(stopping_mean)])+0.2])
    # ax1_2.set_xlabel('episodes')
    # ax1_2.set_ylabel('average stopping time')

    # ax1_2.plot(stopping_mean, label = 'mean: {}'.format(np.round(stopping_mean[-1],2)),color='dodgerblue')
    # ax1_2.fill_between(fill_axis,stopping_mean,stopping_mean + stopping_stddev, alpha = 0.2, color='dodgerblue', label = 'standard deviation: {}'.format(np.round(stopping_stddev[-1],2)))
    # ax1_2.fill_between(fill_axis,stopping_mean,stopping_mean - stopping_stddev, alpha = 0.2, color='dodgerblue')
    # ax1_2.legend()
    #ax1.hlines(base_mean, xmin = 0, xmax = 5000, color = 'black')
    #ax1.fill_between([0,5000],[base_mean -base_stddev,base_mean-base_stddev], [base_mean +base_stddev,base_mean + base_stddev], alpha = 0.2, color='black')
    

    fig2 =plt.figure()
    fig2.suptitle('Epsiode: {}'.format(i))
    gs2 = fig2.add_gridspec(2,2 , height_ratios = [1.5,1])
    fig2.set_tight_layout('tight')
    fig2.set_figheight(7)
    fig2.set_figwidth(9)
   
    
    ax2_1 = fig2.add_subplot(gs2[0,0] ,projection='3d')
    ax2_1.view_init(elev=15, azim=-70, roll=0)

    
    ax2_1.plot_surface(X,Y, V_true, color= 'black', alpha = 0.2)
    # ax2_1.plot_surface(X,Y, V1_mean, color= 'dodgerblue')
    # ax2_1.plot_surface(X,Y, V2_mean, color= 'coral')
    ax2_1.plot_surface(X,Y, 0.5*(V1_mean + V2_mean), color= 'dodgerblue')
    ax2_1.set_title('value function', fontsize = 9, fontweight='bold')
    

    ax2_1.set_zlim([-1.5, 8])

    ax2_2 = fig2.add_subplot(gs2[0,1])
    ax2_2.plot(x_space, true_f_e,color= 'black', alpha = 0.2, label = '$V^\epsilon(x) = -log(h(0,y) + \epsilon)$')
    # ax2_2.plot(x_space, f_e1_mean, color= 'dodgerblue')
    # ax2_2.plot(x_space, f_e2_mean, color= 'coral')
    ax2_2.plot(x_space, 0.5*(f_e1_mean+f_e2_mean), color= 'dodgerblue', label = 'mean value function $V(0,y)$')
    ax2_2.fill_between(x_space ,0.5*(f_e1_mean+f_e2_mean),0.5*(f_e1_mean+f_e2_mean) - 0.5*( f_e1_stddev +  f_e2_stddev), alpha = 0.2, color='dodgerblue', label = 'standard deviation')
    ax2_2.fill_between(x_space ,0.5*(f_e1_mean+f_e2_mean),0.5*(f_e1_mean+f_e2_mean) + 0.5*( f_e1_stddev +  f_e2_stddev), alpha = 0.2, color='dodgerblue')
    ax2_2.set_title('value function x = 0', fontsize = 9, fontweight='bold')
    ax2_2.set_ylim([-1.5,8])
    ax2_2.legend()
    
    ax2_3 = fig2.add_subplot(gs2[1,0])
    ax2_3.set_aspect('equal')
    #ax2_2.pcolormesh(X,Y, V1_t0_stddev,vmin=0, vmax=max_V_stddev)
    #ax2_2.pcolormesh(X,Y, V2_t0_stddev,vmin=0, vmax=max_V_stddev)
    ax2_3.set_title('standard deviation', fontsize = 9, fontweight='bold')
    cont2_3 = ax2_3.pcolormesh(X,Y, 0.5*(V1_stddev + V2_stddev),vmin=0, vmax=3)

    c_bar3 = fig2.colorbar(cont2_3, orientation='vertical')
    
    ax2_4 = fig2.add_subplot(gs2[1,1])
    # ax2_4.plot(x_space, f_e1_stddev, color= 'dodgerblue')
    # ax2_4.plot(x_space, f_e2_stddev, color= 'coral')
    ax2_4.plot(x_space,0.5*( f_e1_stddev +  f_e2_stddev), color= 'dodgerblue')
    ax2_4.set_title('standard deviation x = 0', fontsize = 9, fontweight='bold')
    ax2_4.set_ylim([0,2])

    fig3 =plt.figure()
    fig3.suptitle('Epsiode: {}'.format(i))
    gs3 = fig2.add_gridspec(2,2 , height_ratios = [1.5,1])
    fig3.set_tight_layout('tight')
    fig3.set_figheight(6)
    fig3.set_figwidth(7)
   

    ax3_1 = fig3.add_subplot(gs3[0,0],projection='3d')
    ax3_1.view_init(elev=15, azim=-70, roll=0)

   
    ax3_1.plot_surface(X,Y, np.clip(P_x_true, a_min=-5, a_max=5), color= 'black', alpha = 0.2, label="$u^*(x) = -\sigma \nabla V^\epsilon(x)$")
    ax3_1.plot_surface(X,Y, P_x_mean, color= 'dodgerblue')
    #ax3_1.set_zlim([-4.1,4.1])
    ax3_1.set_title('x component policy', fontsize = 9, fontweight='bold')
    ax3_1.set_zlim([-5,5])

    ax3_2 = fig3.add_subplot(gs3[0,1],projection='3d')
    ax3_2.view_init(elev=15, azim=-70, roll=0)
    ax3_2.plot_surface(X,Y, np.clip(P_y_true, a_min=-5, a_max=5), color= 'black', alpha = 0.2, label="$u^*(x) = -\sigma \nabla V^\epsilon(x)$")
    ax3_2.plot_surface(X,Y, P_y_mean, color= 'dodgerblue')

    #ax3_2.set_zlim([-4.1,4.1])
    ax3_2.set_title('y component policy', fontsize = 9, fontweight='bold')

    ax3_2.set_zlim([-5,5])
    
    

    ax3_3 = fig3.add_subplot(gs3[1,0])
    ax3_3.set_aspect('equal')
    ax3_3.pcolormesh(X,Y,  P_x_stddev,vmin=0, vmax=4)
    ax3_3.set_title('standard deviation', fontsize = 9, fontweight='bold')
   
   
    ax3_4 = fig3.add_subplot(gs3[1,1])
    ax3_4.set_aspect('equal')
    cont4 = ax3_4.pcolormesh(X,Y,  P_y_stddev,vmin=0, vmax=3)
    ax3_4.set_title('standard deviation', fontsize = 9, fontweight='bold')

  
    fig3.colorbar(cont4, orientation='vertical')
    
    fig4 = plt.figure()
    fig4.suptitle('Epsiode: {}'.format(i))
    fig4.tight_layout()
    fig4.set_figheight(6)
    fig4.set_figwidth(6)
    ax4 = fig4.add_subplot()
    ax4.set_title('vector field of policy', fontsize = 9, fontweight='bold')
    ax4.quiver(X2,Y2,P_x_VF_mean, P_y_VF_mean, color= 'dodgerblue')
    ax4.quiver(X2,Y2, true_x_VF, true_y_VF, color = 'black', alpha = 0.4)
    time = np.linspace(0,2*np.pi,100)
    circ_x_1 = 1*np.cos(time)
    circ_y_1 = 1*np.sin(time)
    ax4.plot(circ_x_1, circ_y_1, color = 'black')
    

    circ_x_2 = 3*np.cos(time)
    circ_y_2 = 3*np.sin(time)
    ax4.plot(circ_x_2, circ_y_2, color = 'black')
    
    
    # fig1.savefig('./committor/images/reward_epi_{}'.format(i))
    # fig2.savefig('./committor/images/value_fct_epi_{}'.format(i))
    # fig3.savefig('./committor/images/policy_epi_{}'.format(i))
    # fig4.savefig('./committor/images/VF_epi_{}'.format(i))
    plt.show()
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)
    plt.close(fig4)
    

# max_P_mean_total_t0,max_P_stddev_total_t0,max_P_mean_total_t1,max_P_stddev_total_t1,max_V_mean_total_t0,max_V_mean_total_t1,max_V_stddev_total_t0,max_V_stddev_total_t1
# 4.043304014205932, 2.935147692749849, 3.703972911834717, 1.4305743895184417, 24.30153121948242, 19.183624267578125, 1.5091561974096852, 1.24356255476597
print('max_P_mean_total,max_P_stddev_total,max_V_mean_total,max_V_stddev_total', max_P_mean_total,max_P_stddev_total,max_V_mean_total,max_V_stddev_total)

# min_P_mean_total_t0,min_P_stddev_total_t0,min_P_mean_total_t1,min_P_stddev_total_t1,min_V_mean_total_t0,min_V_mean_total_t1,min_V_stddev_total_t0,min_V_stddev_total_t1
#  -4.05456805229187 0.0 -4.011117887496948 0.0 0.0 0.0 0.0 0.0
print('min_P_mean_total,min_P_stddev_total,min_V_mean_total,min_V_mean_total', min_P_mean_total,min_P_stddev_total,min_V_mean_total,min_V_mean_total)
