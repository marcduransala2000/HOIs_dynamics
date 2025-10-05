import numpy as np
import matplotlib.pyplot as plt


save_dynamics_equal_physiological_rates = True
save_dynamics_different_physiological_rates = True


# ------------------------------------------------------------------------------------------------

#EQUAL RATES
if save_dynamics_equal_physiological_rates == True:
    
    alpha = 0.1
    
    data = np.loadtxt('RK_dynamics_equal_rates_alpha_' + str(round(alpha,4)) + '.txt')

    t = data[:, 0]
    x1 = data[:, 1]
    x2 = data[:, 2]
    x3 = data[:, 3]
    
    plt.plot(t, x1, 'm')
    plt.plot(t, x2, 'b')
    plt.plot(t, x3, 'y')
    
    plt.ylim(0,1)
    plt.yticks([0,0.5,1])
    plt.tick_params(axis='x', labelsize=18) 
    plt.tick_params(axis='y', labelsize=18)
    plt.xlabel(r"$t$", fontsize=25)
    plt.ylabel(r"$x_i(t)$", fontsize=25)
    
    plt.savefig('RK_dynamics_equal_rates_alpha_' + str(round(alpha,4)) + '.svg', dpi=600, bbox_inches='tight')
    plt.show()
    
    
# ------------------------------------------------------------------------------------------------

#DIFFERENT RATES'
if save_dynamics_different_physiological_rates == True:
    
    alpha = 0.1

    data = np.loadtxt('RK_dynamics_different_rates_alpha_' + str(round(alpha,4)) + '.txt')
    
    t = data[:, 0]
    x1 = data[:, 1]
    x2 = data[:, 2]
    x3 = data[:, 3]
    
    plt.plot(t, x1, 'm')
    plt.plot(t, x2, 'b')
    plt.plot(t, x3, 'y')
    
    plt.ylim(0,1)
    plt.yticks([0,0.5,1])
    plt.tick_params(axis='x', labelsize=18) 
    plt.tick_params(axis='y', labelsize=18)
    plt.xlabel(r"$t$", fontsize=25)
    plt.ylabel(r"$x_i(t)$", fontsize=25)
    
    plt.savefig('RK_dynamics_different_rates_alpha_' + str(round(alpha,4)) + '.svg', dpi=600, bbox_inches='tight')
    plt.show()
