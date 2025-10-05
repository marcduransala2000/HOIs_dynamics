import numpy as np
import pandas as pd
import os
import random

H = [[0.5, 0.34, 0.76], [0.66,0.5,0.25], [0.24,0.75,0.5]] #interaction matrix

#FUNCTIONS
def F(x1,x2, f1, f2, f3):
    return f1*x1+f2*x2+f3*(1-x1-x2)
def D(x1,x2, d1, d2, d3):
    return d1*x1+d2*x2+d3*(1-x1-x2)

#DIFFERENTIAL EQUATIONS - HIGHER-ORDER AND PAIRWISE
def x_111(x1,x2,alpha, f1, f2, f3, d1, d2, d3):
    return alpha*(x1/F(x1, x2, f1, f2, f3)**3*( D(x1, x2, d1, d2, d3)*( f1*2*H[0][0]*(H[0][0]+H[0][0])*f1*x1*f1*x1 + f1*2*H[0][0]*(H[0][1]+H[0][1])*f1*x1*f2*x2 + f1*2*H[0][0]*(H[0][2]+H[0][2])*f1*x1*f3*(1-x1-x2) + f1*2*H[0][1]*(H[0][0]+H[1][0])*f2*x2*f1*x1 + f1*2*H[0][1]*(H[0][1]+H[1][1])*f2*x2*f2*x2 + f1*2*H[0][1]*(H[0][2]+H[1][2])*f2*x2*f3*(1-x1-x2) + f1*2*H[0][2]*(H[0][0]+H[2][0])*f3*(1-x1-x2)*f1*x1 + f1*2*H[0][2]*(H[0][1]+H[2][1])*f3*(1-x2-x1)*f2*x2 + f1*2*H[0][2]*(H[0][2]+H[2][2])*f3*(1-x2-x1)*f3*(1-x2-x1)  ) - d1*F(x1,x2,f1, f2, f3)**3  ) ) + (1-alpha)*(x1/F(x1, x2, f1, f2, f3)**2*( D(x1, x2, d1, d2, d3)*( f1*2*H[0][0]*f1*x1 + f1*2*H[0][1]*f2*x2 + f1*2*H[0][2]*f3*(1-x1-x2)) - d1*F(x1, x2, f1, f2, f3)**2 ) )

def x_222(x1,x2,alpha, f1, f2, f3, d1, d2, d3):
    return alpha*(x2/F(x1, x2, f1, f2, f3)**3*( D(x1, x2, d1, d2, d3)*( f2*2*H[1][0]*(H[1][0]+H[0][0])*f1*x1*f1*x1 + f2*2*H[1][0]*(H[1][1]+H[0][1])*f1*x1*f2*x2 + f2*2*H[1][0]*(H[1][2]+H[0][2])*f1*x1*f3*(1-x1-x2) + f2*2*H[1][1]*(H[1][0]+H[1][0])*f2*x2*f1*x1 + f2*2*H[1][1]*(H[1][1]+H[1][1])*f2*x2*f2*x2 + f2*2*H[1][1]*(H[1][2]+H[1][2])*f2*x2*f3*(1-x1-x2) + f2*2*H[1][2]*(H[1][0]+H[2][0])*f3*(1-x1-x2)*f1*x1 + f2*2*H[1][2]*(H[1][1]+H[2][1])*f3*(1-x2-x1)*f2*x2 + f2*2*H[1][2]*(H[1][2]+H[2][2])*f3*(1-x2-x1)*f3*(1-x2-x1)  ) - d2*F(x1,x2,f1, f2, f3)**3  ) ) + (1-alpha)*(x2/F(x1, x2, f1, f2, f3)**2*( D(x1, x2, d1, d2, d3)*( f2*2*H[1][0]*f1*x1 + f2*2*H[1][1]*f2*x2 + f2*2*H[1][2]*f3*(1-x1-x2)) - d2*F(x1, x2, f1, f2, f3)**2 ) )

#RK4 SOLVER
def rk4(nsteps, h_int, dt, x10, x20, x30,alpha,f1, f2, f3, d1, d2, d3):
    writting_period = int(dt/h_int)
        
        
    h_2 = 0.5 * h_int
    h_6 = h_int / 6
       
    data = np.zeros((4, nsteps))
        
    #INITIAL CONDITION
        
    data[0, 0] = 0 #time
    data[1, 0] = x10 
    data[2, 0] = x20 
    data[3, 0] = x30 
        
        
    x1 = x10
    x2 = x20
    x3 = x30
       
    t = 0
    
    numerical_extinction = False
    
    if save_dynamics_equal_physiological_rates == True:
        
        f = open('RK_dynamics_equal_rates_alpha_' + str(round(alpha,4)) + '.txt','w')
        f.write(str(t) + " " + str(x1) + " " + str(x2) + " " + str(x3) + "\n")
        
    if save_dynamics_different_physiological_rates == True:
        
        f = open('RK_dynamics_different_rates_alpha_' + str(round(alpha,4)) + '.txt','w')
        f.write(str(t) + " " + str(x1) + " " + str(x2) + " " + str(x3) + "\n")
        
    
    for i in range(nsteps):
            
        for j in range(writting_period):
            
            k_11 = x_111(x1,x2,alpha,f1, f2, f3, d1, d2, d3) 
            k_12 = x_222(x1,x2,alpha,f1, f2, f3, d1, d2, d3)
        
            k_21 = x_111( (x1 + h_2 * k_11), (x2 + h_2 * k_12),alpha,f1, f2, f3, d1, d2, d3)
            k_22 = x_222( (x1 + h_2 * k_11), (x2 + h_2 * k_12),alpha,f1, f2, f3, d1, d2, d3)
                
            k_31 = x_111( (x1 + h_2 * k_21), (x2 + h_2 * k_22),alpha,f1, f2, f3, d1, d2, d3)
            k_32 = x_222( (x1 + h_2 * k_21), (x2 + h_2 * k_22),alpha,f1, f2, f3, d1, d2, d3)
                
            k_41 = x_111( (x1 + h_int * k_31), (x2 + h_int * k_32),alpha,f1, f2, f3, d1, d2, d3)
            k_42 = x_222( (x1 + h_int * k_31), (x2 + h_int * k_32),alpha,f1, f2, f3, d1, d2, d3)
    
            x1 += h_6 * (k_11 + 2 * k_21 + 2 * k_31 + k_41)
            x2 += h_6 * (k_12 + 2 * k_22 + 2 * k_32 + k_42)
            x3 = 1 - x1 - x2
                
            #print(x1,x2,x3)
                    
            t += h_int
            
        if save_dynamics_equal_physiological_rates == True or save_dynamics_different_physiological_rates == True:
            
            f.write(str(t) + " " + str(x1) + " " + str(x2) + " " + str(x3) + "\n")
            
        data[0, i]= t
        data[1, i] = x1
        data[2, i] = x2
        data[3, i] = x3
        
        # BREAK LOOP IF EXPLOTES       
        if x1 < 1e-12 or x2 < 1e-12 or x3 < 1e-12:
            numerical_extinction = True
            break

            
    f.close()
        
    return data, numerical_extinction


def rk4_alpha(nsteps, h_int, dt, x10, x20, x30, ALPHAS, f1, f2, f3, d1, d2, d3):
    AC = np.nan 
    for alpha in ALPHAS:
        #print(alpha)
        
        data, numerical_extinction = rk4(nsteps, h_int, dt, x10, x20, x30, alpha, f1, f2, f3, d1, d2, d3)
        
        
        if numerical_extinction == True:
            continue
        
        if np.isnan(data).any():
            continue

        final_x1, final_x2, final_x3 = data[1, -1], data[2, -1], data[3, -1]
        if final_x1 < 1e-3 or final_x2 < 1e-3 or final_x3 < 1e-3:
            continue
        
        Af = data[1, nsteps-int(nsteps/5):-1]  # 20% of latest points
        Ai = data[1, int(nsteps/5) : int(nsteps/5) + int(nsteps/5)]  # 20% to 40% of earliest points
        
        Af = (max(Af) - min(Af)) / 2  # amplitude of oscillations
        Ai = (max(Ai) - min(Ai)) / 2
        #print(Af, Ai)
        
        amplitude_decrease = (Af + 0.005) - Ai
        std_last = np.std(data[1, nsteps - int(nsteps / 10): -1])
        
        if amplitude_decrease < 0 or std_last < 0.01:  # final amplitude is smaller than initial or last 10% points are cte --> transition occurs
            AC = alpha
            break
    
    return AC

def f(std,mean):
    return np.random.normal(mean,std)
def d(std,mean):
    return np.random.normal(mean,std)

def run_simulation_with_gaussian_sampling(nsteps, h_int, dt, ALPHAS, num_trials, std_list, mean_f_d=1.0):
    results = []
    for std in std_list:
        
        num_trials_done = 0  # Track the number of accepted trials for this std
        
        while num_trials_done < num_trials:
            
            # Generate f1, f2, f3, d1, d2, d3 from Gaussian distributions
            f1 = f(std, mean_f_d)
            f2 = f(std, mean_f_d)
            f3 = f(std, mean_f_d)
            d1 = d(std, mean_f_d)
            d2 = d(std, mean_f_d)
            d3 = d(std, mean_f_d)
            #print(f1,f2,f3,d1,d2,d3)

            # Normalize f and d by the maximum value to avoid numerical issues
            max_fd = max(f1, f2, f3, d1, d2, d3)
            f1_norm = f1 / max_fd
            f2_norm = f2 / max_fd
            f3_norm = f3 / max_fd
            d1_norm = d1 / max_fd
            d2_norm = d2 / max_fd
            d3_norm = d3 / max_fd
            
            if any(value <= 0 for value in [f1, f2, f3, d1, d2, d3]):
                # Skip this trial and continue generating new values
                continue

            # Mean rate for adjusting nsteps if necessary
            mean_rate = (f1_norm + f2_norm + f3_norm + d1_norm + d2_norm + d3_norm) / 6
            adjusted_nsteps = int(nsteps / mean_rate)

            x10, x20, x30 = 0.32, 0.21, 0.47

            alpha_c = rk4_alpha(adjusted_nsteps, h_int, dt, x10, x20, x30, ALPHAS,
                                f1_norm, f2_norm, f3_norm, d1_norm, d2_norm, d3_norm)

            p_c_i = 1 if not np.isnan(alpha_c) else 0

            # Calculate std of f and d
            std_f = np.std([f1, f2, f3])
            std_d = np.std([d1, d2, d3])

            # Calculate ratios_C and their std
            ratios_C = [f1 / d1, f2 / d2, f3 / d3]
            std_C = np.std(ratios_C)

            results.append({
                'std_gaussian': std,
                'f': [f1, f2, f3],
                'std_f': std_f,
                'd': [d1, d2, d3],
                'std_d': std_d,
                'ratios_C': ratios_C,
                'std_ratios_C': std_C,
                'alpha_c': alpha_c,
                'p_c_i': p_c_i # 1 if there is coexistence, otherwise 0
            })
            num_trials_done += 1
            
    return results

def run_simulation_with_real_data(plant_data, nsteps, h_int, dt, ALPHAS, num_trials):
    results = []
    used_combinations = set()
    while len(results) < num_trials:

        indices = tuple(sorted(random.sample(range(len(plant_data)), 3)))
        #print(indices)
        if indices in used_combinations:
            continue
        used_combinations.add(indices)
        species_data_1 = plant_data.iloc[indices[0]]
        species_data_2 = plant_data.iloc[indices[1]]
        species_data_3 = plant_data.iloc[indices[2]]
        #print(species_data_1)
        #print(species_data_2)
        #print(species_data_3)
        max_fd = max(species_data_1[2], species_data_2[2], species_data_3[2],species_data_1[3], species_data_2[3], species_data_3[3])
        F1, D1 = species_data_1[2], species_data_1[3]
        F2, D2 = species_data_2[2], species_data_2[3]
        F3, D3 = species_data_3[2], species_data_3[3]
        f1, d1 = species_data_1[2] / max_fd, species_data_1[3] / max_fd 
        f2, d2 = species_data_2[2] / max_fd, species_data_2[3] / max_fd
        f3, d3 = species_data_3[2] / max_fd, species_data_3[3] / max_fd
        #print(f1, d1, f2, d2, f3, d3)
        
        mean_rate = (f1 + f2 + f3 + d1 + d2 + d3) / 6
        #print('mean rate', mean_rate)
        
        adjusted_nsteps = int(nsteps / mean_rate)
        #print('adjusted nsteps', adjusted_nsteps)
        
        alpha_c = rk4_alpha(adjusted_nsteps, h_int, dt, x10, x20, x30, ALPHAS, f1, f2, f3, d1, d2, d3)
        #print(alpha_c)
        p_c_i = 1 if not np.isnan(alpha_c) else 0
        
        # Calculate std of f and d separately
        std_f = np.std([F1, F2, F3])
        std_d = np.std([D1, D2, D3])
        
        # Calculate std and proportion of each pair (f_i, d_i)
        ratios_C = [F1/D1, F2/D2, F3/D3]
        std_C = np.std([F1 / D1, F2 / D2, F3 / D3])
        
        results.append({
            'indices': indices,
            'f': [F1, F2, F3],
            'std_f': std_f,
            'd': [D1, D2, D3],
            'std_d': std_d,
            'ratios_C': ratios_C,           
            'std_ratios_C': std_C,
            'alpha_c': alpha_c,
            'p_c_i': p_c_i # 1 if there is coexistence, otherwise 0
        })
        
    return results

# ------------------------------------------------------------------------------------------------

# OPTIONS
# 1- RUN ONLY THE RK4 DYNAMICS for a specific fraction of HOIs (alpha) - EQUAL PHYSIOLOGICAL RATES 
# 2- RUN ONLY THE RK4 DYNAMICS for a specific fraction of HOIs (alpha) - DIFFERENT PHYSIOLOGICAL RATES
# 3- RUN THE MAIN CODE FOR GAUSSIAN DATA
# 4- RUN THE MAIN CODE FOR REAL DATA

option = 1

# ------------------------------------------------------------------------------------------------

# RUN ONLY THE RK4 DYNAMICS for a specific HOIs fraction of interactions (alpha) - EQUAL PHYSIOLOGICAL RATES 

if option == 1:
    save_dynamics_equal_physiological_rates = True
    save_dynamics_different_physiological_rates = False
    
    nsteps= 20000
    h_int=0.1
    dt=0.1
    x10 = 0.32
    x20 = 0.21
    x30 = 0.47
    alpha = 0.1
    f1, f2, f3, d1, d2, d3 = 1,1,1,1,1,1
    rk4(nsteps, h_int, dt, x10, x20, x30,alpha,f1, f2, f3, d1, d2, d3)


# ------------------------------------------------------------------------------------------------

# RUN ONLY THE RK4 DYNAMICS for a specific HOIs fraction of interactions (alpha) - DIFFERENT PHYSIOLOGICAL RATES

if option == 2:
    save_dynamics_equal_physiological_rates = False
    save_dynamics_different_physiological_rates = True
    
    nsteps= 20000
    h_int=0.1
    dt=0.1
    x10 = 0.32
    x20 = 0.21
    x30 = 0.47
    alpha = 0.1
    f1, f2, f3, d1, d2, d3 = 0.96,1.04,0.99,0.97,0.94,1.09
    rk4(nsteps, h_int, dt, x10, x20, x30,alpha,f1, f2, f3, d1, d2, d3)


# ------------------------------------------------------------------------------------------------

# RUN THE MAIN CODE FOR GAUSSIAN DATA

if option == 3:
    save_dynamics_different_physiological_rates = False
    save_dynamics_equal_physiological_rates = False
    
    nsteps = 5000
    h_int = 1
    dt = 1
    ALPHAS = np.linspace(0, 1.0, 81)
    num_trials_per_std = 10000  # Number of trials per std
    std_list = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.75, 1]
    mean_f_d = 1.0  # Mean value for f and d
    
    # Run simulation
    results = run_simulation_with_gaussian_sampling(
        nsteps, h_int, dt, ALPHAS,
        num_trials_per_std, std_list, mean_f_d
    )
    
    # Convert results to DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('simulation_results_gaussian_sampling.csv', index=False)


# ------------------------------------------------------------------------------------------------

# RUN THE MAIN CODE FOR REAL DATA

if option == 4:
    os.chdir(r'C:/Users/mduran/Desktop/MSc/TFM\RK/real_test')
    
    save_dynamics_different_physiological_rates = False
    save_dynamics_equal_physiological_rates = False
    
    nsteps = 5000
    h_int = 1
    dt = 1
    ALPHAS = np.linspace(0, 1, 81)
    num_trials = 15000
    x10, x20, x30 = 0.32, 0.21, 0.47
    
    plant_types = ['Seagrasses', 'Trees', 'LandHerbsSaltMarsh']
    
    # Ejecutar simulaciones para cada tipo de planta
    for plant_type in plant_types:
        plant_data = pd.read_csv('dataFD_' + plant_type + '.csv', delimiter=',', skiprows=1, header=None)
        #print(max(plant_data[3]))
        results = run_simulation_with_real_data(plant_data, nsteps, h_int, dt, ALPHAS, num_trials)
        
        results_df = pd.DataFrame(results)
        
        results_df.to_csv(plant_type+'_simulation_results.csv', index=False)
        alpha_c_values = results_df['alpha_c']
        p_c_i_values = results_df['p_c_i']
        
        avg_alpha_c = alpha_c_values[alpha_c_values.notna()].mean()
        alpha_c_std = alpha_c_values[alpha_c_values.notna()].std()
        p_c = p_c_i_values.sum()/len(p_c_i_values)
    
        l = open(plant_type+'results.txt','w')
        l.write(str(avg_alpha_c) + ' ' + str(alpha_c_std) + ' ' + str(p_c))
        l.close()
      
