"""
MiniBooNE Excess Analysis - Oscillation Models - Chi2 Analysis - Visualisation
Basic tasks:
1- Import the chi2 analyis rawdata
2- Compute the best fit point
3- Plots contour Chi2 goodness of fit for each model
4- Compare the best fit for models
"""

#
# Set enviroment and definitions
#

# Import basic python packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import sys

#
# Import MiniBooNE data
#

# Baseline
L_nu = 541

# Read data from csv file

DR18RF_SO_Nu = pd.read_csv("analysis/MBEA_DR18_OM_chi2_analysis_RF_rawdata_SO_Nu.csv", sep=',')
DR18RF_SO_aNu = pd.read_csv("analysis/MBEA_DR18_OM_chi2_analysis_RF_rawdata_SO_aNu.csv", sep=',')
DR18RF_PHHO_Nu = pd.read_csv("analysis/MBEA_DR18_OM_chi2_analysis_RF_rawdata_PHHO_Nu.csv", sep=',')
DR18RF_PHHO_aNu = pd.read_csv("analysis/MBEA_DR18_OM_chi2_analysis_RF_rawdata_PHHO_aNu.csv", sep=',')

DR20RF_SO_Nu = pd.read_csv("analysis/MBEA_DR20_OM_chi2_analysis_RF_rawdata_SO_Nu.csv", sep=',')
DR20RF_SO_aNu = pd.read_csv("analysis/MBEA_DR20_OM_chi2_analysis_RF_rawdata_SO_aNu.csv", sep=',')
DR20RF_PHHO_Nu = pd.read_csv("analysis/MBEA_DR20_OM_chi2_analysis_RF_rawdata_PHHO_Nu.csv", sep=',')
DR20RF_PHHO_aNu = pd.read_csv("analysis/MBEA_DR20_OM_chi2_analysis_RF_rawdata_PHHO_aNu.csv", sep=',')

# Import data

DR18 = pd.read_csv("data/MB_DR2018.csv", sep=',') # Data Release 2018
DR20 = pd.read_csv("data/MB_DR2020.csv", sep=',') # Data Release 2020

E_bin_dr18 = np.array(DR18['E_Bin'])

N_nue_pred_dr18 = np.array(DR18['nue_pre'])
N_numu_pred_dr18 = np.array(DR18['numu_pre'])
N_nue_obs_dr18 = np.array(DR18['nue_obs'])
N_numu_obs_dr18 = np.array(DR18['numu_obs'])

N_anue_pred_dr18 = np.array(DR18['anue_pre'])
N_anumu_pred_dr18 = np.array(DR18['anumu_pre'])
N_anue_obs_dr18 = np.array(DR18['anue_obs'])
N_anumu_obs_dr18 = np.array(DR18['anumu_obs'])

# Correct for Energy bins
E_bin_dr18 = E_bin_dr18[:9]

N_nue_pred_dr18 = N_nue_pred_dr18[:8]
N_nue_obs_dr18 = N_nue_obs_dr18[:8]

N_numu_pred_dr18 = N_numu_pred_dr18[:8]
N_numu_obs_dr18 = N_numu_obs_dr18[:8]

# Energy bin centers
E_points_dr18 = np.array([])
E_length_dr18 = np.array([])
for i in np.arange(len(E_bin_dr18)-1):
    E_points_dr18 = np.append(E_points_dr18, (E_bin_dr18[i]+E_bin_dr18[i+1])/2)
    E_length_dr18 = np.append(E_length_dr18, (E_bin_dr18[i+1]-E_bin_dr18[i])/2)
#print(E_points_dr18)
L_E_dr18 = ( L_nu / E_points_dr18 )

# Predicted flavour ratios
R_i_ab_nu_dr18 = N_numu_pred_dr18 / N_nue_pred_dr18

# Observed flavour ratios
R_f_ab_nu_dr18 = N_numu_obs_dr18 / N_nue_obs_dr18
Ra_f_ab_nu_dr18 = np.append(R_f_ab_nu_dr18, R_f_ab_nu_dr18[-1])

# Convert to numpy arrays
E_bin_dr20 = np.array(DR20['E_Bin'])

N_nue_pred_dr20 = np.array(DR20['nue_pre'])
N_numu_pred_dr20 = np.array(DR20['numu_pre'])
N_nue_obs_dr20 = np.array(DR20['nue_obs'])
N_numu_obs_dr20 = np.array(DR20['numu_obs'])

N_anue_pred_dr20 = np.array(DR20['anue_pre'])
N_anumu_pred_dr20 = np.array(DR20['anumu_pre'])
N_anue_obs_dr20 = np.array(DR20['anue_obs'])
N_anumu_obs_dr20 = np.array(DR20['anumu_obs'])

# Correct for Energy bins
E_bin_dr20 = E_bin_dr20[:9]

N_nue_pred_dr20 = N_nue_pred_dr20[:8]
N_numu_pred_dr20 = N_numu_pred_dr20[:8]
N_nue_obs_dr20 = N_nue_obs_dr20[:8]
N_numu_obs_dr20 = N_numu_obs_dr20[:8]

N_anue_pred_dr20 = N_anue_pred_dr20[:8]
N_anumu_pred_dr20 = N_anumu_pred_dr20[:8]
N_anue_obs_dr20 = N_anue_obs_dr20[:8]
N_anumu_obs_dr20 = N_anumu_obs_dr20[:8]

# Energy bin centers
E_points_dr20 = np.array([])
E_length_dr20 = np.array([])
for i in np.arange(len(E_bin_dr20)-1):
    E_points_dr20 = np.append(E_points_dr20, (E_bin_dr20[i]+E_bin_dr20[i+1])/2)
    E_length_dr20 = np.append(E_length_dr20, (E_bin_dr20[i+1]-E_bin_dr20[i])/2)
#print(E_points_dr20)
L_E_dr20 = ( L_nu / E_points_dr20 )

# Predicted flavour ratios
R_i_ab_nu_dr20 = N_numu_pred_dr20 / N_nue_pred_dr20
R_i_ab_anu_dr20 = N_anumu_pred_dr20 / N_anue_pred_dr20

# Observed flavour ratios
R_f_ab_nu_dr20 = N_numu_obs_dr20 / N_nue_obs_dr20
R_f_ab_anu_dr20 = N_numu_obs_dr20 / N_nue_obs_dr20

#MiniBooNE Data Release 2018 .....................................................

#
# Case 1 : Standard Oscillations
#
Delta_m2_vec = np.arange(0.0, 1.0, 0.001) # eV2
sin2_2_theta = np.arange(0.0, 1.0, 0.001) 
chi2_SO_nu_vec = np.array(DR18RF_SO_Nu['chi2_SO_nu'])
chi2_SO_anu_vec = np.array(DR18RF_SO_aNu['chi2_SO_anu'])
chi2_SO_nu_anu_vec = chi2_SO_nu_vec + chi2_SO_anu_vec

#
# Case 2 : Pseudo-Hermitian Oscillations
#
sigma_vec = np.arange(0.0, 1.0, 0.001)
alpha_vec = np.arange(0.0, 1.0, 0.001)
chi2_PHHO_nu_vec = np.array(DR18RF_PHHO_Nu['chi2_PHHO_anu'])
chi2_PHHO_anu_vec = np.array(DR18RF_PHHO_aNu['chi2_PHHO_anu'])
chi2_PHHO_nu_anu_vec = chi2_PHHO_nu_vec + chi2_PHHO_anu_vec

#
# chi2 to minimize
#

# Neutrino only .................................................................................
chi2_CriVal_68 = stats.chi2.ppf(0.68,(8)-1)
chi2_CriVal_95 = stats.chi2.ppf(0.95,(8)-1)
chi2_CriVal_99 = stats.chi2.ppf(0.99,(8)-1)

chi2_1 = chi2_SO_nu_vec
chi2_1_mat = np.transpose(chi2_1.reshape((sin2_2_theta.size, Delta_m2_vec.size)))
chi2_BestFit = np.amin(chi2_1_mat)
chi2_BestFit_pos = np.where(chi2_1_mat == chi2_BestFit)
sin2_2_theta_best = sin2_2_theta[chi2_BestFit_pos[1][0]]
Delta_m2_best = Delta_m2_vec[chi2_BestFit_pos[0][0]]

chi2_2 = chi2_PHHO_nu_vec
chi2_2_mat = np.transpose(chi2_2.reshape((sigma_vec.size, alpha_vec.size)))
chi2_2_BestFit = np.amin(chi2_2_mat)
chi2_2_BestFit_pos = np.where(chi2_2_mat == chi2_2_BestFit)
sigma_best = sigma_vec[chi2_2_BestFit_pos[1][0]]
alpha_best = alpha_vec[chi2_2_BestFit_pos[0][0]]


# Case 1 : Standard Oscillations
R_f_ab_1 = np.array([])
Ra_f_ab_1 = np.array([])
for i in np.arange(len(L_E_dr18)):
    p_ab_1 = (sin2_2_theta_best) * (np.sin(1.27*Delta_m2_best*L_E_dr18[i]))**2
    p_ba_1 = (sin2_2_theta_best) * (np.sin(1.27*Delta_m2_best*L_E_dr18[i]))**2
    p_aa_1 = 1 - p_ab_1
    p_bb_1 = 1 - p_ab_1
    
    R_f_ab_1_i = (R_i_ab_nu_dr18[i]*p_aa_1 + p_ba_1)/(p_bb_1 + R_i_ab_nu_dr18[i]*p_ab_1)  
    R_f_ab_1 = np.append(R_f_ab_1, R_f_ab_1_i)

    Ra_f_ab_1_i = (R_i_ab_nu_dr18[i]*p_aa_1 + p_ba_1)/(p_bb_1 + R_i_ab_nu_dr18[i]*p_ab_1)  
    Ra_f_ab_1 = np.append(Ra_f_ab_1, Ra_f_ab_1_i)

Ra_f_ab_1 = np.append(Ra_f_ab_1, Ra_f_ab_1[-1])

# Case 2 : Pseudo-Hermitian Oscillations
R_f_ab_2 = np.array([])
Ra_f_ab_2 = np.array([])
for i in np.arange(len(L_E_dr18)):
    p_ab_2 = (np.sin(alpha_best/2 - 1.27*np.abs(sigma_best*np.cos(alpha_best))*L_E_dr18[i]))**2
    p_ba_2 = (np.sin(alpha_best/2 + 1.27*np.abs(sigma_best*np.cos(alpha_best))*L_E_dr18[i]))**2
    p_aa_2 = (np.cos(alpha_best/2 - 1.27*np.abs(sigma_best*np.cos(alpha_best))*L_E_dr18[i]))**2
    p_bb_2 = (np.cos(alpha_best/2 + 1.27*np.abs(sigma_best*np.cos(alpha_best))*L_E_dr18[i]))**2
    
    R_f_ab_2_i = (R_i_ab_nu_dr18[i]*p_aa_2 + p_ba_2)/(p_bb_2 + R_i_ab_nu_dr18[i]*p_ab_2)    
    R_f_ab_2 = np.append(R_f_ab_2, R_f_ab_2_i)

    Ra_f_ab_2_i = (R_i_ab_nu_dr18[i]*p_aa_2 + p_ba_2)/(p_bb_2 + R_i_ab_nu_dr18[i]*p_ab_2)    
    Ra_f_ab_2 = np.append(Ra_f_ab_2, Ra_f_ab_2_i)

Ra_f_ab_2 = np.append(Ra_f_ab_2, Ra_f_ab_2[-1])

fig, ax = plt.subplots()
#ax.step(E_bin_dr18, Ra_f_ab_1-Ra_f_ab_nu_dr18, where = 'post', label='Standard')
#ax.step(E_bin_dr18, Ra_f_ab_2-Ra_f_ab_nu_dr18, where = 'post', label='PseudoHermitian')
#ax.errorbar(E_points_dr18, R_f_ab_nu_dr18, yerr=0.00, xerr=E_length_dr18, fmt='o', label='Observed')
ax.errorbar(E_points_dr18, R_f_ab_1-R_f_ab_nu_dr18, yerr=0.00, xerr=E_length_dr18, fmt='o', label='Standard')
ax.errorbar(E_points_dr18, R_f_ab_2-R_f_ab_nu_dr18, yerr=0.00, xerr=E_length_dr18, fmt='o', label='PseudoHermitian')
plt.xlabel(r'$E_{\nu} \, [MeV]$')
plt.ylabel(r'Residual $R^f_{\alpha / \beta}$')
ax.set_xlim(E_bin_dr18[0], E_bin_dr18[-1])
ax.set_ylim(-20, +20)
plt.grid()
plt.axhline(y = 0.0, color = 'r', linestyle = '-')
plt.legend(loc=1)
plt.savefig('plots/MBEA_DR18_Nu_Observed_vs_OM_OS_vs_PHH.png')


#MiniBooNE Data Release 2020 .....................................................

#
# Case 1 : Standard Oscillations
#
Delta_m2_vec = np.arange(0.0, 1.0, 0.001) # eV2
sin2_2_theta = np.arange(0.0, 1.0, 0.001) 
chi2_SO_nu_vec = np.array(DR20RF_SO_Nu['chi2_SO_nu'])
chi2_SO_anu_vec = np.array(DR20RF_SO_aNu['chi2_SO_anu'])
chi2_SO_nu_anu_vec = chi2_SO_nu_vec + chi2_SO_anu_vec

#
# Case 2 : Pseudo-Hermitian Oscillations
#
sigma_vec = np.arange(0.0, 1.0, 0.001)
alpha_vec = np.arange(0.0, 1.0, 0.001)
chi2_PHHO_nu_vec = np.array(DR20RF_PHHO_Nu['chi2_PHHO_anu'])
chi2_PHHO_anu_vec = np.array(DR20RF_PHHO_aNu['chi2_PHHO_anu'])
chi2_PHHO_nu_anu_vec = chi2_PHHO_nu_vec + chi2_PHHO_anu_vec

#
# chi2 to minimize
#

# Neutrino only .................................................................................
chi2_CriVal_68 = stats.chi2.ppf(0.68,(8)-1)
chi2_CriVal_95 = stats.chi2.ppf(0.95,(8)-1)
chi2_CriVal_99 = stats.chi2.ppf(0.99,(8)-1)

chi2_1 = chi2_SO_nu_vec
chi2_1_mat = np.transpose(chi2_1.reshape((sin2_2_theta.size, Delta_m2_vec.size)))
chi2_BestFit = np.amin(chi2_1_mat)
chi2_BestFit_pos = np.where(chi2_1_mat == chi2_BestFit)
sin2_2_theta_best = sin2_2_theta[chi2_BestFit_pos[1][0]]
Delta_m2_best = Delta_m2_vec[chi2_BestFit_pos[0][0]]

chi2_2 = chi2_PHHO_nu_vec
chi2_2_mat = np.transpose(chi2_2.reshape((sigma_vec.size, alpha_vec.size)))
chi2_2_BestFit = np.amin(chi2_2_mat)
chi2_2_BestFit_pos = np.where(chi2_2_mat == chi2_2_BestFit)
sigma_best = sigma_vec[chi2_2_BestFit_pos[1][0]]
alpha_best = alpha_vec[chi2_2_BestFit_pos[0][0]]


# Case 1 : Standard Oscillations
R_f_ab_1 = np.array([])
Ra_f_ab_1 = np.array([])
for i in np.arange(len(L_E_dr20)):
    p_ab_1 = (sin2_2_theta_best) * (np.sin(1.27*Delta_m2_best*L_E_dr20[i]))**2
    p_ba_1 = (sin2_2_theta_best) * (np.sin(1.27*Delta_m2_best*L_E_dr20[i]))**2
    p_aa_1 = 1 - p_ab_1
    p_bb_1 = 1 - p_ab_1
    
    R_f_ab_1_i = (R_i_ab_nu_dr20[i]*p_aa_1 + p_ba_1)/(p_bb_1 + R_i_ab_nu_dr20[i]*p_ab_1)  
    R_f_ab_1 = np.append(R_f_ab_1, R_f_ab_1_i)

    Ra_f_ab_1_i = (R_i_ab_nu_dr20[i]*p_aa_1 + p_ba_1)/(p_bb_1 + R_i_ab_nu_dr20[i]*p_ab_1)  
    Ra_f_ab_1 = np.append(Ra_f_ab_1, Ra_f_ab_1_i)

Ra_f_ab_1 = np.append(Ra_f_ab_1, Ra_f_ab_1[-1])

# Case 2 : Pseudo-Hermitian Oscillations
R_f_ab_2 = np.array([])
Ra_f_ab_2 = np.array([])
for i in np.arange(len(L_E_dr20)):
    p_ab_2 = (np.sin(alpha_best/2 - 1.27*np.abs(sigma_best*np.cos(alpha_best))*L_E_dr20[i]))**2
    p_ba_2 = (np.sin(alpha_best/2 + 1.27*np.abs(sigma_best*np.cos(alpha_best))*L_E_dr20[i]))**2
    p_aa_2 = (np.cos(alpha_best/2 - 1.27*np.abs(sigma_best*np.cos(alpha_best))*L_E_dr20[i]))**2
    p_bb_2 = (np.cos(alpha_best/2 + 1.27*np.abs(sigma_best*np.cos(alpha_best))*L_E_dr20[i]))**2
    
    R_f_ab_2_i = (R_i_ab_nu_dr20[i]*p_aa_2 + p_ba_2)/(p_bb_2 + R_i_ab_nu_dr20[i]*p_ab_2)    
    R_f_ab_2 = np.append(R_f_ab_2, R_f_ab_2_i)

    Ra_f_ab_2_i = (R_i_ab_nu_dr20[i]*p_aa_2 + p_ba_2)/(p_bb_2 + R_i_ab_nu_dr20[i]*p_ab_2)    
    Ra_f_ab_2 = np.append(Ra_f_ab_2, Ra_f_ab_2_i)

Ra_f_ab_2 = np.append(Ra_f_ab_2, Ra_f_ab_2[-1])

fig, ax = plt.subplots()
#ax.step(E_bin_dr18, Ra_f_ab_1-Ra_f_ab_nu_dr18, where = 'post', label='Standard')
#ax.step(E_bin_dr18, Ra_f_ab_2-Ra_f_ab_nu_dr18, where = 'post', label='PseudoHermitian')
#ax.errorbar(E_points_dr18, R_f_ab_nu_dr18, yerr=0.00, xerr=E_length_dr18, fmt='o', label='Observed')
ax.errorbar(E_points_dr20, R_f_ab_1-R_f_ab_nu_dr20, yerr=0.00, xerr=E_length_dr20, fmt='o', label='Standard')
ax.errorbar(E_points_dr20, R_f_ab_2-R_f_ab_nu_dr20, yerr=0.00, xerr=E_length_dr20, fmt='o', label='PseudoHermitian')
plt.xlabel(r'$E_{\nu} \, [MeV]$')
plt.ylabel(r'Residual $R^f_{\alpha / \beta}$')
ax.set_xlim(E_bin_dr20[0], E_bin_dr20[-1])
ax.set_ylim(-20, +20)
plt.grid()
plt.axhline(y = 0.0, color = 'r', linestyle = '-')
plt.legend(loc=1)
plt.savefig('plots/MBEA_DR20_Nu_Observed_vs_OM_OS_vs_PHH.png')

#MiniBooNE Data Release 2020 .....................................................

#
# Case 1 : Standard Oscillations
#
Delta_m2_vec = np.arange(0.0, 1.0, 0.001) # eV2
sin2_2_theta = np.arange(0.0, 1.0, 0.001) 
chi2_SO_nu_vec = np.array(DR20RF_SO_Nu['chi2_SO_nu'])
chi2_SO_anu_vec = np.array(DR20RF_SO_aNu['chi2_SO_anu'])
chi2_SO_nu_anu_vec = chi2_SO_nu_vec + chi2_SO_anu_vec

#
# Case 2 : Pseudo-Hermitian Oscillations
#
sigma_vec = np.arange(0.0, 1.0, 0.001)
alpha_vec = np.arange(0.0, 1.0, 0.001)
chi2_PHHO_nu_vec = np.array(DR20RF_PHHO_Nu['chi2_PHHO_anu'])
chi2_PHHO_anu_vec = np.array(DR20RF_PHHO_aNu['chi2_PHHO_anu'])
chi2_PHHO_nu_anu_vec = chi2_PHHO_nu_vec + chi2_PHHO_anu_vec

#
# chi2 to minimize
#

# Neutrino only .................................................................................
chi2_CriVal_68 = stats.chi2.ppf(0.68,(8)-1)
chi2_CriVal_95 = stats.chi2.ppf(0.95,(8)-1)
chi2_CriVal_99 = stats.chi2.ppf(0.99,(8)-1)

chi2_1 = chi2_SO_anu_vec
chi2_1_mat = np.transpose(chi2_1.reshape((sin2_2_theta.size, Delta_m2_vec.size)))
chi2_BestFit = np.amin(chi2_1_mat)
chi2_BestFit_pos = np.where(chi2_1_mat == chi2_BestFit)
sin2_2_theta_best = sin2_2_theta[chi2_BestFit_pos[1][0]]
Delta_m2_best = Delta_m2_vec[chi2_BestFit_pos[0][0]]

chi2_2 = chi2_PHHO_anu_vec
chi2_2_mat = np.transpose(chi2_2.reshape((sigma_vec.size, alpha_vec.size)))
chi2_2_BestFit = np.amin(chi2_2_mat)
chi2_2_BestFit_pos = np.where(chi2_2_mat == chi2_2_BestFit)
sigma_best = sigma_vec[chi2_2_BestFit_pos[1][0]]
alpha_best = alpha_vec[chi2_2_BestFit_pos[0][0]]


# Case 1 : Standard Oscillations
R_f_ab_1 = np.array([])
Ra_f_ab_1 = np.array([])
for i in np.arange(len(L_E_dr20)):
    p_ab_1 = (sin2_2_theta_best) * (np.sin(1.27*Delta_m2_best*L_E_dr20[i]))**2
    p_ba_1 = (sin2_2_theta_best) * (np.sin(1.27*Delta_m2_best*L_E_dr20[i]))**2
    p_aa_1 = 1 - p_ab_1
    p_bb_1 = 1 - p_ab_1
    
    R_f_ab_1_i = (R_i_ab_anu_dr20[i]*p_aa_1 + p_ba_1)/(p_bb_1 + R_i_ab_anu_dr20[i]*p_ab_1)  
    R_f_ab_1 = np.append(R_f_ab_1, R_f_ab_1_i)

    Ra_f_ab_1_i = (R_i_ab_anu_dr20[i]*p_aa_1 + p_ba_1)/(p_bb_1 + R_i_ab_anu_dr20[i]*p_ab_1)  
    Ra_f_ab_1 = np.append(Ra_f_ab_1, Ra_f_ab_1_i)

Ra_f_ab_1 = np.append(Ra_f_ab_1, Ra_f_ab_1[-1])

# Case 2 : Pseudo-Hermitian Oscillations
R_f_ab_2 = np.array([])
Ra_f_ab_2 = np.array([])
for i in np.arange(len(L_E_dr20)):
    p_ab_2 = (np.sin(alpha_best/2 - 1.27*np.abs(sigma_best*np.cos(alpha_best))*L_E_dr20[i]))**2
    p_ba_2 = (np.sin(alpha_best/2 + 1.27*np.abs(sigma_best*np.cos(alpha_best))*L_E_dr20[i]))**2
    p_aa_2 = (np.cos(alpha_best/2 - 1.27*np.abs(sigma_best*np.cos(alpha_best))*L_E_dr20[i]))**2
    p_bb_2 = (np.cos(alpha_best/2 + 1.27*np.abs(sigma_best*np.cos(alpha_best))*L_E_dr20[i]))**2
    
    R_f_ab_2_i = (R_i_ab_anu_dr20[i]*p_aa_2 + p_ba_2)/(p_bb_2 + R_i_ab_anu_dr20[i]*p_ab_2)    
    R_f_ab_2 = np.append(R_f_ab_2, R_f_ab_2_i)

    Ra_f_ab_2_i = (R_i_ab_anu_dr20[i]*p_aa_2 + p_ba_2)/(p_bb_2 + R_i_ab_anu_dr20[i]*p_ab_2)    
    Ra_f_ab_2 = np.append(Ra_f_ab_2, Ra_f_ab_2_i)

Ra_f_ab_2 = np.append(Ra_f_ab_2, Ra_f_ab_2[-1])

fig, ax = plt.subplots()
#ax.step(E_bin_dr18, Ra_f_ab_1-Ra_f_ab_nu_dr18, where = 'post', label='Standard')
#ax.step(E_bin_dr18, Ra_f_ab_2-Ra_f_ab_nu_dr18, where = 'post', label='PseudoHermitian')
#ax.errorbar(E_points_dr18, R_f_ab_nu_dr18, yerr=0.00, xerr=E_length_dr18, fmt='o', label='Observed')
ax.errorbar(E_points_dr20, R_f_ab_1-R_f_ab_anu_dr20, yerr=0.00, xerr=E_length_dr20, fmt='o', label='Standard')
ax.errorbar(E_points_dr20, R_f_ab_2-R_f_ab_anu_dr20, yerr=0.00, xerr=E_length_dr20, fmt='o', label='PseudoHermitian')
plt.xlabel(r'$E_{\bar{\nu}} \, [MeV]$')
plt.ylabel(r'Residual $R^f_{\alpha / \beta}$')
ax.set_xlim(E_bin_dr20[0], E_bin_dr20[-1])
ax.set_ylim(-20, +20)
plt.grid()
plt.axhline(y = 0.0, color = 'r', linestyle = '-')
plt.legend(loc=1)
plt.savefig('plots/MBEA_DR20_aNu_Observed_vs_OM_OS_vs_PHH.png')








# End of analysis
