"""
MiniBooNE Excess Analysis - Oscillation Models - Chi2 Analysis
Basic tasks:
1- Import the data
2- Compute Observed Ratios using Oscillation probabilities of 2 Models
3- Compute Chi2 goodness of fit for each model
4- scan parameters plan
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

#Perform Chi-Square goodness of fit test and print results 
def chi2calculate(observed, expected):
    chi2 = 0.0
    chi2 = (expected-observed)**2/expected
    #chi2_1 = 2*(expected - observed + observed * np.log(observed / expected))
    chi2 = chi2.sum()
    return chi2

def chi2calculatemod(observed, expected):
    chi2 = 0.0
    chi2 = 2*(expected - observed + observed * np.log(observed / expected))
    chi2 = chi2.sum()
    return chi2

#
# Import MiniBooNE data
#

# Baseline
L_nu = 541

# Binned Event Numbers
DR18 = pd.read_csv("data/MB_DR2018.csv", sep=',') # Data Release 2018
DR20 = pd.read_csv("data/MB_DR2020.csv", sep=',') # Data Release 2020

#
# Data Release 2018
#
print("Data Release 2018 ..........................................................")
# Convert to numpy arrays
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
N_numu_pred_dr18 = N_numu_pred_dr18[:8]
N_nue_obs_dr18 = N_nue_obs_dr18[:8]
N_numu_obs_dr18 = N_numu_obs_dr18[:8]

N_anue_pred_dr18 = N_anue_pred_dr18[:8]
N_anumu_pred_dr18 = N_anumu_pred_dr18[:8]
N_anue_obs_dr18 = N_anue_obs_dr18[:8]
N_anumu_obs_dr18 = N_anumu_obs_dr18[:8]

# Another Method include last range
#E_bin_dr18 = E_bin_dr18[:8]
#E_bin_dr18 = np.append_dr18(E_bin, 3000.)

# Energy bin centers
E_points_dr18 = np.array([])
E_length_dr18 = np.array([])
for i in np.arange(len(E_bin_dr18)-1):
    E_points_dr18 = np.append(E_points_dr18, (E_bin_dr18[i]+E_bin_dr18[i+1])/2)
    E_length_dr18 = np.append(E_length_dr18, (E_bin_dr18[i+1]-E_bin_dr18[i])/2)
L_E_dr18 = ( L_nu / E_points_dr18 )

# Predicted flavour ratios
R_i_ab_nu_dr18 = N_numu_pred_dr18 / N_nue_pred_dr18
R_i_ab_anu_dr18 = N_anumu_pred_dr18 / N_anue_pred_dr18

# Observed flavour ratios
R_f_ab_nu_dr18 = N_numu_obs_dr18 / N_nue_obs_dr18
R_f_ab_anu_dr18 = N_anumu_obs_dr18 / N_anue_obs_dr18

#
# Case 1 : SO - Standard Oscillations
#
print("Case 1 : SO - Standard Oscillations")
# loop over wanted ranges
Delta_m2_vec = np.arange(0.0, 1.0, 0.001) # eV2
sin2_2_theta = np.arange(0.0, 1.0, 0.001) 

print("Case 1.1 : Neutrinos")
chi2_SO_nu_vec = np.array([])
for m in np.arange(len(sin2_2_theta)):
    sin22theta = sin2_2_theta[m]
    for n in np.arange(len(Delta_m2_vec)):
        Delta_m2 = Delta_m2_vec[n]
        R_f_ab_SO_nu = np.zeros(len(L_E_dr18))
        for i in np.arange(len(L_E_dr18)):
            p_ab_1 = (sin22theta) * (np.sin(1.27*Delta_m2*L_E_dr18[i]))**2
            p_ba_1 = p_ab_1
            p_aa_1 = 1 - p_ab_1
            p_bb_1 = 1 - p_ba_1
            R_f_ab_SO_nu[i] = (R_i_ab_nu_dr18[i]*p_aa_1 + p_ba_1)/(p_bb_1 + R_i_ab_nu_dr18[i]*p_ab_1)
        # Calculate Chi2 
        chi2_SO_nu_vec = np.append(chi2_SO_nu_vec, chi2calculate(R_f_ab_nu_dr18, R_f_ab_SO_nu))
data = {'chi2_SO_nu': chi2_SO_nu_vec}
df = pd.DataFrame(data)
df.to_csv ('analysis/MBEA_DR18_OM_chi2_analysis_RF_rawdata_SO_Nu.csv', index = False, header=True)

print("Case 1.2 : AntiNeutrinos")
chi2_SO_anu_vec = np.array([])
for m in np.arange(len(sin2_2_theta)):
    sin22theta = sin2_2_theta[m]
    for n in np.arange(len(Delta_m2_vec)):
        Delta_m2 = Delta_m2_vec[n]
        R_f_ab_SO_anu = np.zeros(len(L_E_dr18))
        for i in np.arange(len(L_E_dr18)):
            p_ab_1 = (sin22theta) * (np.sin(1.27*Delta_m2*L_E_dr18[i]))**2
            p_ba_1 = p_ab_1
            p_aa_1 = 1 - p_ab_1
            p_bb_1 = 1 - p_ba_1
            R_f_ab_SO_anu[i] = (R_i_ab_anu_dr18[i]*p_aa_1 + p_ba_1)/(p_bb_1 + R_i_ab_anu_dr18[i]*p_ab_1)
        # Calculate Chi2 
        chi2_SO_anu_vec = np.append(chi2_SO_anu_vec, chi2calculate(R_f_ab_anu_dr18, R_f_ab_SO_anu) )
data = {'chi2_SO_anu': chi2_SO_anu_vec}
df = pd.DataFrame(data)
df.to_csv ('analysis/MBEA_DR18_OM_chi2_analysis_RF_rawdata_SO_aNu.csv', index = False, header=True)


#
# Case 2 : PHHO - Pseudo-Hermitian Hamiltonian Oscillations
#
print("\nCase 2 : PHHO - Pseudo-Hermitian Hamiltonian Oscillations")
# loop over wanted ranges
sigma_vec = np.arange(0.0, 1.0, 0.001)
alpha_vec = np.arange(0.0, 1.0, 0.001)

print("Case 2.1 : Neutrinos")
chi2_PHHO_nu_vec = np.array([])
for n in np.arange(len(sigma_vec)):
    sigma = sigma_vec[n]
    for m in np.arange(len(alpha_vec)):
        alpha = alpha_vec[m]
        R_f_ab_PHHO_nu = np.zeros(len(L_E_dr18))
        for i in np.arange(len(L_E_dr18)):
            p_ab_2 = (np.sin(alpha/2 - 1.27*np.abs(sigma*np.cos(alpha))*L_E_dr18[i]))**2
            p_ba_2 = (np.sin(alpha/2 + 1.27*np.abs(sigma*np.cos(alpha))*L_E_dr18[i]))**2
            p_aa_2 = (np.cos(alpha/2 - 1.27*np.abs(sigma*np.cos(alpha))*L_E_dr18[i]))**2
            p_bb_2 = (np.cos(alpha/2 + 1.27*np.abs(sigma*np.cos(alpha))*L_E_dr18[i]))**2
            R_f_ab_PHHO_nu[i] = (R_i_ab_nu_dr18[i]*p_aa_2 + p_ba_2)/(p_bb_2 + R_i_ab_nu_dr18[i]*p_ab_2)
        # Calculate Chi2
        chi2_PHHO_nu_vec = np.append(chi2_PHHO_nu_vec, chi2calculate(R_f_ab_nu_dr18, R_f_ab_PHHO_nu))
data = {'chi2_PHHO_nu': chi2_PHHO_nu_vec}
df = pd.DataFrame(data)
df.to_csv ('analysis/MBEA_DR18_OM_chi2_analysis_RF_rawdata_PHHO_Nu.csv', index = False, header=True)

print("Case 2.2 : AntiNeutrinos")
chi2_PHHO_anu_vec = np.array([])
for n in np.arange(len(sigma_vec)):
    sigma = sigma_vec[n]
    for m in np.arange(len(alpha_vec)):
        alpha = alpha_vec[m]
        R_f_ab_PHHO_anu = np.zeros(len(L_E_dr18))
        for i in np.arange(len(L_E_dr18)):
            p_ab_2 = (np.sin(alpha/2 - 1.27*np.abs(sigma*np.cos(alpha))*L_E_dr18[i]))**2
            p_ba_2 = (np.sin(alpha/2 + 1.27*np.abs(sigma*np.cos(alpha))*L_E_dr18[i]))**2
            p_aa_2 = (np.cos(alpha/2 - 1.27*np.abs(sigma*np.cos(alpha))*L_E_dr18[i]))**2
            p_bb_2 = (np.cos(alpha/2 + 1.27*np.abs(sigma*np.cos(alpha))*L_E_dr18[i]))**2
            R_f_ab_PHHO_anu[i] = (R_i_ab_anu_dr18[i]*p_aa_2 + p_ba_2)/(p_bb_2 + R_i_ab_anu_dr18[i]*p_ab_2)
        # Calculate Chi2
        chi2_PHHO_anu_vec = np.append(chi2_PHHO_anu_vec, chi2calculate(R_f_ab_anu_dr18, R_f_ab_PHHO_anu))
data = {'chi2_PHHO_anu': chi2_PHHO_anu_vec}
df = pd.DataFrame(data)
df.to_csv ('analysis/MBEA_DR18_OM_chi2_analysis_RF_rawdata_PHHO_aNu.csv', index = False, header=True)

#
# Data Release 2020
#
print("\nData Release 2020 ..........................................................")
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

# Another Method include last range
#E_bin_dr20 = E_bin_dr20[:8]
#E_bin_dr20 = np.append_dr20(E_bin, 3000.)

# Energy bin centers
E_points_dr20 = np.array([])
E_length_dr20 = np.array([])
for i in np.arange(len(E_bin_dr20)-1):
    E_points_dr20 = np.append(E_points_dr20, (E_bin_dr20[i]+E_bin_dr20[i+1])/2)
    E_length_dr20 = np.append(E_length_dr20, (E_bin_dr20[i+1]-E_bin_dr20[i])/2)
L_E_dr20 = ( L_nu / E_points_dr20 )

# Predicted flavour ratios
R_i_ab_nu_dr20 = N_numu_pred_dr20 / N_nue_pred_dr20
R_i_ab_anu_dr20 = N_anumu_pred_dr20 / N_anue_pred_dr20

# Observed flavour ratios
R_f_ab_nu_dr20 = N_numu_obs_dr20 / N_nue_obs_dr20
R_f_ab_anu_dr20 = N_anumu_obs_dr20 / N_anue_obs_dr20

#
# Case 1 : SO - Standard Oscillations
#
print("Case 1 : SO - Standard Oscillations")
# loop over wanted ranges
Delta_m2_vec = np.arange(0.0, 1.0, 0.001) # eV2
sin2_2_theta = np.arange(0.0, 1.0, 0.001) 

print("Case 1.1 : Neutrinos")
chi2_SO_nu_vec = np.array([])
for m in np.arange(len(sin2_2_theta)):
    sin22theta = sin2_2_theta[m]
    for n in np.arange(len(Delta_m2_vec)):
        Delta_m2 = Delta_m2_vec[n]
        R_f_ab_SO_nu = np.zeros(len(L_E_dr20))
        for i in np.arange(len(L_E_dr20)):
            p_ab_1 = (sin22theta) * (np.sin(1.27*Delta_m2*L_E_dr20[i]))**2
            p_ba_1 = p_ab_1
            p_aa_1 = 1 - p_ab_1
            p_bb_1 = 1 - p_ba_1
            R_f_ab_SO_nu[i] = (R_i_ab_nu_dr20[i]*p_aa_1 + p_ba_1)/(p_bb_1 + R_i_ab_nu_dr20[i]*p_ab_1)
        # Calculate Chi2 
        chi2_SO_nu_vec = np.append(chi2_SO_nu_vec, chi2calculate(R_f_ab_nu_dr20, R_f_ab_SO_nu))
data = {'chi2_SO_nu': chi2_SO_nu_vec}
df = pd.DataFrame(data)
df.to_csv ('analysis/MBEA_DR20_OM_chi2_analysis_RF_rawdata_SO_Nu.csv', index = False, header=True)

print("Case 1.2 : AntiNeutrinos")
chi2_SO_anu_vec = np.array([])
for m in np.arange(len(sin2_2_theta)):
    sin22theta = sin2_2_theta[m]
    for n in np.arange(len(Delta_m2_vec)):
        Delta_m2 = Delta_m2_vec[n]
        R_f_ab_SO_anu = np.zeros(len(L_E_dr20))
        for i in np.arange(len(L_E_dr20)):
            p_ab_1 = (sin22theta) * (np.sin(1.27*Delta_m2*L_E_dr20[i]))**2
            p_ba_1 = p_ab_1
            p_aa_1 = 1 - p_ab_1
            p_bb_1 = 1 - p_ba_1
            R_f_ab_SO_anu[i] = (R_i_ab_anu_dr20[i]*p_aa_1 + p_ba_1)/(p_bb_1 + R_i_ab_anu_dr20[i]*p_ab_1)
        # Calculate Chi2 
        chi2_SO_anu_vec = np.append(chi2_SO_anu_vec, chi2calculate(R_f_ab_anu_dr20, R_f_ab_SO_anu) )
data = {'chi2_SO_anu': chi2_SO_anu_vec}
df = pd.DataFrame(data)
df.to_csv ('analysis/MBEA_DR20_OM_chi2_analysis_RF_rawdata_SO_aNu.csv', index = False, header=True)


#
# Case 2 : PHHO - Pseudo-Hermitian Hamiltonian Oscillations
#
print("\nCase 2 : PHHO - Pseudo-Hermitian Hamiltonian Oscillations")
# loop over wanted ranges
sigma_vec = np.arange(0.0, 1.0, 0.001)
alpha_vec = np.arange(0.0, 1.0, 0.001)

print("Case 2.1 : Neutrinos")
chi2_PHHO_nu_vec = np.array([])
for n in np.arange(len(sigma_vec)):
    sigma = sigma_vec[n]
    for m in np.arange(len(alpha_vec)):
        alpha = alpha_vec[m]
        R_f_ab_PHHO_nu = np.zeros(len(L_E_dr20))
        for i in np.arange(len(L_E_dr20)):
            p_ab_2 = (np.sin(alpha/2 - 1.27*np.abs(sigma*np.cos(alpha))*L_E_dr20[i]))**2
            p_ba_2 = (np.sin(alpha/2 + 1.27*np.abs(sigma*np.cos(alpha))*L_E_dr20[i]))**2
            p_aa_2 = (np.cos(alpha/2 - 1.27*np.abs(sigma*np.cos(alpha))*L_E_dr20[i]))**2
            p_bb_2 = (np.cos(alpha/2 + 1.27*np.abs(sigma*np.cos(alpha))*L_E_dr20[i]))**2
            R_f_ab_PHHO_nu[i] = (R_i_ab_nu_dr20[i]*p_aa_2 + p_ba_2)/(p_bb_2 + R_i_ab_nu_dr20[i]*p_ab_2)
        # Calculate Chi2
        chi2_PHHO_nu_vec = np.append(chi2_PHHO_nu_vec, chi2calculate(R_f_ab_nu_dr20, R_f_ab_PHHO_nu))
data = {'chi2_PHHO_anu': chi2_PHHO_nu_vec}
df = pd.DataFrame(data)
df.to_csv ('analysis/MBEA_DR20_OM_chi2_analysis_RF_rawdata_PHHO_Nu.csv', index = False, header=True)

print("Case 2.2 : AntiNeutrinos")
chi2_PHHO_anu_vec = np.array([])
for n in np.arange(len(sigma_vec)):
    sigma = sigma_vec[n]
    for m in np.arange(len(alpha_vec)):
        alpha = alpha_vec[m]
        R_f_ab_PHHO_anu = np.zeros(len(L_E_dr20))
        for i in np.arange(len(L_E_dr20)):
            p_ab_2 = (np.sin(alpha/2 - 1.27*np.abs(sigma*np.cos(alpha))*L_E_dr20[i]))**2
            p_ba_2 = (np.sin(alpha/2 + 1.27*np.abs(sigma*np.cos(alpha))*L_E_dr20[i]))**2
            p_aa_2 = (np.cos(alpha/2 - 1.27*np.abs(sigma*np.cos(alpha))*L_E_dr20[i]))**2
            p_bb_2 = (np.cos(alpha/2 + 1.27*np.abs(sigma*np.cos(alpha))*L_E_dr20[i]))**2
            R_f_ab_PHHO_anu[i] = (R_i_ab_anu_dr20[i]*p_aa_2 + p_ba_2)/(p_bb_2 + R_i_ab_anu_dr20[i]*p_ab_2)
        # Calculate Chi2
        chi2_PHHO_anu_vec = np.append(chi2_PHHO_anu_vec, chi2calculate(R_f_ab_anu_dr20, R_f_ab_PHHO_anu))
data = {'chi2_PHHO_anu': chi2_PHHO_anu_vec}
df = pd.DataFrame(data)
df.to_csv ('analysis/MBEA_DR20_OM_chi2_analysis_RF_rawdata_PHHO_aNu.csv', index = False, header=True)

print("\nSuccessfull!")
# End of analysis
