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

# print to file
orig_stdout = sys.stdout
outfile = open('analysis/MBEA_OM_chi2_RF_BestFit_output.txt', 'w')
sys.stdout = outfile

#
# Import MiniBooNE data
#

# Read data from csv file

DR18RF_SO_Nu = pd.read_csv("analysis/MBEA_DR18_OM_chi2_analysis_RF_rawdata_SO_Nu.csv", sep=',')
DR18RF_SO_aNu = pd.read_csv("analysis/MBEA_DR18_OM_chi2_analysis_RF_rawdata_SO_aNu.csv", sep=',')
DR18RF_PHHO_Nu = pd.read_csv("analysis/MBEA_DR18_OM_chi2_analysis_RF_rawdata_PHHO_Nu.csv", sep=',')
DR18RF_PHHO_aNu = pd.read_csv("analysis/MBEA_DR18_OM_chi2_analysis_RF_rawdata_PHHO_aNu.csv", sep=',')

DR20RF_SO_Nu = pd.read_csv("analysis/MBEA_DR20_OM_chi2_analysis_RF_rawdata_SO_Nu.csv", sep=',')
DR20RF_SO_aNu = pd.read_csv("analysis/MBEA_DR20_OM_chi2_analysis_RF_rawdata_SO_aNu.csv", sep=',')
DR20RF_PHHO_Nu = pd.read_csv("analysis/MBEA_DR20_OM_chi2_analysis_RF_rawdata_PHHO_Nu.csv", sep=',')
DR20RF_PHHO_aNu = pd.read_csv("analysis/MBEA_DR20_OM_chi2_analysis_RF_rawdata_PHHO_aNu.csv", sep=',')


print('chi2_68 chi2_95 chi2_99 chi2_SO_BestFit sin2_2_theta Delta_m2 chi2_PHH_BestFit alpha sigma Case')

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

# Neutrino only
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

print(format(chi2_CriVal_68, '.3f'), ',', format(chi2_CriVal_95, '.3f'), ',', format(chi2_CriVal_99, '.3f'), ',', format(chi2_BestFit, '.3f'), ',', 
format(sin2_2_theta_best, '.3f'), ',',
format(Delta_m2_best, '.3f'), ',', format(chi2_2_BestFit, '.3f'), ',', format(alpha_best, '.3f'), ',', format(sigma_best, '.3f'), 
'\tNeutrinos Only DR18')


# Anti-Neutrino only
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

print(format(chi2_CriVal_68, '.3f'), ',', format(chi2_CriVal_95, '.3f'), ',', format(chi2_CriVal_99, '.3f'), ',', format(chi2_BestFit, '.3f'), ',', 
format(sin2_2_theta_best, '.3f'), ',',
format(Delta_m2_best, '.3f'), ',', format(chi2_2_BestFit, '.3f'), ',', format(alpha_best, '.3f'), ',', format(sigma_best, '.3f'), 
'\tAntiNeutrinos Only DR18')

# All Neutrinos
chi2_CriVal_68 = stats.chi2.ppf(0.68,(16)-1)
chi2_CriVal_95 = stats.chi2.ppf(0.95,(16)-1)
chi2_CriVal_99 = stats.chi2.ppf(0.99,(16)-1)

chi2_1 = chi2_SO_nu_anu_vec
chi2_1_mat = np.transpose(chi2_1.reshape((sin2_2_theta.size, Delta_m2_vec.size)))
chi2_BestFit = np.amin(chi2_1_mat)
chi2_BestFit_pos = np.where(chi2_1_mat == chi2_BestFit)
sin2_2_theta_best = sin2_2_theta[chi2_BestFit_pos[1][0]]
Delta_m2_best = Delta_m2_vec[chi2_BestFit_pos[0][0]]

chi2_2 = chi2_PHHO_nu_anu_vec
chi2_2_mat = np.transpose(chi2_2.reshape((sigma_vec.size, alpha_vec.size)))
chi2_2_BestFit = np.amin(chi2_2_mat)
chi2_2_BestFit_pos = np.where(chi2_2_mat == chi2_2_BestFit)
sigma_best = sigma_vec[chi2_2_BestFit_pos[1][0]]
alpha_best = alpha_vec[chi2_2_BestFit_pos[0][0]]

print(format(chi2_CriVal_68, '.3f'), ',', format(chi2_CriVal_95, '.3f'), ',', format(chi2_CriVal_99, '.3f'), ',', format(chi2_BestFit, '.3f')
, ',', format(sin2_2_theta_best, '.3f'), ',',
format(Delta_m2_best, '.3f'), ',', format(chi2_2_BestFit, '.3f'), ',', format(alpha_best, '.3f'), ',', format(sigma_best, '.3f'), 
'\tAll Neutrinos Only DR18')

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

# Neutrino only
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

print(format(chi2_CriVal_68, '.3f'), ',', format(chi2_CriVal_95, '.3f'), ',', format(chi2_CriVal_99, '.3f'), ',', format(chi2_BestFit, '.3f'), ',', 
format(sin2_2_theta_best, '.3f'), ',',
format(Delta_m2_best, '.3f'), ',', format(chi2_2_BestFit, '.3f'), ',', format(alpha_best, '.3f'), ',', format(sigma_best, '.3f'), '\tNeutrinos Only DR20')


# Anti-Neutrino only
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

print(format(chi2_CriVal_68, '.3f'), ',', format(chi2_CriVal_95, '.3f'), ',', format(chi2_CriVal_99, '.3f'), ',', format(chi2_BestFit, '.3f'), ',', 
format(sin2_2_theta_best, '.3f'), ',',
format(Delta_m2_best, '.3f'), ',', format(chi2_2_BestFit, '.3f'), ',', format(alpha_best, '.3f'), ',', format(sigma_best, '.3f'), '\tAntiNeutrinos Only DR20')

# All Neutrinos
chi2_CriVal_68 = stats.chi2.ppf(0.68,(16)-1)
chi2_CriVal_95 = stats.chi2.ppf(0.95,(16)-1)
chi2_CriVal_99 = stats.chi2.ppf(0.99,(16)-1)

chi2_1 = chi2_SO_nu_anu_vec
chi2_1_mat = np.transpose(chi2_1.reshape((sin2_2_theta.size, Delta_m2_vec.size)))
chi2_BestFit = np.amin(chi2_1_mat)
chi2_BestFit_pos = np.where(chi2_1_mat == chi2_BestFit)
sin2_2_theta_best = sin2_2_theta[chi2_BestFit_pos[1][0]]
Delta_m2_best = Delta_m2_vec[chi2_BestFit_pos[0][0]]

chi2_2 = chi2_PHHO_nu_anu_vec
chi2_2_mat = np.transpose(chi2_2.reshape((sigma_vec.size, alpha_vec.size)))
chi2_2_BestFit = np.amin(chi2_2_mat)
chi2_2_BestFit_pos = np.where(chi2_2_mat == chi2_2_BestFit)
sigma_best = sigma_vec[chi2_2_BestFit_pos[1][0]]
alpha_best = alpha_vec[chi2_2_BestFit_pos[0][0]]

print(format(chi2_CriVal_68, '.3f'), ',', format(chi2_CriVal_95, '.3f'), ',', format(chi2_CriVal_99, '.3f'), ',', format(chi2_BestFit, '.3f'), ',', 
format(sin2_2_theta_best, '.3f'), ',',
format(Delta_m2_best, '.3f'), ',', format(chi2_2_BestFit, '.3f'), ',', format(alpha_best, '.3f'), ',', format(sigma_best, '.3f'), '\tAll Neutrinos Only DR20')

# End of analysis
