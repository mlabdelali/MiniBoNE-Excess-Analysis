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

# Read data from csv file

DR18RF_SO_Nu = pd.read_csv("analysis/MBEA_DR18_OM_chi2_analysis_RF_rawdata_SO_Nu.csv", sep=',')
DR18RF_SO_aNu = pd.read_csv("analysis/MBEA_DR18_OM_chi2_analysis_RF_rawdata_SO_aNu.csv", sep=',')

DR20RF_SO_Nu = pd.read_csv("analysis/MBEA_DR20_OM_chi2_analysis_RF_rawdata_SO_Nu.csv", sep=',')
DR20RF_SO_aNu = pd.read_csv("analysis/MBEA_DR20_OM_chi2_analysis_RF_rawdata_SO_aNu.csv", sep=',')


# MiniBooNE Analysis Contours
DR18_Nu_CL_1 = pd.read_csv("data/MB_DR2018_cont_fake_apr18_contNu_1s.csv", sep=',', header=None)
DR18_Nu_CL_2 = pd.read_csv("data/MB_DR2018_cont_fake_apr18_contNu_90.csv", sep=',', header=None)
DR18_Nu_CL_3 = pd.read_csv("data/MB_DR2018_cont_fake_apr18_contNu_99.csv", sep=',', header=None)

Nu_s22t_1_vec = np.array(DR18_Nu_CL_1[0])
Nu_Dm2_1_vec = np.array(DR18_Nu_CL_1[1])

Nu_s22t_2_vec = np.array(DR18_Nu_CL_2[0])
Nu_Dm2_2_vec = np.array(DR18_Nu_CL_2[1])

Nu_s22t_3_vec = np.array(DR18_Nu_CL_3[0])
Nu_Dm2_3_vec = np.array(DR18_Nu_CL_3[1])

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

fig, ax = plt.subplots()

CS1 = ax.contourf(sin2_2_theta, Delta_m2_vec, chi2_1_mat, levels = [chi2_CriVal_68, chi2_CriVal_95, chi2_CriVal_99], colors= ['red', 'green', 'blue'], alpha=0.6, extend='min')
CS1.levels = ["68%", "90%", "95%"];
#ax.clabel(CS1, CS1.levels, inline=1, fontsize=14)
ax.scatter(sin2_2_theta_best, Delta_m2_best, marker='*', s=200)

#MiniBooNE Analysis
ax.scatter(Nu_s22t_1_vec, Nu_Dm2_1_vec, label=r'MiniBooNE 1$\sigma$', s=5, c='pink')
ax.scatter(Nu_s22t_2_vec, Nu_Dm2_2_vec, label=r'MiniBooNE 90%\sigma$', s=5, c='magenta')
ax.scatter(Nu_s22t_3_vec, Nu_Dm2_3_vec, label='MiniBooNE 99%', s=5, c='black')
ax.scatter(0.918, 0.041, label=r'MiniBooNE Best Fit', marker='*', s=200, c='magenta')
ax.scatter(0.01, 0.4, label=r'MiniBooNE Best Fit 2', marker='*', s=200, c='black')
#MiniBooNE Analysis

#ax.set_title("MiniBooNE data 2018 - Sensitivity Levels \n Best Fit: "  r"$sin^2(2\theta)$ = %.03f , $\Delta m ^2$ = %.03f" "\n" r"$\chi^2$ = %.06f" %(sin2_2_theta_best, Delta_m2_best, chi2_BestFit), fontsize=16)
plt.xlabel(r'$sin^2(2\theta)$', fontsize=14)
plt.ylabel(r'$\Delta m ^2 (eV^2)$', fontsize=14)
plt.xlim(0.0, 0.05)
plt.ylim(0.0, 1.0)
plt.grid(which='both')
plt.savefig('plots/MBEA_DR18_RF_Contours_Compare_SO_Nu_Only.png')

# Anti-Neutrino only .................................................................................
chi2_CriVal_68 = stats.chi2.ppf(0.68,(8)-1)
chi2_CriVal_95 = stats.chi2.ppf(0.95,(8)-1)
chi2_CriVal_99 = stats.chi2.ppf(0.99,(8)-1)

chi2_1 = chi2_SO_anu_vec
chi2_1_mat = np.transpose(chi2_1.reshape((sin2_2_theta.size, Delta_m2_vec.size)))
chi2_BestFit = np.amin(chi2_1_mat)
chi2_BestFit_pos = np.where(chi2_1_mat == chi2_BestFit)
sin2_2_theta_best = sin2_2_theta[chi2_BestFit_pos[1][0]]
Delta_m2_best = Delta_m2_vec[chi2_BestFit_pos[0][0]]

fig, ax = plt.subplots()

CS1 = ax.contourf(sin2_2_theta, Delta_m2_vec, chi2_1_mat, levels = [chi2_CriVal_68, chi2_CriVal_95, chi2_CriVal_99], colors= ['red', 'green', 'blue'], alpha=0.6, extend='min')
CS1.levels = ["68%", "90%", "95%"];
#ax.clabel(CS1, CS1.levels, inline=1, fontsize=14)
ax.scatter(sin2_2_theta_best, Delta_m2_best, marker='*', s=200)

#MiniBooNE Analysis
ax.scatter(Nu_s22t_1_vec, Nu_Dm2_1_vec, label=r'MiniBooNE 1$\sigma$', s=5, c='pink')
ax.scatter(Nu_s22t_2_vec, Nu_Dm2_2_vec, label=r'MiniBooNE 90%\sigma$', s=5, c='magenta')
ax.scatter(Nu_s22t_3_vec, Nu_Dm2_3_vec, label='MiniBooNE 99%', s=5, c='black')
ax.scatter(0.918, 0.041, label=r'MiniBooNE Best Fit', marker='*', s=200, c='magenta')
ax.scatter(0.01, 0.4, label=r'MiniBooNE Best Fit 2', marker='*', s=200, c='black')
#MiniBooNE Analysis

#ax.set_title("MiniBooNE data 2018 - Sensitivity Levels \n Best Fit: "  r"$sin^2(2\theta)$ = %.03f , $\Delta m ^2$ = %.03f" "\n" r"$\chi^2$ = %.06f" %(sin2_2_theta_best, Delta_m2_best, chi2_BestFit), fontsize=16)
plt.xlabel(r'$sin^2(2\theta)$', fontsize=14)
plt.ylabel(r'$\Delta m ^2 (eV^2)$', fontsize=14)
plt.xlim(0.0, 0.05)
plt.ylim(0.0, 1.0)
plt.grid(which='both')
plt.savefig('plots/MBEA_DR18_RF_Contours_Compare_SO_aNu_Only.png')

# All Neutrinos .................................................................................
chi2_CriVal_68 = stats.chi2.ppf(0.68,(16)-1)
chi2_CriVal_95 = stats.chi2.ppf(0.95,(16)-1)
chi2_CriVal_99 = stats.chi2.ppf(0.99,(16)-1)

chi2_1 = chi2_SO_nu_anu_vec
chi2_1_mat = np.transpose(chi2_1.reshape((sin2_2_theta.size, Delta_m2_vec.size)))
chi2_BestFit = np.amin(chi2_1_mat)
chi2_BestFit_pos = np.where(chi2_1_mat == chi2_BestFit)
sin2_2_theta_best = sin2_2_theta[chi2_BestFit_pos[1][0]]
Delta_m2_best = Delta_m2_vec[chi2_BestFit_pos[0][0]]

fig, ax = plt.subplots()

CS1 = ax.contourf(sin2_2_theta, Delta_m2_vec, chi2_1_mat, levels = [chi2_CriVal_68, chi2_CriVal_95, chi2_CriVal_99], colors= ['red', 'green', 'blue'], alpha=0.6, extend='min')
CS1.levels = ["68%", "90%", "95%"];
#ax.clabel(CS1, CS1.levels, inline=1, fontsize=14)
ax.scatter(sin2_2_theta_best, Delta_m2_best, marker='*', s=200)

#MiniBooNE Analysis
ax.scatter(Nu_s22t_1_vec, Nu_Dm2_1_vec, label=r'MiniBooNE 1$\sigma$', s=5, c='pink')
ax.scatter(Nu_s22t_2_vec, Nu_Dm2_2_vec, label=r'MiniBooNE 90%\sigma$', s=5, c='magenta')
ax.scatter(Nu_s22t_3_vec, Nu_Dm2_3_vec, label='MiniBooNE 99%', s=5, c='black')
ax.scatter(0.918, 0.041, label=r'MiniBooNE Best Fit', marker='*', s=200, c='magenta')
ax.scatter(0.01, 0.4, label=r'MiniBooNE Best Fit 2', marker='*', s=200, c='black')
#MiniBooNE Analysis

#ax.set_title("MiniBooNE data 2018 - Sensitivity Levels \n Best Fit: "  r"$sin^2(2\theta)$ = %.03f , $\Delta m ^2$ = %.03f" "\n" r"$\chi^2$ = %.06f" %(sin2_2_theta_best, Delta_m2_best, chi2_BestFit), fontsize=16)
plt.xlabel(r'$sin^2(2\theta)$', fontsize=14)
plt.ylabel(r'$\Delta m ^2 (eV^2)$', fontsize=14)
plt.xlim(0.0, 0.05)
plt.ylim(0.0, 1.0)
plt.grid(which='both')
plt.savefig('plots/MBEA_DR18_RF_Contours_Compare_SO_Nu_ANu.png')

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

fig, ax = plt.subplots()

CS1 = ax.contourf(sin2_2_theta, Delta_m2_vec, chi2_1_mat, levels = [chi2_CriVal_68, chi2_CriVal_95, chi2_CriVal_99], colors= ['red', 'green', 'blue'], alpha=0.6, extend='min')
CS1.levels = ["68%", "90%", "95%"];
#ax.clabel(CS1, CS1.levels, inline=1, fontsize=14)
ax.scatter(sin2_2_theta_best, Delta_m2_best, marker='*', s=200)

#MiniBooNE Analysis
ax.scatter(Nu_s22t_1_vec, Nu_Dm2_1_vec, label=r'MiniBooNE 1$\sigma$', s=5, c='pink')
ax.scatter(Nu_s22t_2_vec, Nu_Dm2_2_vec, label=r'MiniBooNE 90%\sigma$', s=5, c='magenta')
ax.scatter(Nu_s22t_3_vec, Nu_Dm2_3_vec, label='MiniBooNE 99%', s=5, c='black')
ax.scatter(0.918, 0.041, label=r'MiniBooNE Best Fit', marker='*', s=200, c='magenta')
ax.scatter(0.01, 0.4, label=r'MiniBooNE Best Fit 2', marker='*', s=200, c='black')
#MiniBooNE Analysis

#ax.set_title("MiniBooNE data 2020 - Sensitivity Levels \n Best Fit: "  r"$sin^2(2\theta)$ = %.03f , $\Delta m ^2$ = %.03f" "\n" r"$\chi^2$ = %.06f" %(sin2_2_theta_best, Delta_m2_best, chi2_BestFit), fontsize=16)
plt.xlabel(r'$sin^2(2\theta)$', fontsize=14)
plt.ylabel(r'$\Delta m ^2 (eV^2)$', fontsize=14)
plt.xlim(0.0, 0.05)
plt.ylim(0.0, 1.0)
plt.grid(which='both')
plt.savefig('plots/MBEA_DR20_RF_Contours_Compare_SO_Nu_Only.png')

# Anti-Neutrino only .................................................................................
chi2_CriVal_68 = stats.chi2.ppf(0.68,(8)-1)
chi2_CriVal_95 = stats.chi2.ppf(0.95,(8)-1)
chi2_CriVal_99 = stats.chi2.ppf(0.99,(8)-1)

chi2_1 = chi2_SO_anu_vec
chi2_1_mat = np.transpose(chi2_1.reshape((sin2_2_theta.size, Delta_m2_vec.size)))
chi2_BestFit = np.amin(chi2_1_mat)
chi2_BestFit_pos = np.where(chi2_1_mat == chi2_BestFit)
sin2_2_theta_best = sin2_2_theta[chi2_BestFit_pos[1][0]]
Delta_m2_best = Delta_m2_vec[chi2_BestFit_pos[0][0]]

fig, ax = plt.subplots()

CS1 = ax.contourf(sin2_2_theta, Delta_m2_vec, chi2_1_mat, levels = [chi2_CriVal_68, chi2_CriVal_95, chi2_CriVal_99], colors= ['red', 'green', 'blue'], alpha=0.6, extend='min')
CS1.levels = ["68%", "90%", "95%"];
#ax.clabel(CS1, CS1.levels, inline=1, fontsize=14)
ax.scatter(sin2_2_theta_best, Delta_m2_best, marker='*', s=200)

#MiniBooNE Analysis
ax.scatter(Nu_s22t_1_vec, Nu_Dm2_1_vec, label=r'MiniBooNE 1$\sigma$', s=5, c='pink')
ax.scatter(Nu_s22t_2_vec, Nu_Dm2_2_vec, label=r'MiniBooNE 90%\sigma$', s=5, c='magenta')
ax.scatter(Nu_s22t_3_vec, Nu_Dm2_3_vec, label='MiniBooNE 99%', s=5, c='black')
ax.scatter(0.918, 0.041, label=r'MiniBooNE Best Fit', marker='*', s=200, c='magenta')
ax.scatter(0.01, 0.4, label=r'MiniBooNE Best Fit 2', marker='*', s=200, c='black')
#MiniBooNE Analysis

#ax.set_title("MiniBooNE data 2020 - Sensitivity Levels \n Best Fit: "  r"$sin^2(2\theta)$ = %.03f , $\Delta m ^2$ = %.03f" "\n" r"$\chi^2$ = %.06f" %(sin2_2_theta_best, Delta_m2_best, chi2_BestFit), fontsize=16)
plt.xlabel(r'$sin^2(2\theta)$', fontsize=14)
plt.ylabel(r'$\Delta m ^2 (eV^2)$', fontsize=14)
plt.xlim(0.0, 0.05)
plt.ylim(0.0, 1.0)
plt.grid(which='both')
plt.savefig('plots/MBEA_DR20_RF_Contours_Compare_SO_aNu_Only.png')

# All Neutrinos .................................................................................
chi2_CriVal_68 = stats.chi2.ppf(0.68,(16)-1)
chi2_CriVal_95 = stats.chi2.ppf(0.95,(16)-1)
chi2_CriVal_99 = stats.chi2.ppf(0.99,(16)-1)

chi2_1 = chi2_SO_nu_anu_vec
chi2_1_mat = np.transpose(chi2_1.reshape((sin2_2_theta.size, Delta_m2_vec.size)))
chi2_BestFit = np.amin(chi2_1_mat)
chi2_BestFit_pos = np.where(chi2_1_mat == chi2_BestFit)
sin2_2_theta_best = sin2_2_theta[chi2_BestFit_pos[1][0]]
Delta_m2_best = Delta_m2_vec[chi2_BestFit_pos[0][0]]

fig, ax = plt.subplots()

CS1 = ax.contourf(sin2_2_theta, Delta_m2_vec, chi2_1_mat, levels = [chi2_CriVal_68, chi2_CriVal_95, chi2_CriVal_99], colors= ['red', 'green', 'blue'], alpha=0.6, extend='min')
CS1.levels = ["68%", "90%", "95%"];
#ax.clabel(CS1, CS1.levels, inline=1, fontsize=14)
ax.scatter(sin2_2_theta_best, Delta_m2_best, marker='*', s=200)

#MiniBooNE Analysis
ax.scatter(Nu_s22t_1_vec, Nu_Dm2_1_vec, label=r'MiniBooNE 1$\sigma$', s=5, c='pink')
ax.scatter(Nu_s22t_2_vec, Nu_Dm2_2_vec, label=r'MiniBooNE 90%\sigma$', s=5, c='magenta')
ax.scatter(Nu_s22t_3_vec, Nu_Dm2_3_vec, label='MiniBooNE 99%', s=5, c='black')
ax.scatter(0.918, 0.041, label=r'MiniBooNE Best Fit', marker='*', s=200, c='magenta')
ax.scatter(0.01, 0.4, label=r'MiniBooNE Best Fit 2', marker='*', s=200, c='black')
#MiniBooNE Analysis

#ax.set_title("MiniBooNE data 2020 - Sensitivity Levels \n Best Fit: "  r"$sin^2(2\theta)$ = %.03f , $\Delta m ^2$ = %.03f" "\n" r"$\chi^2$ = %.06f" %(sin2_2_theta_best, Delta_m2_best, chi2_BestFit), fontsize=16)
plt.xlabel(r'$sin^2(2\theta)$', fontsize=14)
plt.ylabel(r'$\Delta m ^2 (eV^2)$', fontsize=14)
plt.xlim(0.0, 0.05)
plt.ylim(0.0, 1.0)
plt.grid(which='both')
plt.savefig('plots/MBEA_DR20_RF_Contours_Compare_SO_Nu_aNu.png')

# End of analysis
