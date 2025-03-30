"""
MiniBooNE Excess Analysis - data Visualisation
Basic tasks:
1- Import the data
2- Plot The event ratios for predicted and observed flavours
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

#plt.rcParams.update({'font.size': 14})

# print to file
orig_stdout = sys.stdout
outfile = open('analysis/MBEA_data_Visualisation_output.txt', 'w')
sys.stdout = outfile

#Perform Chi-Square goodness of fit test and print results 
def chi2calculate(observed, expected):
    chi2 = 0.0
    chi2 = (expected-observed)**2/expected
    chi2 = chi2.sum()
    return chi2

def chi2testing(chi2, alpha, nd):
    critical_value = stats.chi2.ppf(1 - alpha,nd-1)
    print("Chi-square_Critical : {}".format(critical_value))
    print("Chi-square_Calculed : {}".format(chi2))
    print("p-value_Calculed : {}".format(1-stats.chi2.cdf(x=chi2,df=nd-1)))
    if chi2>critical_value:
        print("Reject the null hypothesis i.e, we can say that observed and expected frequencies are different")
        print("*"*70)
    else:
        print("fail to reject the null hypothesis i.e, observed and expected frequencies are same")
        print("*"*70)

#
# Import MiniBooNE data
#

DR18 = pd.read_csv("data/MB_DR2018.csv", sep=',') # Data Release 2018
DR20 = pd.read_csv("data/MB_DR2020.csv", sep=',') # Data Release 2020

#
# Data Release 2018
#
print("."*70)
print("Data Release 2018")
print("."*70)
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

# chi2 test Number of events for predicted vs observed
print("chi2 test Number of events ... predicted vs observed")
print("Electron Neutrinos ...")
chi2testing(chi2calculate(N_nue_obs_dr18, N_nue_pred_dr18), 0.05, len(N_nue_pred_dr18))
print("Muon Neutrinos ... ")
chi2testing(chi2calculate(N_numu_obs_dr18, N_numu_pred_dr18), 0.05, len(N_numu_pred_dr18))
print("Electron AntiNeutrinos ...")
chi2testing(chi2calculate(N_anue_obs_dr18, N_anue_pred_dr18), 0.05, len(N_anue_pred_dr18))
print("Muon AntiNeutrinos ...")
chi2testing(chi2calculate(N_anumu_obs_dr18, N_anumu_pred_dr18), 0.05, len(N_anumu_pred_dr18))

# Energy bin centers
E_points_dr18 = np.array([])
E_length_dr18 = np.array([])
for i in np.arange(len(E_bin_dr18)-1):
    E_points_dr18 = np.append(E_points_dr18, (E_bin_dr18[i]+E_bin_dr18[i+1])/2)
    E_length_dr18 = np.append(E_length_dr18, (E_bin_dr18[i+1]-E_bin_dr18[i])/2)
#print(E_points_dr18)

# Predicted flavour ratios
R_i_ab_nu_dr18 = N_numu_pred_dr18 / N_nue_pred_dr18
R_i_ab_nu2_dr18 = np.append(R_i_ab_nu_dr18, R_i_ab_nu_dr18[-1])

R_i_ab_anu_dr18 = N_anumu_pred_dr18 / N_anue_pred_dr18
R_i_ab_anu2_dr18 = np.append(R_i_ab_anu_dr18, R_i_ab_anu_dr18[-1])

# Observed flavour ratios
R_f_ab_nu_dr18 = N_numu_obs_dr18 / N_nue_obs_dr18

R_f_ab_anu_dr18 = N_numu_obs_dr18 / N_nue_obs_dr18

# chi2 test Ratios of flavour events for predicted vs observed
print("chi2 test Ratios of flavour events (numu / nue) ... predicted vs observed")
print("Neutrinos ...")
chi2testing(chi2calculate(R_f_ab_nu_dr18, R_i_ab_nu_dr18), 0.05, len(R_i_ab_nu_dr18))
print("AntiNeutrinos ...")
chi2testing(chi2calculate(R_f_ab_anu_dr18, R_i_ab_anu_dr18), 0.05, len(R_i_ab_anu_dr18))

# Plot Predicted vs Observed Flavour ratios
#fig, ax = plt.subplots(figsize=(8, 6))
fig, ax = plt.subplots()
ax.step(E_bin_dr18, R_i_ab_nu2_dr18, where = 'post', label='Predicted')
ax.errorbar(E_points_dr18, R_f_ab_nu_dr18, yerr=0.00, xerr=E_length_dr18, fmt='o', label='Observed')
plt.xlabel(r'$E_{\nu} \, [MeV]$', loc="right", rotation="horizontal")
plt.ylabel(r'$R_{\nu _{\mu} / \nu _{e}}$')
ax.set_xlim(E_bin_dr18[0], E_bin_dr18[-1])
plt.grid()
plt.legend(loc=1)
plt.savefig('plots/MiniBooNE_data_release_2018_Observed_vs_Predicted_Flavour_Ratios_nu_mode.png')

#fig, ax = plt.subplots(figsize=(8, 6))
fig, ax = plt.subplots()
ax.step(E_bin_dr18, R_i_ab_anu2_dr18, where = 'post', label='Predicted')
ax.errorbar(E_points_dr18, R_f_ab_anu_dr18, yerr=0.00, xerr=E_length_dr18, fmt='o', label='Observed')
plt.xlabel(r'$E_{\nu} \, [MeV]$', loc="right", rotation="horizontal")
plt.ylabel(r'$R_{\bar{\nu} _{\mu} / \bar{\nu} _{e}}$')
ax.set_xlim(E_bin_dr18[0], E_bin_dr18[-1])
plt.grid()
plt.legend(loc=1)
plt.savefig('plots/MiniBooNE_data_release_2018_Observed_vs_Predicted_Flavour_Ratios_anu_mode.png')

#
# Data Release 2020
#
print("."*70)
print("Data Release 2020")
print("."*70)
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

# chi2 test Number of events for predicted vs observed
print("chi2 test Number of events ... predicted vs observed")
print("Electron Neutrinos ...")
chi2testing(chi2calculate(N_nue_obs_dr20, N_nue_pred_dr20), 0.05, len(N_nue_pred_dr20))
print("Muon Neutrinos ...")
chi2testing(chi2calculate(N_numu_obs_dr20, N_numu_pred_dr20), 0.05, len(N_numu_pred_dr20))
print("Electron AntiNeutrinos ...")
chi2testing(chi2calculate(N_anue_obs_dr20, N_anue_pred_dr20), 0.05, len(N_anue_pred_dr20))
print("Muon AntiNeutrinos ...")
chi2testing(chi2calculate(N_anumu_obs_dr20, N_anumu_pred_dr20), 0.05, len(N_anumu_pred_dr20))

# Energy bin centers
E_points_dr20 = np.array([])
E_length_dr20 = np.array([])
for i in np.arange(len(E_bin_dr20)-1):
    E_points_dr20 = np.append(E_points_dr20, (E_bin_dr20[i]+E_bin_dr20[i+1])/2)
    E_length_dr20 = np.append(E_length_dr20, (E_bin_dr20[i+1]-E_bin_dr20[i])/2)
#print(E_points_dr20)

# Predicted flavour ratios
R_i_ab_nu_dr20 =  N_numu_pred_dr20 / N_nue_pred_dr20
R_i_ab_nu2_dr20 = np.append(R_i_ab_nu_dr20, R_i_ab_nu_dr20[-1])

R_i_ab_anu_dr20 =  N_anumu_pred_dr20 / N_anue_pred_dr20
R_i_ab_anu2_dr20 = np.append(R_i_ab_anu_dr20, R_i_ab_anu_dr20[-1])

# Observed flavour ratios
R_f_ab_nu_dr20 = N_numu_obs_dr20 / N_nue_obs_dr20

R_f_ab_anu_dr20 = N_numu_obs_dr20 / N_nue_obs_dr20

# chi2 test Ratios of flavour events for predicted vs observed
print("chi2 test Ratios of flavour events (numu / nue) ... predicted vs observed")
print("Neutrinos ...")
chi2testing(chi2calculate(R_f_ab_nu_dr20, R_i_ab_nu_dr20), 0.05, len(R_i_ab_nu_dr20))
print("AntiNeutrinos ...")
chi2testing(chi2calculate(R_f_ab_anu_dr20, R_i_ab_anu_dr20), 0.05, len(R_i_ab_anu_dr20))

# Plot Predicted vs Observed Flavour ratios
#fig, ax = plt.subplots(figsize=(8, 6))
fig, ax = plt.subplots()
ax.step(E_bin_dr20, R_i_ab_nu2_dr20, where = 'post', label='Predicted')
ax.errorbar(E_points_dr20, R_f_ab_nu_dr20, yerr=0.00, xerr=E_length_dr20, fmt='o', label='Observed')
plt.xlabel(r'$E_{\nu} \, [MeV]$', loc="right", rotation="horizontal")
plt.ylabel(r'$R_{\nu _{\mu} / \nu _{e}}$')
ax.set_xlim(E_bin_dr20[0], E_bin_dr20[-1])
plt.grid()
plt.legend(loc=1)
plt.savefig('plots/MiniBooNE_data_release_2020_Observed_vs_Predicted_Flavour_Ratios_nu_mode.png')

#fig, ax = plt.subplots(figsize=(8, 6))
fig, ax = plt.subplots()
ax.step(E_bin_dr20, R_i_ab_anu2_dr20, where = 'post', label='Predicted')
ax.errorbar(E_points_dr20, R_f_ab_anu_dr20, yerr=0.00, xerr=E_length_dr20, fmt='o', label='Observed')
plt.xlabel(r'$E_{\nu} \, [MeV]$', loc="right", rotation="horizontal")
plt.ylabel(r'$R_{\bar{\nu} _{\mu} / \bar{\nu} _{e}}$')
ax.set_xlim(E_bin_dr20[0], E_bin_dr20[-1])
plt.grid()
plt.legend(loc=1)
plt.savefig('plots/MiniBooNE_data_release_2020_Observed_vs_Predicted_Flavour_Ratios_anu_mode.png')

#
# Compaire Data Release 2018 vs 2020
#

# chi2 test predicted vs observed
print("Observed Electron Neutrinos ... DR18 vs DR20")
chi2testing(chi2calculate(N_nue_obs_dr18, N_nue_obs_dr20), 0.05, len(N_nue_obs_dr18))
print("Observed Muon Neutrinos ... DR18 vs DR20")
chi2testing(chi2calculate(N_numu_obs_dr18, N_numu_obs_dr20), 0.05, len(N_numu_obs_dr18))

# Plot Predicted vs Observed Flavour ratios ... DR18 vs DR20
#fig, ax = plt.subplots(figsize=(8, 6))
fig, ax = plt.subplots()
ax.step(E_bin_dr18, R_i_ab_nu2_dr18, where = 'post', label='DR18 Predicted')
ax.errorbar(E_points_dr18, R_f_ab_nu_dr18, yerr=0.00, xerr=E_length_dr20, fmt='o', label='DR18 Observed')
ax.step(E_bin_dr20, R_i_ab_nu2_dr20, where = 'post', label='DR20 Predicted')
ax.errorbar(E_points_dr20, R_f_ab_nu_dr20, yerr=0.00, xerr=E_length_dr20, fmt='o', label='DR20 Observed')
plt.xlabel(r'$E_{\nu} \, [MeV]$', loc="right", rotation="horizontal")
plt.ylabel(r'$R_{\nu _{\mu} / \nu _{e}}$')
ax.set_xlim(E_bin_dr20[0], E_bin_dr20[-1])
plt.grid()
plt.legend(loc=1)
plt.savefig('plots/MiniBooNE_data_release_2018_vs_2020_Observed_vs_Predicted_Flavour_Ratios_nu_mode.png')

# End of analysis