#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 10:06:10 2023

@author: louis
"""


#%% Importing Modules
'''
This cell's purpose is to import the modules that will be needed later in the code.
'''

import numpy as np #Importing NumPy.
import matplotlib.pyplot as plt #Importing MatPlotLib.
import scipy.stats as st
import scipy.integrate as integrate
plt.show()
np.random.seed(1)

N_b = 10e5 # Number of background events, used in generation and in fit.
b_tau = 30. # Spoiler.

def generate_data(n_signals = 400):
    ''' 
    Generate a set of values for signal and background. Input arguement sets 
    the number of signal events, and can be varied (default to higgs-like at 
    announcement). 
    
    The background amplitude is fixed to 9e5 events, and is modelled as an exponential, 
    hard coded width. The signal is modelled as a gaussian on top (again, hard 
    coded width and mu).
    '''
    vals = []
    vals += generate_signal( n_signals, 125., 1.5)
    vals += generate_background( N_b, b_tau)
    return vals


def generate_signal(N, mu, sig):
    ''' 
    Generate N values according to a gaussian distribution.
    '''
    return np.random.normal(loc = mu, scale = sig, size = N).tolist()


def generate_background(N, tau):
    ''' 
    Generate N values according to an exp distribution.
    '''
    return np.random.exponential(scale = tau, size = int(N)).tolist()


def get_B_chi(vals, mass_range, nbins, A, lamb):
    ''' 
    Calculates the chi-square value of the no-signal hypothesis (i.e background
    only) for the passed values. Need an expectation - use the analyic form, 
    using the hard coded scale of the exp. That depends on the binning, so pass 
    in as argument. The mass range must also be set - otherwise, its ignored.
    '''
    bin_heights, bin_edges = np.histogram(vals, range = mass_range, bins = nbins)
    half_bin_width = 0.5*(bin_edges[1] - bin_edges[0])
    ys_expected = get_B_expectation(bin_edges + half_bin_width, A, lamb)
    chi = 0

    # Loop over bins - all of them for now. 
    for i in range( len(bin_heights) ):
        chi_nominator = (bin_heights[i] - ys_expected[i])**2
        chi_denominator = ys_expected[i]
        chi += chi_nominator / chi_denominator
    
    return chi # B has 2 parameters.


def get_SB_chi(vals, mass_range, nbins, A, lamb, mu, sig, signal_amp):
    ''' 
    Calculates the chi-square value of the background + signal hypothesis for the passed values. Need an expectation - use the analyic form, 
    using the hard coded scale of the exp. That depends on the binning, so pass 
    in as argument. The mass range must also be set - otherwise, its ignored.
    '''
    bin_heights, bin_edges = np.histogram(vals, range = mass_range, bins = nbins)
    half_bin_width = 0.5*(bin_edges[1] - bin_edges[0])
    ys_expected = get_SB_expectation(bin_edges + half_bin_width, A, lamb, mu, sig, signal_amp)
    chi = 0

    # Loop over bins - all of them for now. 
    for i in range( len(bin_heights) ):
        chi_nominator = (bin_heights[i] - ys_expected[i])**2
        chi_denominator = ys_expected[i]
        chi += chi_nominator / chi_denominator
    
    return chi # SB has 5 parameters.


def get_B_expectation(xs, A, lamb):
    ''' 
    Return a set of expectation values for the background distribution for the 
    passed in x values. 
    '''
    return [A/lamb*np.exp(-x/lamb) for x in xs]


def signal_gaus(x, mu, sig, signal_amp):
    return signal_amp/(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)


def get_SB_expectation(xs, A, lamb, mu, sig, signal_amp):
    ys = []
    for x in xs:
        ys.append(A/lamb*np.exp(-x/lamb) + signal_gaus(x, mu, sig, signal_amp))
    return ys




print("Modules imported.")
#%% Functions

def exponential(x, a, b): #A general exponential function.
    #In this case, b = -1/lambda
    func = a * np.exp( b * x )
    return func

def gaussian(x, amp, mean, std): #A general gaussian function.
    func = amp * np.exp( ( ( x - mean )**2 ) / ( 2 * (std**2) ) )
    return func
    
def distribution(x, a, b, amp, mean, std): #A combination of the exponential and gaussian function, which should fit the distribution we have.
    func = exponential(x, a, b) + gaussian(x, amp, mean, std)
    return func

def rangeOfData(data): #Finds the range of the data inputted.
    return (max(data) - min(data))

def histogram_x_error(bin_bounds): #Finds the error on each midpoint representing each bin. Was very wrong.
    x_err = np.array([])
    for i in range(1, len(bin_bounds)):
        error = ( bin_bounds[i] - bin_bounds[i-1] ) / 2
        x_err = np.append(x_err, error)
    return x_err

def histogram_x_error2(bin_bounds, hist_data): #Finds actual error by using standard deviation from the mean of each bin. 
    x_err = np.array([])
    for i in range(0, len(bin_bounds) - 1):
        midpoint = ( bin_bounds[i] + bin_bounds[i + 1] ) / 2 #Creates a midpoint for the bin
        err_data = [num for num in hist_data if (num >= bin_bounds[i] and num < bin_bounds[i+1])]
        err = np.std(err_data - midpoint)
        x_err = np.append(x_err, err)
    print(x_err)
    return x_err

def histogram_y_error(bin_vals): #Finds histogram y uncertainty
    y_err = np.array([])
    for val in bin_vals:
        err = np.sqrt(val)
        y_err = np.append(y_err, err)
    return y_err

print("Functions defined.")
#%% Generating Data
'''
This cell's purpose is to generate data to be used later.
'''

data = generate_data(n_signals=400) 
#Generates data (with units GeV) to be processed later. Func takes one input and even defaults to 400 if there is no input.

print("Data generated.")
#%% Plotting Histogram
'''
This cell's purpose is to plot a histogram of the data generated. 
The resulting plot allows us to see a rough distribution of the energy values.
'''

num_bins = 30 #Variable which determines the number of bins used in histogram.

'''
The PDF says use num_bins = 30. Anything more than 40 becomes quite unusable but lower than 30 gives interesting results. Try 15 and 20. But play around with this variable if you want.
'''

hist_range = [104, 155] #104 GeV to 155 GeV as given is the document.

bin_heights, bin_edges, patches = plt.hist(data, range = hist_range, bins = num_bins) #Creates histogram, with (num_bins) bins, showing energy values from 104 GeV to 155 GeV.
plt.xlabel('Rest Mass [GeV]', fontsize='x-large', fontfamily='times new roman')
plt.ylabel('Number of Entries', fontsize='x-large', fontfamily='times new roman')
plt.show()

print('Histogram plotted.')
#%% Plotting Histogram Midpoints

'''
Now uncertainties must be calculated. x-uncertainty depends is the width of the bin. 
y-uncertainty is the square root of the frequency of each bin.
'''

x_error = histogram_x_error(bin_edges) #Using function defined above. The correct solution is the first error function.
y_error = histogram_y_error(bin_heights) #Uses function defined above, derived from the standard y-error used for histograms.


'''
Bin heights are being plotted as it can give a clearer indication of trends in the data.
'''

midpoints = np.array([]) #Creates an array which will store the midpoints of each bin
for pos in range(0, len(bin_edges) - 1):
    midpoint = ( bin_edges[pos] + bin_edges[pos + 1] ) / 2 #Creates a midpoint for the bin
    midpoints = np.append(midpoints, midpoint) #Adds midpoint to the list of midpoints

'''
Second histogram plotted as it shows an alternative view with only points, rather than bars.
'''
yticks = np.arange(0, 2001, 200, 'int16')
xticks = np.arange(105, 165, 5)
ylabels = np.array([])
for i in range(0, len(yticks)):
    ylabels = np.append(ylabels, yticks[i] if (i%2) == 0 else '')
xlabels = np.array([])
for i in range(0, len(xticks)):
    xlabels = np.append(xlabels, xticks[i] if (i%2) == 1 else '')

ylabels[0] = int(0)

#plt.plot(midpoints, bin_heights, 'o', color='orange')
plt.errorbar(midpoints, bin_heights, xerr = x_error, yerr = y_error, fmt = 'o', ecolor='black', capsize=2, label='Data', markersize=5)
plt.legend()
plt.xlabel('Rest Mass [GeV]', fontsize='x-large', fontfamily='times new roman')
plt.ylabel('Number of Entries', fontsize='x-large', fontfamily='times new roman')
plt.xticks(xticks,xlabels, fontsize='large')
plt.yticks(yticks,ylabels, fontsize='large')
plt.xlim((102.5, 157.5))
plt.ylim((0,2050))
plt.show()

print("Test histograms plotted.")

#%% Test

# %% Background Signal



A=1 # let normalisation factor be 1 for now  
# create a set of data below 120GeV to be used for background paramterisation 
background_freq = []
background_mass = []
for i,x in enumerate(midpoints):
    if x<120:
        background_freq.append(bin_heights[i])
        background_mass.append(x)
        
        
        
print("background data collected, there are",len(background_freq), "points.")


# # Method 1 : Maximum Likelihood 
# I am working on making an actual max log plot, it is almost there, feel free to look it over

# def log_likelihood(lambda_val, data):
#     n = len(data)
#     log_likelihood = n * np.log(lambda_val) - lambda_val * np.sum(data)
#     return log_likelihood

# lamb_test = np.linspace(1,100,100)
# L_est =[]
# lamb_0 = 30
# background =get_B_expectation(background_mass,1,lamb_0)

# likelihoods = [log_likelihood(lmbda, background) for lmbda in lamb_test]
# float(lamb_test[np.where(likelihoods==np.max(likelihoods))])

# print("From Likelihood graph, lambda =",lamb_test)
# plt.plot(lamb_test, likelihoods)
# plt.xlabel('Lambda')
# plt.ylabel('Negative Log-Likelihood')
# plt.title('Likelihood Plot')    
# print("Likelihood Plot plotted, Lambda estimate found")
# plt.show()



# For now, we can simply use that from max likelihood, that lambda is the mean 

lamb_estimate = sum(data) / len(data)




# define background signal

def get_B_expect(xs, A, lamb=lamb_estimate):
    ''' 
    Return a set of expectation values for the background distribution for the 
    passed in x values. 
    '''
    return A/lamb*np.exp(-xs/lamb)


# finding A



# Match the area underneath the graph for both by using the bins from 104-120GeV
A = np.linspace(1000000,2000000,1000) # range found by trial and error
area= np.sum(np.array(background_freq)*(16/30)) 
possible_values = []
for i,x in enumerate(A):

    estimate = np.sum(np.array(get_B_expect(np.array(background_mass),x))*16/30)

    if (estimate/area)<1.1 and (estimate/area)>0.9:


            possible_values.append(x)
# Check if a value was found
if isinstance(possible_values[0],float):
    print("A estimate found")
else:
    print("No A estimate found")


# All the values are extremely similar, so the mean adds no change
A_estimate = np.mean(possible_values)
# A_estimate = np.sum(background_mass) / np.sum(exponential_distribution(background_mass, 1, lamb_estimate))
# print("From Method 1:\n Lambda =",lamb_estimate,"A = ",A_estimate)


# THEREFORE from method 1 the curve looks like:
yticks = np.arange(0, 2001, 200, 'int16')
xticks = np.arange(105, 165, 5)
ylabels = np.array([])
for i in range(0, len(yticks)):
    ylabels = np.append(ylabels, yticks[i] if (i%2) == 0 else '')
xlabels = np.array([])
for i in range(0, len(xticks)):
    xlabels = np.append(xlabels, xticks[i] if (i%2) == 1 else '')
    
ylabels[0] = int(0)

#plt.errorbar(midpoints, bin_heights, xerr = x_error, yerr = y_error, fmt = 'none')
mass_range = np.arange(104,170,1)
plt.plot(mass_range, get_B_expect(mass_range, A_estimate, lamb_estimate),":",label="Background Estimate")
#plt.plot(midpoints, bin_heights, 'o', color='orange')
plt.errorbar(midpoints, bin_heights, xerr = x_error, yerr = y_error, fmt = 'o', ecolor='black', capsize=2, label="Data", markersize=5)
#plt.errorbar(midpoints, bin_heights, xerr = x_error, yerr = y_error, fmt = 'x',label="Data")
plt.legend()
plt.xlabel('Rest Mass [GeV]', fontsize='x-large', fontfamily='times new roman')
plt.ylabel('Number of Entries', fontsize='x-large', fontfamily='times new roman')
plt.xticks(xticks,xlabels, fontsize='large')
plt.yticks(yticks,ylabels, fontsize='large')
plt.xlim((102.5, 157.5))
plt.ylim((0,2050))
print("Estimated Lambda = ", lamb_estimate)
print("Estimated A value =", A_estimate)


 





# Method 2
# stating the function to remind myself what I am working with
def b_chi(y,x,lamb,A,y_err=y_error,x_err=x_error):
    x=np.array(x)
    x_err = np.array(x_err)[:len(x)]
    y_err = np.array(y_err)[:len(y)] #Added by Adam to fix error generated by him making the y_error an array in his section of the code.  
    y_fit = get_B_expect(x,A,lamb)
    total_unc = np.sqrt(y_err**2+x_err**2)
    residuals = (y-y_fit)/total_unc
    chi_sq = np.sum(residuals**2)
    
    return chi_sq
    


lamb_guess = np.linspace(30,31,100)
A_guess = np.linspace(1609000,1610000,1000)
chi = 1e10     #set initial chi squared to soemthing ridiculously high
# 2D search to find min chi squared
for i,x in enumerate(lamb_guess): 

    for j,y  in enumerate(A_guess):

       chi_test = b_chi(background_freq, background_mass, x, y,y_error,x_error)

       if  chi_test < chi:   #if the current chi squared value is lower than the one before, replace it, if not keep it. After all elements are searched, this should result in the lowest chi squared

           chi = chi_test

           lamb_chi_estimate = x

           A_chi_estimate = y

print("min Chi-squared value is",chi)
print("From the Chi squared method,")
print("Estimated Lambda =", lamb_chi_estimate)
print("Estimated A value =", A_chi_estimate)
# Chi square and lambda values from both are very similar.


#%%
#part 4a starts here
#generate data in full range
A=1 # let normalisation factor be 1 for now  
# create a set of data below 120GeV to be used for background paramterisation 
freq = []
mass = []
for i,x in enumerate(midpoints):
    if x<160:
        freq.append(bin_heights[i])
        mass.append(x)
        
print("data collected, there are",len(freq), "points.") #should be same as number of bins

#compute min chi square for full range data, same code as previous part:)
def b_chi(y,x,lamb,A,y_err=y_error,x_err=x_error):
    x=np.array(x)
    x_err = np.array(x_err)[:len(x)]
    y_err = np.array(y_err)[:len(y)] #Added by Adam to fix error generated by him making the y_error an array in his section of the code.  
    y_fit = get_B_expect(x,A,lamb)
    total_unc = np.sqrt(y_err**2+x_err**2)
    residuals = (y-y_fit)/total_unc
    chi_sq = np.sum(residuals**2)
    
    return chi_sq

lamb_guess = np.linspace(30,31,100)
A_guess = np.linspace(1609000,1610000,1000)
chi = 1e10     #set initial chi squared to soemthing ridiculously high
# 2D search to find min chi squared
for i,x in enumerate(lamb_guess): 

    for j,y  in enumerate(A_guess):

       chi_test = b_chi(freq, mass, x, y,y_error,x_error)

       if  chi_test < chi:   #if the current chi squared value is lower than the one before, replace it, if not keep it. After all elements are searched, this should result in the lowest chi squared

           chi = chi_test

           full_lamb_chi_estimate = x

           full_A_chi_estimate = y

print("min Chi-squared value in full range is",chi)
print("From the Chi squared method,")
print("Estimated Lambda in full range=", full_lamb_chi_estimate)
print("Estimated A value in full range =", full_A_chi_estimate)
#higher chi value, thus a worse fit

#compute chi value in the entire range assuming there is only background radiation

# def get_B_chi(vals, mass_range, nbins, A, lamb):
#     ''' 
#     Calculates the chi-square value of the no-signal hypothesis (i.e background
#     only) for the passed values. Need an expectation - use the analyic form, 
#     using the hard coded scale of the exp. That depends on the binning, so pass 
#     in as argument. The mass range must also be set - otherwise, its ignored.
#     '''
#     bin_heights, bin_edges = np.histogram(vals, range = [104,155], bins = 30)
#     half_bin_width = 0.5*(bin_edges[1] - bin_edges[0])
#     ys_expected = get_B_expectation(bin_edges + half_bin_width, A, lamb)
#     chi = 0

#     # Loop over bins - all of them for now. 
#     for i in range( len(bin_heights) ):
#         chi_nominator = (bin_heights[i] - ys_expected[i])**2
#         chi_denominator = ys_expected[i]
#         chi += chi_nominator / chi_denominator
    
#     return chi/float(nbins-2) # B has 2 parameters.




chi_full_range=b_chi(freq,mass,lamb_chi_estimate,A_chi_estimate,y_error,x_error)
print("Chi-squared value in full range is",chi_full_range)


#calculate p-value for signal+background
from scipy.stats import chi2
#dof is 30-2=28
p_value_full_range = 1 - st.chi2.cdf(chi_full_range, 28)
print('p-value is',p_value_full_range)
#since this is significantly smaller than 0.005, we can reject the null hypothesis

#for reference, calculate p-value for background only
p_value_bg = 1 - st.chi2.cdf(chi, 9) #9 bins were used
print('p-value for background only radiation is',p_value_bg)
#%%
#4b starts here
def b_chi(y,x,lamb,A,y_err=y_error,x_err=x_error):
    x=np.array(x)
    x_err = np.array(x_err)[:len(x)]
    y_err = np.array(y_err)[:len(y)] #Added by Adam to fix error generated by him making the y_error an array in his section of the code.  
    y_fit = get_B_expect(x,A,lamb)
    total_unc = np.sqrt(y_err**2+x_err**2)
    residuals = (y-y_fit)/total_unc
    chi_sq = np.sum(residuals**2)
    
    return chi_sq


           
           
bgchi=[]
for i in range(0,50):
    data=generate_data(0)
    bin_heights, bin_edges, patches = plt.hist(data, range = [104,155], bins = 30) #Creates histogram, with (num_bins) bins, showing energy values from 104 GeV to 155 GeV.
    
    x_error = histogram_x_error(bin_edges) #Using function defined above. The correct solution is the first error function.
    y_error = histogram_y_error(bin_heights) #Uses function defined above, derived from the standard y-error used for histograms.

    midpoints = np.array([]) #Creates an array which will store the midpoints of each bin
    for pos in range(0, len(bin_edges) - 1):
        midpoint = ( bin_edges[pos] + bin_edges[pos + 1] ) / 2 #Creates a midpoint for the bin
        midpoints = np.append(midpoints, midpoint) #Adds midpoint to the list of midpoints
    
    lamb_guess = np.linspace(30,31,100)
    A_guess = np.linspace(1609000,1610000,1000)
    chi = 1e10     #set initial chi squared to soemthing ridiculously high
    # 2D search to find min chi squared
    for i,x in enumerate(lamb_guess): 

        for j,y  in enumerate(A_guess):

           chi_test = b_chi(bin_heights, midpoints, x, y,y_error,x_error)

           if  chi_test < chi:   #if the current chi squared value is lower than the one before, replace it, if not keep it. After all elements are searched, this should result in the lowest chi squared

               chi = chi_test

                
               
    bgchi.append(chi)

#%%
plt.hist(bgchi, bins = 30)
plt.title('chi-squared value distribution')
plt.xlabel('frequency')
plt.ylabel('chi-squared value')
plt.show()

#%%
#4c starts here
from scipy.stats import chi2

# Define significance level and degrees of freedom
significance_level = 0.05
df = 28  # degrees of freedom

# Find the critical value
critical_value = chi2.ppf(1 - significance_level, df)

print("Critical value:", critical_value)   


def b_chi(y,x,lamb,A,y_err=y_error,x_err=x_error):
    x=np.array(x)
    x_err = np.array(x_err)[:len(x)]
    y_err = np.array(y_err)[:len(y)] #Added by Adam to fix error generated by him making the y_error an array in his section of the code.  
    y_fit = get_B_expect(x,A,lamb)
    total_unc = np.sqrt(y_err**2+x_err**2)
    residuals = (y-y_fit)/total_unc
    chi_sq = np.sum(residuals**2)
    
    return chi_sq

#find the least difference between critical chi and actual chi
difference=[]
for i in range(220,230):
    data=generate_data(i)
    bin_heights, bin_edges, patches = plt.hist(data, range = [104,155], bins = 30) #Creates histogram, with (num_bins) bins, showing energy values from 104 GeV to 155 GeV.
    
    x_error = histogram_x_error(bin_edges) #Using function defined above. The correct solution is the first error function.
    y_error = histogram_y_error(bin_heights) #Uses function defined above, derived from the standard y-error used for histograms.

    midpoints = np.array([]) #Creates an array which will store the midpoints of each bin
    for pos in range(0, len(bin_edges) - 1):
        midpoint = ( bin_edges[pos] + bin_edges[pos + 1] ) / 2 #Creates a midpoint for the bin
        midpoints = np.append(midpoints, midpoint) 
    chitest=b_chi(bin_heights,midpoints,lamb_chi_estimate,A_chi_estimate,y_error,x_error)
    diff=chitest-critical_value
    difference.append(diff)
    print(chitest)
    
    
print(min(difference))
#by checking variable explorer we can find this minimum difference occurs when i=230
#therefore when signal amplitude=230, the p-value gives the most close to 0.05

#%% Signal Estimation (Part 5)

#This is the result for 5(a). ***NB: dof = nbins - 5 (we estimate 3 extra parameters for the Gaussian signal)
chiSB = get_SB_chi(data, hist_range, num_bins, A_chi_estimate, lamb_chi_estimate, 125, 1.5, 700)
print("Signal + Background Chi-Squared value =", chiSB)

p_value_part5 = 1 - st.chi2.cdf(chiSB, 25) #30-5
print("The corresponding p-value =", p_value_part5)

#This is the result for 4(a) but repeated for easy comparison
chiB = get_B_chi(data, hist_range, num_bins, A_chi_estimate, lamb_chi_estimate)
print("Background Chi-Squared value =", chiB)

#Below is part 5(b). ***NB: dof = nbins - 4 (because we are no longer estimating the mean of the Gaussian)
K = 100 #Number of signal masses to test. This should not be too large as each iteration of the for loop takes a decent amount of time to run, so even K=100 takes a while
test_masses = np.linspace(104, 155, K)
chi_tms = np.zeros(len(test_masses))

for i, x in enumerate(test_masses):
    tmp = get_SB_chi(data, hist_range, num_bins, A_chi_estimate, lamb_chi_estimate, x, 1.5, 700)
    chi_tms[i] = tmp
    #print(x, tmp)

plt.title("Chi-squared Values for a Range of Test Signal Masses")
plt.xlabel("Test Signal Mass (GeV)")
plt.ylabel("Chi-squared Value")
plt.plot(test_masses, chi_tms)
plt.show()

#The idea is that we are most confident (where chi-squared is smallest, see sudden dip in the graph) of the presence of a signal at the test mass of 125 GeV --> we can deduce the mass of Higgs Boson to be 125 GeV 


































