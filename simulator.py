# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 09:45:37 2021

@author: Laptop
"""


import random
import numpy as np

from matplotlib import pyplot as plt

### Function for choosing policies: different policies are indexed
# num_med: the number of different types of medicines; 
# sys: amount of medicines in stock; 
# cap: the capacity of the machine; 
# ind: the index of policy selected; 
def POLICY(num_med, sys, cap, ind):
    decision = np.zeros(num_med+1)
    if ind == 1: # the simplest policy for testing the program
        for m in range(1, M+1):
            if sys[m] <= 3:   # If the number of one type of medicine in stock <=3 then replenish it
                decision[m] = cap - sys[m] 
        return decision


### Function for generating the arrival process for each medicine throughout the planning horizon
# num_med: the number of different types of medicines; 
# th: the planning horizon;
# par: the parameters of the arrival processes of each medicine; (fit by input modeling)
def GENERATE_ARRIVAL(num_med, th, par):
    arrival = np.zeros([num_med+1, th+1])
    for i in range(1, num_med+1):
        if par[i] == 0:
            continue
        t = 0
        j = 1
        while t <= th:
            interarrival = random.expovariate(par[i]) # to be determined after the input modeling
            if interarrival <= 1:
                arrival[i][j] = t + 1
            else:
                arrival[i][j] = round(t + interarrival)  
                
            t = arrival[i][j]
            if arrival[i][j] > th:
                arrival[i][j] = 0
            j = j + 1
        
    return arrival


### Function for generating the demand coming at some time period
# num_med: the number of different types of medicines; 
# arr_series: the demand arrival time for each medicines;
# ti: current time period;
def DEMAND(num_med, arr_series, ti):
    demand = np.zeros(num_med+1)
    if ti == 0:
        return demand
    for m in range(1, num_med+1):
        if ti in arr_series[m]:
            demand[m] += random.randint(1,10)  # to be revised after the input modeling
            
    return demand




#T = int(input('Please enter the planning horizon:'))
#M = int(input('Please enter the number of medicines:'))
#C = int(input('Please enter the capacity of a single medicine on the machine:'))
#Pol = int(input('Please enter the index of policy:'))
T = 100
M = 10
C = 10
Pol = 1


# Initialization
I_t = np.zeros([M+1, T+1]) # the status of each medicine in stock throughout the planning horizon
D_t = np.zeros([M+1, T+1]) # the demand of each medicine throughout the planning horizon
R_t = np.zeros([M+1, T+1]) # the replenishment decision of each medicine throughout the planning horizon
P = np.zeros([M+1]) # the parameters of the arrival process of each medicine

P[1], P[2] = 0.2, 0.5

Success_Delivery = np.zeros([M+1, T+1]) # store the time period that success delivery occurs
Lack_Event = np.zeros([M+1, T+1]) # store the time period that lack event occurs
LACK = np.array([]) # store the time, medicine, and lack amount of all "lack events"
lack_t = np.zeros(T+1) # store the lack amount in each time period 
arrival_time_series = GENERATE_ARRIVAL(M, T, P) # generate arrival processes of each medicine


# System status at time 0
for i in range(1, M+1):
    I_t[i][0] = C

    
j, k = np.zeros(M+1), np.zeros(M+1)
# The planning horzion starts
for t in range(T): 
    D_t[1:(M+1), t] = DEMAND(M, arrival_time_series, t)[1:]  # generate the demand
    R_t[1:(M+1), t] = POLICY(M, I_t[:, t], C, Pol)[1:]  # apply policy to decide which medicines need replenishment, and how much
    I_t[1:(M+1), t] = I_t[1:(M+1), t]+R_t[1:(M+1), t] # system status updated by replenishment
    # system status updated by medicine delivery, and 'lack events' are recorded
    for m in range(1, M+1):
        if D_t[m][t] == 0:
            continue
        if I_t[m][t] > D_t[m][t]:  # able to deliver medicine m
            I_t[m][t+1] = I_t[m][t] - D_t[m][t]
            Success_Delivery[m][int(j[m])] = t
            j[m] = j[m] + 1
        else:                      # not able to deliver medicine m
            I_t[m][t+1] = 1
            LACK = np.append(LACK, [t, m, D_t[m][t]-(I_t[m][t]-1)])
            lack_t[t] = lack_t[t] + (D_t[m][t]-(I_t[m][t]-1))
            Lack_Event[m][int(k[m])] = t
            k[m] = k[m] + 1

# Performance metrics
lack_times = int(len(LACK)/3)
total_lack_amount = sum(lack_t)



# Plots
plt.plot(range(T+1), I_t[1], linestyle='-')
plt.xlabel('Time')
plt.ylabel('The number of medicine 1 in stock')
plt.title('The track of medicine 1')
    
    
    
    
    
    
    
    
    
    
    
    