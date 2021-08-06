# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 09:45:37 2021

@author: Laptop
"""


import random
import math
import numpy as np
import pandas as pd

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
                decision[m] = cap[m] - sys[m] 
        return decision


### Function for generating the arrival process for each medicine throughout the planning horizon
# num_med: the number of different types of medicines; 
# th: the planning horizon;
# par: the parameters of the arrival processes of each medicine; (fit by input modeling)
# def GENERATE_ARRIVAL(num_med, th, par):
#     arrival = np.zeros([num_med+1, th+1])
#     for i in range(1, num_med+1):
#         if par[i] == 0:
#             continue
#         t = 0
#         j = 1
#         while t <= th:
#             interarrival = random.expovariate(par[i]) # to be determined after the input modeling
#             if interarrival <= 1:
#                 arrival[i][j] = t + 1
#             else:
#                 arrival[i][j] = round(t + interarrival)  
                
#             t = arrival[i][j]
#             if arrival[i][j] > th:
#                 arrival[i][j] = 0
#             j = j + 1
        
#     return arrival


# ### Function for generating the demand coming at some time period
# # num_med: the number of different types of medicines; 
# # arr_series: the demand arrival time for each medicines;
# # ti: current time period;
# def DEMAND(num_med, arr_series, ti):
#     demand = np.zeros(num_med+1)
#     if ti == 0:
#         return demand
#     for m in range(1, num_med+1):
#         if ti in arr_series[m]:
#             demand[m] += random.randint(1,10)  # to be revised after the input modeling
            
#     return demand
def Sample_ECDF(p):
    parray = []
    for q in range(len(p)):
        parray.append(eval(p[q]))
    
        
    rand = random.random()
    for i in range(len(parray)):
        if sum(parray[0:i]) < rand and rand < sum(parray[0:(i+1)]):
            return i
    
    
def DEMAND(num_med, ti, EDmatrix):
    dv = np.zeros(num_med+1)
    if ti == 24:
        return dv
    for m in range(1, num_med+1):
        ep = EDmatrix[m][ti]
        dv[m] = Sample_ECDF(ep)
        
    return dv






T = 360    # the planning horizon
M = 10     # the number of medicines
C = [0]+[15]*M    # the capacity of each medicine on the machine
Pol = 1      # the index of policy



### Read empirical distribution
M_dic = [0, '1178', '2736', '6647', '7778', '7796']
ECDFM = [0]
ecdf1178 = np.loadtxt('D:/RA 2021/药房库存project/data analysis/7.29-8.4/epd_1178.csv', dtype=np.str, delimiter=',')
ecdf1178 = np.insert(ecdf1178, 24, np.zeros([1,np.size(ecdf1178,1)]),axis=0)
ECDFM += [ecdf1178]
ecdf2736 = np.loadtxt('D:/RA 2021/药房库存project/data analysis/7.29-8.4/epd_2736.csv', dtype=np.str, delimiter=',')
ecdf2736 = np.insert(ecdf2736, 24, np.zeros([1,np.size(ecdf2736,1)]),axis=0)
ECDFM += [ecdf2736]
ecdf6297 = np.loadtxt('D:/RA 2021/药房库存project/data analysis/7.29-8.4/epd_6297.csv', dtype=np.str, delimiter=',')
ecdf6297 = np.insert(ecdf6297, 24, np.zeros([1,np.size(ecdf6297,1)]),axis=0)
ECDFM += [ecdf6297]
ecdf6647 = np.loadtxt('D:/RA 2021/药房库存project/data analysis/7.29-8.4/epd_6647.csv', dtype=np.str, delimiter=',')
ecdf6647 = np.insert(ecdf6647, 24, np.zeros([1,np.size(ecdf6647,1)]),axis=0)
ECDFM += [ecdf6647]
ecdf6923 = np.loadtxt('D:/RA 2021/药房库存project/data analysis/7.29-8.4/epd_6923.csv', dtype=np.str, delimiter=',')
ecdf6923 = np.insert(ecdf6923, 24, np.zeros([1,np.size(ecdf6923,1)]),axis=0)
ECDFM += [ecdf6923]
ecdf6936 = np.loadtxt('D:/RA 2021/药房库存project/data analysis/7.29-8.4/epd_6936.csv', dtype=np.str, delimiter=',')
ecdf6936 = np.insert(ecdf6936, 24, np.zeros([1,np.size(ecdf6936,1)]),axis=0)
ECDFM += [ecdf6936]
ecdf7633 = np.loadtxt('D:/RA 2021/药房库存project/data analysis/7.29-8.4/epd_7633.csv', dtype=np.str, delimiter=',')
ecdf7633 = np.insert(ecdf7633, 24, np.zeros([1,np.size(ecdf7633,1)]),axis=0)
ECDFM += [ecdf7633]
ecdf7770 = np.loadtxt('D:/RA 2021/药房库存project/data analysis/7.29-8.4/epd_7770.csv', dtype=np.str, delimiter=',')
ecdf7770 = np.insert(ecdf7770, 24, np.zeros([1,np.size(ecdf7770,1)]),axis=0)
ECDFM += [ecdf7770]
ecdf7778 = np.loadtxt('D:/RA 2021/药房库存project/data analysis/7.29-8.4/epd_7778.csv', dtype=np.str, delimiter=',')
ecdf7778 = np.insert(ecdf7778, 24, np.zeros([1,np.size(ecdf7778,1)]),axis=0)
ECDFM += [ecdf7778]
ecdf7796 = np.loadtxt('D:/RA 2021/药房库存project/data analysis/7.29-8.4/epd_7796.csv', dtype=np.str, delimiter=',')
ecdf7796 = np.insert(ecdf7796, 24, np.zeros([1,np.size(ecdf7796,1)]),axis=0)
ECDFM += [ecdf7796]    








# Initialization


rep_time = 100  # number of simulation replications
# Performance metrics
lack_matrix_vector = np.zeros([rep_time+1, M+1, T+1])
total_lack_vector = np.zeros(rep_time+1)

for i in range(1, rep_time+1):
    print('--------------replication '+str(i)+'----------------')
    I_t = np.zeros([M+1, T+1]) # the status of each medicine in stock throughout the planning horizon
    D_t = np.zeros([M+1, T+1]) # the demand of each medicine throughout the planning horizon
    R_t = np.zeros([M+1, T+1]) # the replenishment decision of each medicine throughout the planning horizon
    
    Success_Delivery_Amount = np.zeros([M+1, T+1]) # store the time and amount of success delivery
    Lack_Amount = np.zeros([M+1, T+1]) # store the lack amount in each time period 
    LACK = np.array([]) # store the time, medicine, and lack amount of all "lack events"
    
    # System status at time 0
    for m in range(1, M+1):
        I_t[m][0] = C[m]
    
        
    # The planning horzion starts
    I_t[1:(M+1), 0] = C[1:]
    
    for t in range(T): 
        d = math.ceil(t/360)
        tp = t%360 if t%360!=0 else 360
        ti = math.ceil(tp/15)
        R_t[1:(M+1), t] = POLICY(M, I_t[:, t], C, Pol)[1:]  # apply policy to decide which medicines need replenishment, and how much
        D_t[1:(M+1), t] = DEMAND(M, ti, ECDFM)[1:]  # generate the demand
       
        # system status updated by medicine delivery, and 'lack events' are recorded
        for m in range(1, M+1):
            if D_t[m][t] == 0:
                I_t[m][t+1] = I_t[m][t]
                continue
            if I_t[m][t] > D_t[m][t]:  # able to deliver medicine m
                I_t[m][t+1] = I_t[m][t] - D_t[m][t]
                Success_Delivery_Amount[m][t] = D_t[m][t]               
            else:                       # not able to deliver medicine m
                LACK = np.append(LACK, [t, m, D_t[m][t]-(I_t[m][t]-1)])
                Lack_Amount[m][t] = Lack_Amount[m][t] + (D_t[m][t]-(I_t[m][t]-1))                                 
                I_t[m][t+1] = 1       
                
        I_t[1:(M+1), t+1] = I_t[1:(M+1), t+1] + R_t[1:(M+1), t] # system status updated by replenishment
        
    lack_matrix_vector[i] = Lack_Amount
    total_lack_vector[i] = np.sum(Lack_Amount)
  
average_total_lack = np.mean(total_lack_vector)   
    
# Performance metrics
plt.figure(1)
n, bins, patches = plt.hist(x=total_lack_vector[1:], bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Total lack amount')
plt.ylabel('Frequency')
plt.title('Histogram for total lack amounts in '+str(rep_time)+' replications')
maxfreq = n.max()
# 设置y轴的上限
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
plt.show()


# Plots
mid = int(input('Plot the delivery and replenishment curve of medicine: '))
plt.figure(2)
plt.plot(range(T+1), R_t[mid], linestyle='-', color = 'blue', label='Replenishment')
plt.plot(range(T+1), Success_Delivery_Amount[mid], linestyle = '-', color = 'green', label='Delivery')
plt.plot(range(T+1), Lack_Amount[mid], linestyle = '-', color = 'red', label='Lack')
plt.plot(range(T+1), I_t[mid], linestyle = '-', color = 'coral', label='stock level')
plt.xlabel('Time')
plt.ylabel('Amount')
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0)
plt.title('The delivery and replenishment time and amount of medicine '+str(mid))
plt.show()
    

    
### Export demand vector
df = pd.DataFrame(D_t)
df.to_csv('D:/RA 2021/药房库存project/data analysis/7.29-8.4/demand.csv')
    
    
  
# Test the simualtion demand   
# for i in range(100):
#     print(i)
#     for t in range(T):
#         d = math.ceil(t/360)
#         tp = t%360 if t%360!=0 else 360
#         ti = math.ceil(tp/15)
#         D_t[1:(M+1), t] += DEMAND(M, ti, ECDFM)[1:]  # generate the demand

# D_t = D_t/100
# df = pd.DataFrame(D_t)
# df.to_csv('D:/RA 2021/药房库存project/data analysis/7.29-8.4/demand.csv')

    
    
    
    