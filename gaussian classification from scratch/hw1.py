import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_np = np.genfromtxt('hw1Data.csv',delimiter=',')

banker=data_np[data_np[:,0]==0]
plt.hist(x=banker[:,-1]);

football=data_np[data_np[:,0]==1]
plt.hist(x=football[:,-1]);

police=data_np[data_np[:,0]==2]
plt.hist(x=police[:,-1]);

waiter=data_np[data_np[:,0]==3]
plt.hist(x=waiter[:,-1]);

# Q1) banker is gaussian distribution
#     football player is exponential distribution
#     police is a gaussian distribution
#     waiter is exponential distribution

# Q2
def learnParams(Data):
    class_params_date = []
    class_params_cost = []
    for i in range(4):
        class_data = Data[Data[:,0]==i]
        class_date = np.array([np.mean(class_data[:,1]),np.std(class_data[:,1])])
        class_params_date.append(class_date)
        class_cost = np.array([np.mean(class_data[:,2]),np.std(class_data[:,2])])
        class_params_cost.append(class_cost)
    return np.array(class_params_date), np.array(class_params_cost)
    
paramsDate, paramsCost =learnParams(data_np)
print('Q2')
print(paramsDate)
print(paramsCost)
print('\n')

# Q3
def learnPriors(Data):
    final_priors = []
    for i in range(4):
        class_data = Data[Data[:,0]==i]
        prior = (len(class_data)/len(Data))
        final_priors.append(prior)
    return final_priors
    

priors = learnPriors(data_np)
print('Q3')
print(priors)
print('\n')

# Q4
patientFeats = data_np[:,1:]
def labelBayes(patientFeats,paramsDate,paramsCost,priors):
    learnprior = [paramsDate,paramsCost]
    like = []
    for feat in patientFeats:
        for idx,val in enumerate(priors):
            for date_par in [learnprior]:
                max_date = (1/np.sqrt(2*3.14*(date_par[0][idx][1]**2)))*2.7183**(-(feat[0]-date_par[0][idx][0])**2/(2*(date_par[0][idx][1]**2)))
                max_cost = (1/np.sqrt(2*3.14*(date_par[1][idx][0]**2)))*2.7183**(-(feat[1]-date_par[1][idx][1])**2/(2*(date_par[1][idx][0]**2)))
                likelihood = (max_cost * max_date * val)
                like.append(likelihood)
    final = (np.reshape(like, (-1, 4)))
    return np.argmax(final, axis=1)
    
print('Q4')
labelsOut = labelBayes(patientFeats,paramsDate,paramsCost,priors)
print(labelsOut)
print('\n')

# Q5
def learnCalendar(Data):
    return np.round(Data[0],0)
print('Q5')
print(learnCalendar(learnParams(data_np)))

/Users/dougymenns/documents/fordham/fall 2021/machine learning/hw1