import numpy as np
import pandas as pd  # pandas data frame
import pickle

def load_data(path = './Stock_data.pkl'):
    '''function to load data from pickle file
    Input:
    path: path to pickle file
    '''
    with open(path,'rb') as f:
        data = pickle.load(f)
    return data

def approx_optimal_CRP(price_relatives_stack, N=1000):
    '''function to implement approximate optimal constant rebalance portfolio strategy
    Inputs
    price_relative_stack : array of prices relatives of stocks
    N : number of samples taken from probability simplex, more the better'''
    
    m,n = price_relatives_stack.shape # m = no. of different stocks, n = no. of samples
    CRP = np.ones([1, m]) 
    
    # generate N random probability vectors from the simplex : method 1
#     a = np.random.random([N,m])
#     thetas = a/np.sum(a,axis = 1)[:,np.newaxis]
    
    # generate N random probability vectors from the simplex : method 2
    thetas = np.random.dirichlet([1]*m,N)
    
    gains = thetas@np.prod(price_relatives_stack, axis = 1)
    idx = np.argmax(gains)
    
    return thetas[idx, :] , gains[idx] 


def approx_UP(price_relatives_stack, alpha=1/2, N=100):
    '''function to find the net wealth gain by Universal Portfolio algorithm
    Inputs
    price_relatives_stack : numpy array containing price relatives (rows - stocks, columns - time)
    alpha : dirchlet paramter
    N : Number of samples drawn from dirichlet distribution
    Outputs
    wealth : final expected wealth gain from Universal Portfolio algorithm '''
    ### FILL IN YOUR CODE ###
    m,n = price_relatives_stack.shape # m = no. of different stocks, n = no. of samples
    wealth = 1
    portfolios = []
    growth = []
    
    # generate N portfolios at random from the simplex of dirchlet(alpha) distribution
    thetas = np.random.dirichlet([alpha]*m,N)
    
    # find wealth ratio produced from the N portfolios
    temp = thetas@price_relatives_stack # N x n matrix
    temp_c = np.prod(temp, axis = 1) # N x 1 matrix with final wealth ratio
    
    b1 = np.mean(thetas* (temp_c )[:,np.newaxis] , axis = 0)/np.mean(temp_c )
    wealth1 = b1@np.prod(price_relatives_stack,axis = 1 )
    
    return wealth1


def approx_UP_advanced(price_relatives_stack, alpha=1/2, N=100, adap = 'off'):
    '''function to find the Universal Portfolio at each day and find net wealth gain
    Inputs
    price_relatives_stack : numpy array containing price relatives (rows - stocks, columns - time)
    alpha : dirchlet paramter
    N : Number of samples drawn from dirichlet distribution
    adap : If on, would implement adaptive algorithm (lesser latency, better performace)
    Outputs
    wealth : final expected wealth gain
    portfolios : Portfolios learned by the algorithm each day
    growth : wealth gain achieved by algorithm at each day '''
    
    ### FILL IN YOUR CODE ###
    m,n = price_relatives_stack.shape # m = no. of different stocks, n = no. of samples
    wealth = 1
    portfolios = []
    growth = []
    
    # generate N portfolios at random from the simplex of dirchlet(alpha) distribution
    thetas = np.random.dirichlet([alpha]*m,N)
    
    # find wealth ratio at each day, from the N portfolios
    temp = thetas @ price_relatives_stack # N x n matrix
    temp_c = np.cumprod(temp, axis = 1) # N x n matrix with cumulative wealth ratio
    
    for i in range(n):
        if i == 0:
            b1 = np.ones(m)/m
        
        else :
            # for each portfolio find CRP wealth till preceding day
            temp1 = temp_c[:,i-1] # N x 1 matrix, each entry is wealth ratio
            w = temp1/np.sum(temp1)
            if adap == 'on':
                # find weighted average CRP wealth of each portfolio (Monte Carlo) 
                b1 = np.mean(thetas* (temp1*w )[:,np.newaxis] , axis = 0)/np.mean(temp1*w )
            else: 
                # find weighted average CRP wealth of each portfolio (Monte Carlo) 
                b1 = np.mean(thetas* (temp1 )[:,np.newaxis] , axis = 0)/np.mean(temp1 )
        
        # update the wealth gained by UP
        gain = b1@price_relatives_stack[:,i]
        
        portfolios.append( b1 )
        wealth *= gain
        growth.append(gain)

    return wealth, np.array(portfolios), np.array(growth)
