import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_data():
    hidden_neurons = 5
    X = np.array(([0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],[0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],[1,0,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1],[1,1,0,0],[1,1,0,1],[1,1,1,0],[1,1,1,1]))
    actual_output = np.array([0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0]).reshape(-1,1)
    w1 = np.random.randn(X.shape[1],hidden_neurons)
    w2 = np.random.randn(hidden_neurons,actual_output.shape[1])
    return X,actual_output,w1,w2

def forward(X,w1,w2):
    Z1 = X.dot(w1)
    A1 = 1 / (1 + np.exp(-Z1))
    Z2 = A1.dot(w2)
    A2 = 1 / (1 + np.exp(-Z2))
    return A2,A1
    
def derivate_w1(w2,hidden_output,X,delta1):
    delta2 = delta1.dot(w2.T)*hidden_output*(1-hidden_output)
    derivative_w1 = X.T.dot(delta2)
    return derivative_w1
    
def derivate_w2(actual_output,predicted_output,learning_rate,hidden_output):
    delta1 = (predicted_output - actual_output) *learning_rate
    derivative_w2 = hidden_output.T.dot(delta1)
    return derivative_w2,delta1
    
def cost_function(predicted_output,actual_output):
    cost_value = (-1 / len(actual_output))*((actual_output*np.log(predicted_output)) + ((1-actual_output)*(np.log(1-predicted_output))))
    return cost_value.sum()

def classification_rate(predicted_output,actual_output):
    P = (predicted_output > 0.5).astype(int)
    cr = ((P == actual_output).sum()) / len(actual_output)
    return cr
    
    
def main():
    X,actual_output,w1,w2 = get_data()
    learning_rate = 0.01
    cost = []

    for epoch in range(50000):
        predicted_output,hidden_output = forward(X,w1,w2)
        c = cost_function(predicted_output,actual_output)
        success_rate = classification_rate(predicted_output,actual_output)
        print('Success rate: %s cost :%s'%(str(success_rate),str(c)))
        cost.append(c)
        derivative_w2,delta1 = derivate_w2(actual_output,predicted_output,learning_rate,hidden_output)
        w2 -= derivative_w2
        w1 -= derivate_w1(w2,hidden_output,X,delta1)
    
    plt.figure()
    plt.xlabel('No of epochs')
    plt.ylabel('Cost function')
    plt.plot(cost)
    plt.show()
if __name__ == '__main__':
    main()