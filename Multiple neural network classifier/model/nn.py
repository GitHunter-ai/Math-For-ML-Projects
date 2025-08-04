import pandas as pd
import numpy as np
np.random.seed(0)




class NeuralNetwork():
    def __init__(self, n_h=2, learning_rate=0.01, num_iterations=1000):
        # Architecture
        self.n_h = n_h  # number of hidden units
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

        # Parameters (weights and biases)
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None



        # Gradients
        self.dW1 = None
        self.db1 = None
        self.dW2 = None
        self.db2 = None

        # Forward pass cache
        self.Z1 = None
        self.A1 = np.zeros((1, 1))
        self.Z2 = None
        self.A2 = np.zeros((1, 1))
        
        

        # Normalization stats
        self.mean = None
        self.std = None
        
    def normalize(self,X):
        self.mean=X.mean(axis=0)
        self.std=X.std(axis=0)
        return (X-self.mean)/self.std
    
    
    
    def initialize_parameters(self,X,Y):
        Y=Y.reshape(-1,1)
        self.W1=np.random.randn(X.shape[1],self.n_h)*0.01
        self.b1=np.zeros((1,self.n_h))
        self.W2=np.random.randn(self.n_h,Y.shape[1])*0.01
        self.b2=np.zeros((1,Y.shape[1]))
        
    def forward_prop(self, X):
        """
        Perform forward propagation for  Neural Network.

        Arguments:
        X -- input data, shape (m, n_features)

        Returns:
        Y_hat -- predictions, shape (m, 1)
        """
        
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = np.tanh(self.Z1)
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = 1/(1+np.exp(-self.Z2))
        
        

    
    def compute_cost(self, Y, Y_hat=None):
        A2 = self.A2 if Y_hat is None else Y_hat
        cost = -np.mean(Y * np.log(A2 + 1e-8) + (1 - Y) * np.log(1 - A2 + 1e-8))
        return cost

    
    
    def backward_prop(self, X, Y):
        m = X.shape[0]
        
        dZ2 = self.A2 - Y                           # (m, 1)
        self.dW2 = (self.A1.T @ dZ2) / m                # (n_h, 1)
        self.db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = dZ2 @ self.W2.T                      # (m, n_h)
        dZ1 = dA1 * (1 - self.A1**2)               # tanh derivative
        self.dW1 = (X.T @ dZ1) / m                      # (n_x, n_h)
        self.db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
    
    def update_parms(self):
        
        self.W1 -= self.learning_rate*self.dW1
        self.b1 -= self.learning_rate*self.db1
        self.W2 -= self.learning_rate*self.dW2
        self.b2 -= self.learning_rate*self.db2

      

    def fit(self,X,Y,print_cost=False):
        """Trains the model using gradient descent."""
        
        X_norm=self.normalize(X)
        self.initialize_parameters(X,Y)
        
        
        for iteration in range(self.num_iterations):
            self.forward_prop(X_norm)
            self.backward_prop(X_norm,Y)
            self.update_parms()
            cost=self.compute_cost(Y)
            
            if print_cost and iteration % 50 == 0:
                print(f"cost after {iteration} iteration:{cost}")
                
    
        return self


    def predict(self,X):
        X_norm=(X-self.mean)/self.std
        self.forward_prop(X_norm)
        prediction= self.A2>0.5
        return prediction.astype(int)

    
    


    def score(self, X, Y):
        """
        Compute R² score on provided data.
        """
        X_norm = (X - self.mean) / self.std
        Y_hat = self.predict(X_norm)
        return np.mean(Y_hat == Y.reshape(-1, 1))

