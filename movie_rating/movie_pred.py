import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



class SigmoidNeuron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def predict(self, inputs):
        z = np.dot(inputs, self.weights) + self.bias
        return self.sigmoid(z)

def main():
    

    
    data = pd.read_csv('new_movies.csv')

    data.dropna(inplace=True)

    features = data[['budget', 'gross', 'runtime']].values
    ratings = data['score'].values

    # Create a scaler
    scaler = MinMaxScaler()

    # Scale the features
    features = scaler.fit_transform(features)
    print(features[1])


    # Features: [budget, duration, genre]
    # features = np.array([
    #     [0.8, 0.7, 0.2],  
    #     [0.3, 0.5, 0.8],  
    #     [0.6, 0.4, 0.5],  
    #     [0.01263213, 0.02067008, 0.15755627],
    # ])
    # ratings = np.array([0.9, 0.6, 0.7, 0.8])  
    
    input_size = features.shape[1]
    neuron = SigmoidNeuron(input_size)
    
    epochs = 1000
    learning_rate = 0.1
    

    for epoch in range(epochs):
        for i in range(len(features)):
            prediction = neuron.predict(features[i])
            error = prediction - ratings[i]
            neuron.weights -= learning_rate * error * features[i]
            neuron.bias -= learning_rate * error
        
        
    new_movies = np.array([
        [0.7, 0.6, 0.4],    
        [0.5, 0.8, 0.6],    
    ])
    
    
    for i, movie in enumerate(new_movies):
        predicted_rating = neuron.predict(movie)
      
        print(f"Predicted rating for New Movie {i+1}: {predicted_rating}")

if __name__ == "__main__":
    main()
