from sklearn.datasets import make_moons,make_circles,load_breast_cancer
import pandas as pd


X, Y = make_moons(n_samples=1000, noise=0.2, random_state=42)
df = pd.DataFrame(X, columns=['x1', 'x2'])
df['label'] = Y
df.to_csv('Data/moons.csv', index=False)


X1, Y1 = make_circles(n_samples=500, noise=0.1, factor=0.1)
df1 = pd.DataFrame(X1, columns=['x1', 'x2'])
df1['label'] = Y1
df1.to_csv('Data/circles.csv', index=False)

data = load_breast_cancer()
X2 = data.data
X2 = X2[:, [0, 2]] # mean radius and mean perimeter (often informative)
Y2 = data.target
df2 = pd.DataFrame(X2, columns=['x1', 'x2'])
df2['label'] = Y2
df2.to_csv('Data/breast_cancer.csv', index=False)



 
