import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Reading data file
df = pd.read_csv('C:\\Users\\xieya\\Documents\\GitHub\\mushroom-neural-network\\source\\agaricus-lepiota.data')

features = df.iloc[:,0:]
#One hot encoding
features = pd.get_dummies(features).astype(float)
features.head()

#Output the encoded file
features.to_csv('hot_encoded.txt', index=False,header=None)

#Split the data into training(70), validation(15) and test(15) datasets
train, validate, test = np.split(features.sample(frac=1), [int(.7*len(features)), int(.85*len(features))])

#Convert data to int
train = train.astype(int)
validate = validate.astype(int)
test = test.astype(int)

#Save the datasets
train.to_csv('C:\\Users\\xieya\\Documents\\GitHub\\mushroom-neural-network\\training.txt', index=False,header=None,)
validate.to_csv('C:\\Users\\xieya\\Documents\\GitHub\\mushroom-neural-network\\val.txt', index=False,header=None,)
test.to_csv('C:\\Users\\xieya\\Documents\\GitHub\\mushroom-neural-network\\testing.txt', index=False,header=None,)