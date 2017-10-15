import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA as sklearnPCA
from csv import DictReader
import numpy
from sklearn.preprocessing import StandardScaler
import pandas
import random
target_values_train=[]
list_of_df=[]
list_of_targets=[]
list_of_features = [41, 1, 42, 7, 53, 48, 6, 4, 45, 46, 47, 29,
                    30, 31, 32, 33, 34, 50, 52, 49, 44, 2, 3, 43, 20, 9,
                    18, 51, 14, 12, 8, 11, 13, 5, 22, 10, 15, 16, 17, 19,
                    21, 23, 24, 25, 26, 27, 28, 35, 36]

with open('train.txt') as f:
    reader = DictReader(f, delimiter='\t')
    for row in reader:
        target_values_train.append(int(row['human-generated']))
y_train = numpy.asarray(target_values_train)
y_train = y_train[:, numpy.newaxis]

scaler = StandardScaler()

csv_reader = pandas.read_csv('train-indices.csv', iterator=True, chunksize=500000, delimiter=';', skiprows=1,
                             usecols=list_of_features)
df_train = pandas.concat(csv_reader, ignore_index=True)  # Creating the dataframe
df_train.info()

shuffledRange = list(range(len(df_train)))
random.shuffle(shuffledRange)
df_train = numpy.nan_to_num(df_train)
for i in range(0, 500000):
    list_of_df.append(df_train[shuffledRange[i]])
    list_of_targets.append(y_train[shuffledRange[i]])
df_train_batch = numpy.array(list_of_df)
y_train = numpy.array(list_of_targets)
scaler.fit(df_train_batch)
df_train_scaled = scaler.transform(df_train_batch)

pca = sklearnPCA(n_components=2) #2-dimensional PCA
transformed = pd.DataFrame(pca.fit_transform(df_train_scaled))

    
plt.scatter(transformed[y_train == 0][0], transformed[y_train == 0][1], label='Bot', c='red')
plt.scatter(transformed[y_train == 1][0], transformed[y_train == 1][1], label='Human', c='blue')


plt.legend()
plt.show()













