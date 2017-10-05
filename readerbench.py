from pandas import *
from csv import DictReader
import numpy
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn import metrics
import csv
from GeneralFunctions import *
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    target_values_train = []
    target_values_validation = []
    id = []
    predictions = []
    list_of_scores = []
    list_of_averages = []
    list_best_features = []
    count = 0

    with open('train.txt') as f:
        reader = DictReader(f, delimiter='\t')
        for row in reader:
            target_values_train.append(int(row['human-generated']))
    y_train = numpy.asarray(target_values_train)
    y_train = y_train[:, numpy.newaxis]

    y_split_train = split_data_frame(y_train, chunk_size=400000)
    model = SelectKBest()
    scaler = StandardScaler()

    csv_reader = pandas.read_csv('train-indices.csv', iterator=True, chunksize=400000, delimiter=';', skiprows=1)
    #df_train = pandas.DataFrame()

    for chunk in csv_reader:
        chunks = []
        chunks.append(chunk)
        df_train = pandas.concat(chunks, ignore_index=True)
        del df_train['id']
        print(df_train.info())

        y_train = y_split_train[count]
        model.fit(numpy.nan_to_num(df_train), y_train)
        list_of_scores.append(model.scores_)
        count += 1

    for j in range(0, len(model.scores_)):
        summ = 0
        for item in list_of_scores:
            summ += item[j]
        list_of_averages.append(summ / (len(list_of_scores)))

    print(list_of_averages)
    print(len(list_of_averages))
    dict_of_averages = {v: k for v, k in enumerate(list_of_averages)}
    sorted_dict = sorted(dict_of_averages.items(), key=lambda value: value[1])
    sorted_dict = dict(sorted_dict)
    keys_list = list(sorted_dict.keys())
    for i in range(len(list_of_averages) - 50, len(list_of_averages)):
        list_best_features.append(keys_list[i])
    print(list_best_features)
    print(len(list_best_features))

    csv_reader_train = pandas.read_csv('train-indices.csv', iterator=True, chunksize=400000, delimiter=';', skiprows=1,
                                       usecols = list_best_features)
    df_train = pandas.concat(csv_reader, ignore_index=True)  # Creating the dataframe
    df_train.info()
    df_train = numpy.nan_to_num(df_train)  # Transform the dataframe into a numpy array, eliminating the NaN values

    scaler.fit(df_train)
    df_train_scaled = scaler.transform(df_train)  # Scaled data for traing
    X_search, y_search = random_select(df_train_scaled, y_train, 50000)
    X_training, y_training = random_select(df_train_scaled, y_train, 100000)

    parameter_candidates = [
        {'alpha': [0.0001, 0.001, 0.01, 0.1, 1], 'loss': ['log'], 'n_iter': [10, 20, 30]},
    ]

    clf_cv = GridSearchCV(estimator=SGDClassifier(class_weight='balanced'), param_grid=parameter_candidates, n_jobs=-1)
    clf_cv.fit(X_search, y_search.ravel())
    print('Best score for data1:', clf_cv.best_score_)
    print('Best aplpha', clf_cv.best_estimator_.alpha)
    print('Best loss:', clf_cv.best_estimator_.loss)
    print('Best n_iter', clf_cv.best_estimator_.n_iter)
    print(clf_cv.cv_results_)
    clf = SGDClassifier(alpha=clf_cv.best_estimator_.alpha, loss= clf_cv.best_estimator_.loss,
                        n_iter=clf_cv.best_estimator_.loss)
    clf.fit(X_training, y_training)
    with open('validation.txt') as f:
        reader = DictReader(f, delimiter='\t')
        for row in reader:
            target_values_validation.append(int(row['human-generated']))

    y_validation = numpy.asarray(target_values_validation)
    y_validation = y_validation[:, numpy.newaxis]

    csv_reader_validation = read_csv('validation indices.csv', iterator=True, chunksize=100000, delimiter=';',
                                     usecols=list_best_features)
    df_validation = concat(csv_reader_validation, ignore_index=True)
    df_validation = numpy.nan_to_num(df_validation)
    X_validation = scaler.transform(df_validation)  # Scaled validation data

    predicted = clf.predict_proba(X_validation)  # Getting probabilities for each response to be human or machine
    print(predicted.shape)
    print(clf.classes_)
    # Getting the probabilities for response to be human (1) and write them in the csv submit file
    m, n = predicted.shape
    for j in range(0, m):
        predictions.append(predicted[j][1])

    for k in range(0, len(predictions)):
        id.append(k)

    with open("submit.csv", "a") as f:
        writer = csv.writer(f)
        for row in zip(id, predictions):
            writer.writerow(row)

    auc = metrics.roc_auc_score(y_validation, predictions)
    print(auc)








