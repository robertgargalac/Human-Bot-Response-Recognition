from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from csv import DictReader
import numpy
import csv
from sklearn.svm import NuSVC
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from GeneralFunctions import *
stop_words = set(stopwords.words("english"))
word_lemmatizer = WordNetLemmatizer()
target_values = []
predictions = []
id = []
X_train_list = []
X_validation_list = []
target_values_validation = []



#Function to process each phrase (token extraction, get rid of stop words and alpha num char and lemmatize each token)


def nltk_text_processing(phrase, lemmatized_list):

    clean_phrase = token_deletion(phrase)
    tokenized_phrase = word_tokenize(clean_phrase)
    filtered_phrase = [word for word in tokenized_phrase if not word in stop_words]
    np_phrase = [word for word in filtered_phrase if word.isalnum()]
    for word in np_phrase:
        lemmatized_list.append(word_lemmatizer.lemmatize(word))

#Function to randomly select batch of samples


if __name__ == '__main__':
    with open('train.txt') as f:
        reader = DictReader(f, delimiter='\t')
        for row in reader:
            lemmatized_list_context = []
            lemmatized_list_response = []
            target_values.append(int(row['human-generated']))  # Getting the labels (0 & 1)
            # Process each phrase from context and response columns

            nltk_text_processing(row['response'], lemmatized_list_response)
            nltk_text_processing(row['context'], lemmatized_list_context)

            # Getting the number of words which appear in both context and response  for each context-response pair

            get_freq(lemmatized_list_response, lemmatized_list_context, X_train_list)

    # Transforming the lists with labels and features into numpy arrays
    y_train = numpy.asarray(target_values)
    y_train = y_train[:, numpy.newaxis]
    X_train = numpy.asarray(X_train_list)
    X_train = X_train.reshape(-1, 1)
    print(y_train.shape)
    print(X_train.shape)

    # Scaling the train matrix for a better performance with the classifier
    scaler = StandardScaler()
    scaler_norm = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_train_search, y_train_search = random_select(X_train_scaled, y_train, 100000)

    parameter_candidates = [
        {'nu': [0.01, 0.05, 0.1, 0.15], 'kernel': ['rbf'], 'gamma': numpy.logspace(-3, 2, 6)},
    ]
    clf_cv = GridSearchCV(estimator=NuSVC(class_weight='balanced'), param_grid=parameter_candidates,
                          n_jobs=-1)  # It performs the grid search with all power of the CPU
    clf_cv.fit(X_train_search, y_train_search.ravel())
    print('Best score for data1:', clf_cv.best_score_)
    print('Best nu:', clf_cv.best_estimator_.nu)
    print('Best gamma:', clf_cv.best_estimator_.gamma)
    print('Best kernel:', clf_cv.best_estimator_.kernel)
    print(clf_cv.cv_results_)

    # Getting 500k samples (randomly chosen) for SVM train
    X_train_scaled_random, y_train_random = random_select(X_train_scaled, y_train, 500000)

    # SVM classifier with rbf kernel
    clf = NuSVC(nu=clf_cv.best_estimator_.nu, gamma=clf_cv.best_estimator_.gamma, kernel=clf_cv.best_estimator_.kernel,
                probability=True)
    clf.fit(X_train_scaled_random, y_train_random)

    with open('validation.txt') as fv:
        reader = DictReader(fv, delimiter='\t')
        for row in reader:
            lemmatized_list_context_val = []
            lemmatized_list_response_val = []
            target_values_validation.append(int(row['human-generated']))
            nltk_text_processing(row['response'], lemmatized_list_response_val)
            nltk_text_processing(row['context'], lemmatized_list_context_val)
            get_freq(lemmatized_list_response_val, lemmatized_list_context_val, X_validation_list)

    y_validation = numpy.asarray(target_values_validation)
    y_validation = y_validation[:, numpy.newaxis]
    X_validation = numpy.asarray(X_validation_list)
    X_validation = X_validation.reshape(-1, 1)
    print(y_validation.shape)
    print(X_validation.shape)

    # Scaling the validation data regarding to train matrix
    X_validation_scaled = scaler.transform(X_validation)

    # Getting the probabilities for both classes for each response in validation dataset
    predict_prob = clf.predict_proba(X_validation_scaled)

    # Getting the prediction (0 or 1) for each response in validation dataset
    predict = clf.predict(X_validation_scaled)
    print(predict_prob.shape)
    print(predict.shape)

    m, n = predict_prob.shape

    # Extract from the matrix o probabilities, just the "1" column
    for j in range(0, m):
        predictions.append(predict_prob[j][1])

    for k in range(0, len(predictions)):
        id.append(k)

    # Writing in the csv file the probability for each response to be human-generated (1)
    with open("submit.csv", "a") as f1:
        writer1 = csv.writer(f1)
        for row in zip(id, predictions):
            writer1.writerow(row)
    # Writing in the csv file the predictions ( either "0" or "1") for each response
    with open("submit1.csv", "a") as f2:
        writer2 = csv.writer(f2)
        for row in zip(id, predict):
            writer2.writerow(row)

    # Computes the ROC_AUC score
    auc = metrics.roc_auc_score(y_validation, predictions)
    print(auc)
    auc1 = metrics.roc_auc_score(y_validation, predict)
    print(auc1)



