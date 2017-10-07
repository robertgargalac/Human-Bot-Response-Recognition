import gensim
import numpy
from csv import DictReader
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import NuSVC
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import csv
from GeneralFunctions import *

#Getting the best 3 cosine similarities between word vector from context and response
def get_features(context, response, X):
    list_of_similarities = []
    w2vec_list_response = []
    w2vec_list_context = []


    #Getting the word`s vectors from response and context, using word2vec pre-trained model
    for word in context:
        try:
            word_vect = model[word]
            w2vec_list_context.append(word_vect)

        except KeyError:
            print('The word is not in dictionary')

    for word in response:
        try:
            word_vect = model[word]
            w2vec_list_response.append(word_vect)
        except KeyError:
            print('The word is not in dictionary')

    #If one of the context or response phrases are less than 3 word, returns a list with 3 items [0, 0, 0]
    #Meaning that those 2 phrases have no similarity
    if len(w2vec_list_response) < 3 or len(w2vec_list_context) < 3:
        for i in range(0, 3):
            list_of_similarities.append(0)
    else:
        for vec_c in w2vec_list_context:

            for vec_r in w2vec_list_response:

                #Transforms each word vector into a numpy array
                word_vector_response = numpy.asarray(vec_r)
                word_vector_response = word_vector_response.reshape(1, -1)
                word_vector_context = numpy.asarray(vec_c)
                word_vector_context = word_vector_context.reshape(1, -1)

                #Computing the similitarity between 2 word vectors
                similarity = cosine_similarity(word_vector_context, word_vector_response)
                list_of_similarities.append(similarity)
    #Getting the best 3 similarities for a context-response pair
    list_of_similarities.sort(reverse=True)
    list_best_similarities = [list_of_similarities[i] for i in range(0, 3)]
    X.append(list_best_similarities)

def nltk_text_processing(phrase, lemmatized_list, en_stop_words):
    clean_phrase = token_deletion(phrase)
    tokenized_phrase = word_tokenize(clean_phrase)
    filtered_phrase = [word for word in tokenized_phrase if not word in en_stop_words]
    np_phrase = [word for word in filtered_phrase if word.isalnum()]
    for word in np_phrase:
        lemmatized_list.append(word_lemmatizer.lemmatize(word))


if __name__ == '__main__': #avoid recursive spawning of subprocesses

    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    stop_words = set(stopwords.words("english"))
    word_lemmatizer = WordNetLemmatizer()
    target_values = []
    target_values_validation = []
    X_train_list = []
    X_validation_list = []
    predictions = []
    id = []
    count = 0

    with open('train.txt') as f:
        reader = DictReader(f, delimiter='\t')
        for row in reader:
            lemmatized_list_context = []
            lemmatized_list_response = []
            target_values.append(int(row['human-generated']))
            nltk_text_processing(row['response'], lemmatized_list_response, stop_words)
            nltk_text_processing(row['context'], lemmatized_list_context, stop_words)
            get_features(lemmatized_list_context, lemmatized_list_response, X_train_list)
            count += 1
            print(count, "lines were processed")
            if count > 4500000:
                break

    y_train = numpy.asarray(target_values)
    y_train = y_train[:, numpy.newaxis]
    X_train = numpy.asarray(X_train_list)
    print(X_train)
    print(X_train.shape)
    print(y_train.shape)
    print(X_train.shape)
    ''''#Performing a GridSearch on a 100k subset (randomly chosen) for geting the best hyper-params
    X_train_search, y_train_search = random_select(X_train, y_train, 500000)

    parameter_candidates = [
        {'nu': [0.01, 0.05, 0.1, 0.15], 'kernel': ['rbf'], 'gamma': numpy.logspace(-3, 2, 6)},
    ]

    clf_cv = GridSearchCV(estimator=NuSVC(class_weight='balanced'), param_grid=parameter_candidates,
                          n_jobs=-1)  # It performs the grid search with all virtual cores of the CPU
    clf_cv.fit(X_train_search, y_train_search.ravel())

    print('Best score for data1:', clf_cv.best_score_)
    print('Best nu:', clf_cv.best_estimator_.nu)
    print('Best gamma:', clf_cv.best_estimator_.gamma)
    print('Best kernel:', clf_cv.best_estimator_.kernel)
    print(clf_cv.cv_results_)'''
    X_train_random, y_train_random = random_select(X_train, y_train, 500000)

    #Train the SVM classifier with hyper-params resulted from grid search
    clf = NuSVC(nu=0.1, gamma=10, kernel='rbf', probability=True)
    clf.fit(X_train_random, y_train_random)

    with open('validation.txt') as fv:
        reader = DictReader(fv, delimiter='\t')
        for row in reader:
            lemmatized_list_context_val = []
            lemmatized_list_response_val = []
            target_values_validation.append(int(row['human-generated']))
            nltk_text_processing(row['response'], lemmatized_list_response_val, stop_words)
            nltk_text_processing(row['context'], lemmatized_list_context_val, stop_words)
            get_features(lemmatized_list_response_val, lemmatized_list_context_val, X_validation_list)

    y_validation = numpy.asarray(target_values_validation)
    y_validation = y_validation[:, numpy.newaxis]
    X_validation = numpy.asarray(X_validation_list)
    print(y_validation.shape)
    print(X_validation.shape)

    predict_prob = clf.predict_proba(X_validation)
    predict = clf.predict(X_validation)
    print(predict_prob.shape)
    print(predict.shape)

    m, n = predict_prob.shape
    for j in range(0, m):
        predictions.append(predict_prob[j][1])

    for k in range(0, len(predictions)):
        id.append(k)

    with open("submit4.csv", "a") as f1:
        writer1 = csv.writer(f1)
        for row in zip(id, predictions):
            writer1.writerow(row)

    with open("submit5.csv", "a") as f2:
        writer2 = csv.writer(f2)
        for row in zip(id, predict):
            writer2.writerow(row)

