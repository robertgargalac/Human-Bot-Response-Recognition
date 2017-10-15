import gensim
import numpy
from csv import DictReader
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from sklearn.metrics.pairwise import cosine_similarity
import random
import nltk
from sklearn.svm import NuSVC
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
import csv
from GeneralFunctions import *

#Geting the cosine similarity between phrase`s vectors obtained by summing up every mean of word vector
def get_features(lem_response, lem_context, X):
    sum_response = 0
    sum_context = 0
    # Taking the word vector using w2vec pre-trained model and transform it into a numpy array
    for word in lem_response:

        try:
            w2vec = model[word]
            vector_of_word = numpy.asarray(w2vec)
            vector_of_word = vector_of_word.reshape(1, -1)
            sum_response += vector_of_word
        except KeyError:
            print('The word is not in dictionary')

    for word in lem_context:
        try:
            w2vec = model[word]
            vector_of_word = numpy.asarray(w2vec)
            vector_of_word = vector_of_word.reshape(1, -1)
            sum_context += vector_of_word
        except KeyError:
            print('The word is not in dictionary')
    #If in one phrase is no word, "0" value will be added to the feature list (meaning that phrases have no similarity)
    try:
        print(sum_context.shape)
        print(sum_response.shape)
        X.append(cosine_similarity(sum_response, sum_context))
    except AttributeError:
        print('one of the phrases has no significant words')
        X.append(0)


#Function to process each phrase(tokenize, lemmatize given the pos tag, get rid of alpha num and stop words)
def text_processing(phrase, lemmatized_list):

    clean_phrase = token_deletion(phrase)
    tokenized_phrase = word_tokenize(clean_phrase)
    filtered_phrase = [w for w in tokenized_phrase if not w in stop_words]
    np_phrase = [word for word in filtered_phrase if word.isalnum()]
    pos_tags = nltk.pos_tag(np_phrase) #A list of tuples made of the word and its nktk pos tag
    for i in range(0, len(pos_tags)):
        if get_wordnet_pos(pos_tags[i][1]) == wn.ADV:
            #Getting the lemma of an adverb
            try:
                format_word = np_phrase[i] + '.r.1'
                adv_lemmatize = wn.synset(format_word).lemmas()[0].pertainyms()[0].name()
                lemmatized_list.append(adv_lemmatize)
            except:
                # If no lemma was found for adv, returns the lemma of that word considering it as a noun(default pos tag)
                lemmatized_list.append(word_lemmatizer.lemmatize(np_phrase[i]))
        else:
            pos_tag_wordnet = get_wordnet_pos(pos_tags[i][1])
            lemmatized_list.append(word_lemmatizer.lemmatize(np_phrase[i], pos_tag_wordnet))

def nltk_text_processing(phrase, lemmatized_list, en_stop_words):
    clean_phrase = token_deletion(phrase)
    tokenized_phrase = word_tokenize(clean_phrase)
    filtered_phrase = [word for word in tokenized_phrase if not word in en_stop_words]
    np_phrase = [word for word in filtered_phrase if word.isalnum()]
    for word in np_phrase:
        lemmatized_list.append(word_lemmatizer.lemmatize(word))


if __name__ == '__main__':

    X_train_list = []
    X_validation_list = []
    target_values = []
    target_values_validation = []
    predictions = []
    stop_words = set(stopwords.words("english"))
    word_lemmatizer = WordNetLemmatizer()
    model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    id=[]
    count = 0

    with open('train.txt') as f:
        reader = DictReader(f, delimiter='\t')
        for row in reader:
            target_values.append(int(row['human-generated']))
            lemmatized_list_response = []
            lemmatized_list_context = []
            nltk_text_processing(row['response'], lemmatized_list_response, stop_words)
            nltk_text_processing(row['context'], lemmatized_list_context, stop_words)
            get_features(lemmatized_list_response, lemmatized_list_context, X_train_list)
            count += 1
            print('Number of processed lines', count)

    y_train = numpy.asarray(target_values)
    y_train = y_train[:, numpy.newaxis]
    X_train = numpy.asarray(X_train_list)
    print(X_train.shape)
    X_train = X_train.reshape(-1, 1)
    print(y_train.shape)
    print(X_train.shape)

    X_train_search, y_train_search = random_select(X_train, y_train, 100000)

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
    X_train_random, y_train_random = random_select(X_train, y_train, 500000)

    clf = NuSVC(nu=clf_cv.best_estimator_.nu, gamma=clf_cv.best_estimator_.gamma, kernel=clf_cv.best_estimator_.kernel,
                probability=True)
    clf.fit(X_train_random, y_train_random)

    with open('validation.txt') as fv:
        reader = DictReader(fv, delimiter='\t')
        for row in reader:
            target_values_validation.append(int(row['human-generated']))
            lemmatized_list_context_val = []
            lemmatized_list_response_val = []
            nltk_text_processing(row['response'], lemmatized_list_response_val, stop_words)
            nltk_text_processing(row['context'], lemmatized_list_context_val, stop_words)
            get_features(lemmatized_list_response_val, lemmatized_list_context_val, X_validation_list)

    y_validation = numpy.asarray(target_values_validation)
    y_validation = y_validation[:, numpy.newaxis]
    X_validation = numpy.asarray(X_validation_list)
    X_validation = X_validation.reshape(-1, 1)
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

    with open("submit2.csv", "a") as f1:
        writer1 = csv.writer(f1)
        for row in zip(id, predictions):
            writer1.writerow(row)

    with open("submit3.csv", "a") as f2:
        writer2 = csv.writer(f2)
        for row in zip(id, predict):
            writer2.writerow(row)

    auc = metrics.roc_auc_score(y_validation, predictions)
    print(auc)
    auc1 = metrics.roc_auc_score(y_validation, predict)
    print(auc1)

