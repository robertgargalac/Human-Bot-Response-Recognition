from nltk.corpus import wordnet as wn
import numpy
import random

def token_deletion(text):
    text = text.replace('@@ ', '')
    text = text.replace('<at>', '')
    text = text.replace('<url>', '')
    text = text.replace('<number>', '')
    text = text.replace('<heart>', '')
    text = text.replace('<cont>', '')
    text = text.replace('<first_speaker> ', '')
    text = text.replace('<second_speaker> ', '')
    text = text.replace('<third_speaker> ', '')
    text = text.replace('<minor_speaker> ', '')
    text = text.replace('<at> ', '')
    text = text.replace('<url> ', '')
    text = text.replace('<number> ', '')
    text = text.replace('<heart> ', '')
    text = text.replace('<cont> ', '')
    return text

#Function which returns the wordnet pos tag given a nltk pos tag
def get_wordnet_pos(nltk_tag):

    if nltk_tag.startswith('J'):
        return wn.ADJ
    elif nltk_tag.startswith('V'):
        return wn.VERB
    elif nltk_tag.startswith('N'):
        return wn.NOUN
    elif nltk_tag.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN

 #It returns a random subset with n_samples
def random_select(x, y, n_samples):
    list_of_features = []
    list_of_targets = []
    shuffled_range = list(range(len(y)))
    random.shuffle(shuffled_range)
    for i in range(0, n_samples):
        list_of_features.append(x[shuffled_range[i]])
        list_of_targets.append(y[shuffled_range[i]])

    x = numpy.array(list_of_features)
    y = numpy.array(list_of_targets)
    return x, y

def get_norm_freq(response, context, Xnorm):
    counter=0
    number_of_tokens = len(response) + len(context)
    for word in response:
        if context.count(word) > 0:
            counter = counter + 1
    if counter == 0:
        Xnorm.append(counter)
    else:
        Xnorm.append(counter/number_of_tokens)

# Function which returns how many identical word are in both response and context
def get_freq(response, context, x):
    counter = 0
    for word in response:
        if context.count(word) > 0:
            counter = counter + 1
    x.append(counter)


def split_data_frame(df, chunk_size):
    listofdf = list()
    number_chunks = len(df) // chunk_size + 1
    for i in range(number_chunks):
        listofdf.append(df[i*chunk_size:(i+1)*chunk_size])
    return listofdf
