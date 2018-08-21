import nltk
import numpy as np
import re

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.lower().split('\t') for line in f.read().splitlines()]
    return data

cornell = read_data("cornell_final_shuffle.txt")
cornell_raw   = [row[0] for row in cornell]

import itertools
cornell_allchar = itertools.chain.from_iterable(cornell_raw)
cornell_idchar = {token: idx for idx, token in enumerate(set(cornell_allchar))}

def loadvector(File):
    print("Loading word vectors")
    f = open(File,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

glove_twit = loadvector('glove100.txt')

def underscore(hashtag):
    result=''
    for i in range(len(hashtag)):
      if i>0:
        if hashtag[i].isalpha()==True:
          result = result+hashtag[i]
        else:
          result = result+' '
    return result

def split_hashtag(hashtagestring):
    fo = re.compile(r'#[A-Z]{2,}(?![a-z])|[A-Z][a-z]+')
    fi = fo.findall(hashtagestring)
    result = ''
    for var in fi:
        result += var + ' '
    return result

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.01
set_session(tf.Session(config=config))
from keras.models import load_model
modelcws = load_model('modelcws/rnnconvdnn100-12-0.9617.hdf5')

def hash_pred(sent,model,dic1,dic2,maxlen,wdim):
    conv = np.zeros((1,maxlen,wdim,1))
    rnn  = np.zeros((1,maxlen,len(dic2)))
    charcount=-1
    for j in range(len(sent)):
      if charcount<maxlen-1 and sent[j]!=' ':
        charcount=charcount+1
        if sent[j] in dic1:
          conv[0][charcount,:,0]=dic1[sent[j]]
        if sent[j] in dic2:
          rnn[0][charcount,dic2[sent[j]]]=1
    z = model.predict([conv,rnn])[0]
    print(z)
    sent_raw = ''
    count_char=-1
    for j in range(len(sent)):
      if sent[j]!=' ':
        count_char=count_char+1
        sent_raw = sent_raw+sent[j]
        if z[count_char]>0.5:
          sent_raw = sent_raw+' '
    return sent_raw, z[:count_char], count_char

def digitalize(z):
    y = np.zeros(len(z))
    for i in range(len(z)):
      y[i] = np.heaviside(z[i]-0.5,1)
    return y

def hash_space(tag):
    tag_re = ''
    for i in range(len(tag)):
      if tag[i].isdigit() == True:
        tag_re = tag_re+'#'
      elif tag[i].isalpha() == True:
        tag_re = tag_re+tag[i].lower()
      else:
        tag_re = tag_re+tag[i]
    sent_raw, z, count_char = hash_pred(tag_re,modelcws,glove_twit,cornell_idchar,100,100)
    return sent_raw

big  = re.compile(r"[A-Z]")
small= re.compile(r"[a-z]")

def segment(hashtag):
#    if hashtag in glove_twit:
#      return hashtag
#    elif '_' in hashtag:
#      return underscore(hashtag)
    if '_' in hashtag:
      return underscore(hashtag)
    else:
      if re.search(big,hashtag) and  re.search(small,hashtag):
        return split_hashtag(hashtag)
      else:
        return hash_space(hashtag[1:])
