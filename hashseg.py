import numpy as np
import re

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.lower().split('\t') for line in f.read().splitlines()]
    return data

import string
idchar = {}
for i in range(len(string.ascii_lowercase)):
  idchar.update({string.ascii_lowercase[i]:i})

for i in range(10):
  idchar.update({i:i+26})

idchar.update({'#':36})

big  = re.compile(r"[A-Z]")
small= re.compile(r"[a-z]")
num  = re.compile(r"[0-9]")

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
modelcws = load_model('modelcws/rnnconvdnn100_sigmoid_concat-12-0.9664.hdf5')

threshold=0.35

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
        if z[count_char]>threshold:
          sent_raw = sent_raw+' '
    return sent_raw, z[:count_char], count_char

def hash_space(tag):
    tag_re = ''
    for i in range(len(tag)):
      if tag[i].isalpha() == True:
        tag_re = tag_re+tag[i].lower()
      else:
        tag_re = tag_re+tag[i]
    sent_raw, z, count_char = hash_pred(tag_re,modelcws,glove_twit,idchar,100,100)
    return sent_raw

def segment(hashtag):
    if '_' in hashtag:
      return underscore(hashtag)
    else:
      if re.search(big,hashtag) and  re.search(small,hashtag):
        return split_hashtag(hashtag)
      else:
        return hash_space(hashtag[1:])

def testfunc():
  print(segment('#iwanttosleep'))
  print(segment('#tiredashell'))
  print(segment('#whatdoyouwant'))
  print(segment('#19wksandcounting'))
  print(segment('#youcanthearit'))

def digitalize(z):
  y = np.zeros(len(z))
  for i in range(len(z)):
    y[i] = np.heaviside(z[i]-threshold,1)
  return y

def featurize_space(sent,maxlen):
    onehot = np.zeros(maxlen)
    countchar = -1
    for i in range(len(sent)-1):
      if sent[i]!=' ' and i<maxlen:
        countchar=countchar+1
        if sent[i+1]==' ':
          onehot[countchar] = 1
    return onehot

def hash_space_eval(tag):
  tag_re = ''
  for i in range(len(tag)):
    if tag[i].isalpha() == True:
      tag_re = tag_re+tag[i].lower()
    else:
      tag_re = tag_re+tag[i]
  sent_raw, z, count_char = hash_pred(tag_re,modelcws,glove_twit,idchar,100,100)
  print(sent_raw)
  return digitalize(z), count_char
