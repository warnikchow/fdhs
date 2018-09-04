# FDHS
Fully Data-driven Contextual Hashtag Segmentation

## Requirements
Keras (TensorFlow), Numpy, NLTK, RegEx, itertools

## Dictionary
Visit: https://nlp.stanford.edu/projects/glove/
Download: Twitter (2B tweets, 27B tokens, 1.2M vocab, uncased, 25d, 50d, 100d, & 200d vectors, 1.42 GB download): glove.twitter.27B.zip
Download link: http://nlp.stanford.edu/data/glove.twitter.27B.zip
* Download this and locate 100dim dictionary to the same folder with 'hashseg.py', in file name 'glove100.txt'.
* Dictionary-free version is under implementation!

## System Description
* The system was trained with 'train.py' (line by line; will be disclosed later!)
* Easy start: start your work in the folder
<pre><code> git clone https://github.com/warnikchow/fdhs </code></pre>
* Locate dictionary inside the folder
* Utilize the segmentation toolkit by following command:
<pre><code> from fdhs.hashseg import segment as seg </code></pre>
* Sample usage:
<pre><code> seg('#what_do_you_want') 
 >> 'what do you want' </code></pre>
<pre><code> seg('#WhatDoYouWant')  
 >> 'What Do You Want' </code></pre>
<pre><code> seg('#whatdoyouwant') 
 >> 'what do you want' </code></pre>
