# SentimentalAnalysis
Q：What is this project for?

A: It is a project that include three sentimental analysis classifiers. It can be use to classify tweets from negative, postive and neutral. A full connected nueral network has been used based on a bayes probability features, and a 2-layers LSTM networks is handling on Word2Vec feature. See detail from report.

Q: How to run this code?

A: This project is developed based on python 3.6.4. Several package has been used to tackle certain task: Here is a table listing the package need to be installed before running code.

pickle		Can be installed by pip or Anaconda

numpy		Can be installed by pip or Anaconda

keras		A learning framework can be easily installed in Anaconda, tensoflow required.

matplotlib	Can be installed by pip or Anaconda 

nltk		Need bigrams, stopwords etc 

textblob	Can be install by pip

After installed all this package, a pre-trained word2Vect Glove.twitter.27B.50. required to be put in to semeval-tweets folder. 
Then you can run this code by type command in to terminal: 
    python classification.py.
The process of pre-processing and model training may takes several minutes to be done.

Q: I want to use trained model to evaluate

A: The pre-processed data set and trained model are not submitted through tabula, due to the limitation of submission requirement. 

The whole foleder has been uploaded on figShare, so that you can access all codes, data set and trained model. 

https://figshare.com/articles/cache_rar/6004094

All pre-processed data set and trained model, should be placed under cache folder

Q: Where can I download Glove.twitter.27B.50.?
A: Glove.twitter.27B.50. is a public dataset which can be accessed from Stanford University: 

https://nlp.stanford.edu/projects/glove/
http://nlp.stanford.edu/data/glove.twitter.27B.zip

Required Files:

      semeval-tweets/Glove.twitter.27B.50.txt

Pretrained model and data set:

	cache/word2prob.dat
	
	cache/train_label_uni.txt
	
	cache/train_label_bi.txt
	
	cache/train_data_uni.txt
	
	cache/train_data_bi.txt
	
	cache/soft_y.txt
	
	cache/soft_x.txt
	
	cache/plain_model_bi.dat
	
	cache/plain_model.dat
	
	cache/LSTMKeras.dat
	
	cache/emojify_data.csv
	
	cache/bigram2prob.dat
      
 
