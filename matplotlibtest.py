import pandas as pd 
import matplotlib.pyplot as plt 	
import numpy as np 
import preprocessing
# %matplotlib inline

# def loadDict():

# 	stop_words = [unicode(x.strip(), 'utf-8') for x in open('kamus/stopword.txt','r').read().split('\n')]
# 	noise = [unicode(x.strip(), 'utf-8') for x in open('kamus/noise.txt','r').read().split('\n')]
# 	stop_words.extend(noise)
# 	return stop_words

# def loadData():
# 	train_df_raw = pd.read_csv('trainmatplotlib.csv',sep=';',names=['tweets','label'],header=None)
# 	# test_df_raw = pd.read_csv('dataset/testing.csv',sep=';',names=['tweets','label'],header=None)
# 	train_df_raw = train_df_raw[train_df_raw['tweets'].notnull()]
# 	# test_df_raw = test_df_raw[test_df_raw['tweets'].notnull()]
	# print train_df_raw
def loadDict():
	print "loading dictionary ... "
	stop_words = [unicode(x.strip(), 'utf-8') for x in open('kamus/stopword.txt','r').read().split('\n')]
	noise = [unicode(x.strip(), 'utf-8') for x in open('kamus/noise.txt','r').read().split('\n')]
	stop_words.extend(noise)
	return stop_words

# Load DataTraining
def loadData():
	# train_df_raw = pd.read_csv('trainingdata1.csv',sep=';',names=['tweets','label'],header=None)
	train_df_raw = pd.read_csv('trainingdata1.csv',sep=';',names=['tweets','label'],header=None)
	train_df_raw = train_df_raw[train_df_raw['tweets'].notnull()]
	return train_df_raw

def extractTweetTraining(train_df_raw_in):
	#ekstrak make training and testing 
	docs_train=train_df_raw_in['tweets'].tolist()
	return docs_train

def extractTweetTrainingLabel(train_df_raw_in):
	y_train=[x if x=='positif' else 'negatif' for x in train_df_raw_in['label'].tolist()]
	return y_train

def get_label(train_df_raw_glob):
	candidates = []
	# labels = train_df_raw_in[tweets].lower()
	# labels = [x if x=='positif' else 'negatif' for x in train_df_raw['label'].tolist()]
	labels = y_train
	return ",",join(candidates, labels)


train_df_raw_glob = loadData()
train_df_raw_glob["candidates"] = train_df_raw_glob.apply(get_label, axis=1)



#ekstrak make training and testing 
# docs_train=train_df_raw['tweets'].tolist()
# # docs_test=test_df_raw['tweets'].tolist()
# y_train=[x if x=='positif' else 'negatif' for x in train_df_raw['label'].tolist()]
# y_test=[x if x=='positif' else 'negatif' for x in test_df_raw['label'].tolist()]
# tweet = pd.read_csv('trainmatplotlib.csv',sep=';',names=['tweets','label'],header=None)
# tweet = tweet[tweet['tweets'].notnull()]
# print y_train

# def get_label(row):
# 	Label = []
# 	text = row[0][1]
# 	if "kamus/positif_ta.txt" in text :
# 		Label.append("Positif")
# 	if "kamus/negatif_ta.txt" in text :
# 		Label.append("Negatif")
# 	return ",".join(Label)

# tweet["Label"] = tweet.apply(get_label, axis=1)

counts = train_df_raw-glob["candidates"].value_counts()
plt.bar(range(len(counts)), counts)
plt.show()

print(counts)