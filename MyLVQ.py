# LVQ for the Ionosphere Dataset
from random import seed
import pandas as pd
import preprocessing
from random import randrange
from csv import reader
from math import sqrt
from sklearn.feature_extraction.text import TfidfVectorizer

# Load Dictionary
def loadDict():
	print "loading dictionary ... "
	stop_words = [unicode(x.strip(), 'utf-8') for x in open('kamus/stopword.txt','r').read().split('\n')]
	noise = [unicode(x.strip(), 'utf-8') for x in open('kamus/noise.txt','r').read().split('\n')]
	stop_words.extend(noise)
	return stop_words

# Load DataTraining
def loadDataTraining():
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

# Load DataTest
def loadDataTest():
	# test_df_raw = pd.read_csv('testingdata1.csv',sep=';',names=['tweets','label'],header=None)
	test_df_raw = pd.read_csv('testingdata1.csv',sep=';',names=['tweets','label'],header=None)
	test_df_raw = test_df_raw[test_df_raw['tweets'].notnull()]
	return test_df_raw

def extractTweetTest(test_df_raw_in):
	docs_test=test_df_raw_in['tweets'].tolist()
	return docs_test

def extractTweetTestLabel(test_df_raw_in):
	y_test=[x if x=='positif' else 'negatif' for x in test_df_raw_in['label'].tolist()]
	return y_test


def vectorProcessing(stop_words_in):
	vectorizer = TfidfVectorizer(max_df=1.0, max_features=10000,
                             min_df=0, preprocessor=preprocessing.preprocess,
                             stop_words=stop_words_in,tokenizer=preprocessing.get_fitur
                            )
	return vectorizer

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	# print "class data: ", unique
	lookup = dict()
	for i, value in enumerate(unique):
		if i == 0:
			lookup[value] = 1
		else:
			lookup[value] = 0
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

def str_column_to_int_datatest(datatest, column):
	class_values = [row[column] for row in datatest]
	unique = set(class_values)
	# print "class data: ", unique
	lookup = dict()
	for i, value in enumerate(unique):
		if i == 0:
			lookup[value] = 0
		else:
			lookup[value] = 1
	for row in datatest:
		row[column] = lookup[row[column]]
	return lookup

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds) #membagi data total sejumlah n_folds
	for i in range(n_folds):
		fold = list()
		if i == 0:
			index = 0
		else:
			# index = 50
			index = 50
		# print "Kelompok data: ", i
		while len(fold) < fold_size:
			# print index
			fold.append(dataset_copy[index]) 
			index+=1
		# print fold[20]
		dataset_split.append(fold)
	return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0
	for i in range(len(actual)):
		if actual[i] == predicted[i]:
			correct += 1
	ac = correct / float(len(actual)) * 100.0

	return correct / float(len(actual)) * 100.0

# # Evaluate an algorithm using a cross validation split
# def evaluate_algorithm(dataset, datatest, algorithm, n_folds, *args):
# 	# print args
# 	folds = cross_validation_split(dataset, n_folds)

# 	# print "Len folds: ", len(folds)
# 	# print "Len fold atas: ", len(folds[0])
	
# 	count =0
# 	for x in dataset:
# 		# print len(x)
# 		count+=1
# 	# print "Jum Data", count
# 	scores = list()
	
# 	for fold in folds:
# 		train_set = list(folds)
# 		train_set.remove(fold) #fungsi untuk menghapus populasi total dengan populasi data yg lg di observasi (fold)
# 		train_set = sum(train_set, [])
# 		# print len(train_set[0]) #50 x 234
# 		# i = 0
# 		# for t in train_set:
# 		# 	print "data ke-",i," : ", len(t) # Cari tau kenapa jadi sama semua
# 		# 	i+=1
		
# 		test_set = list()
# 		# print len(folds)
		
# 		for row in fold:
# 			row_copy = list(row)
# 			test_set.append(row_copy)
# 			row_copy[-1] = None

# 		# print "Test:", len(test_set) #Jumlah Data Uji = 70 baris
# 		# print "Train:", len(train_set) #Jumlah Data Training = 280 baris
# 		# print len(train_set)
# 		# print len(test_set)
# 		# print len(datatest)
# 		print test_set

# 		predicted = algorithm(train_set, test_set, *args)
# 		# predicted = algorithm(train_set, datatest, *args)
# 		# print "row: ", row
# 		# print "Len fold: ", len(fold)
# 		# print "fold: ", fold
# 		actual = [row[-1] for row in fold]
		# print "actual: ", actual
		# print "predict: ", predicted
# 		accuracy = accuracy_metric(actual, predicted)
# 		scores.append(accuracy)
	
# 	return scores

	# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, datatest, algorithm, n_folds, *args):
	# print args
	folds_train = cross_validation_split(dataset, n_folds)
	folds_test = cross_validation_split(datatest, n_folds)
	
	# print "Train FOLDS: ", folds_train


	# print "Len folds: ", len(folds)
	# print "Len fold atas: ", len(folds[0])
	
	count =0
	for x in dataset:
		# print len(x)
		count+=1
	# print "Jum Data", count
	scores = list()
	
	for fold_test in folds_test:
		# test_set_fold = list(folds_test)
		# print "row test fold1: ",test_set_fold
		# test_set_fold.remove(fold_test)
		# print "row test fold2: ",test_set_fold
		
		test_set = list()
		for row in fold_test:
			row_copy = list(row)
			row_copy[-1] = None
			test_set.append(row_copy)
			# row_copy[-1] = None
		test_set_final = test_set
		# print "Data Uji: ", test_set_final

		for fold_train in folds_train:
			actual = list()
			for row in fold_train:
				# print "row di actual", row
				actual.append(row[-1])
			# print "Fold Train: ",fold_train
			# train_set = list(folds_train)
			# train_set.remove(fold_train) #fungsi untuk menghapus populasi total dengan populasi data yg lg di observasi (fold)
			# print "train set1: ",len(train_set)
			# train_set = sum(train_set, [])
			
			
			# print "Data Training Before: ", fold_train
			
			
			predicted = algorithm(fold_train, test_set_final, *args)
			# print "Data Training After: ", fold_train
			

			print "actual: ", actual
			print "predict: ", predicted
			accuracy = accuracy_metric(actual, predicted)
			# print "Akurasi: ", accuracy
			scores.append(accuracy)
		folds_train = folds_train
	
	return scores

# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	# print len(row1)
	# print len(row2)
	
	for i in range(len(row1)-1):
		# print "row1: ",row1[i]
		# print "row2: ",row2[i]
		# try:
		distance += (row1[i] - row2[i])**2
		# print "distance: ", distance
		# except Exception as e:
		# 	print "Error: ", e
		# 	print "data ke: ", i
		# 	print "row1: ", row1[i-1]
		# 	print "row2: ", row2[i-1]
	# print "Dist Tot: ", distance
	# print "Akar Dist Tot: ", sqrt(distance)
	return sqrt(distance)

# Locate the best matching unit
def get_best_matching_unit(codebooks, test_row):
	distances = list()
	# print len(codebooks[0])
	# print len(test_row)
	i = 0
	for codebook in codebooks:
		# print len(codebooks)
		# print len(test_row)
		# print "codebook:-",i,": ", codebook
		# print "codebook:", codebook
		# print "test_row: ", test_row
		dist = euclidean_distance(codebook, test_row) #Nilai distance dihitung untuk setiap baris codebook (20 line) ke 1 baris data uji
		distances.append((codebook, dist))
		i+=1
		
	distances.sort(key=lambda tup: tup[1])
	# jarak ED antara 1 baris test uji terhadap semua baris di codeboooks disimpan di list distances.
	# list distances kemudian di urutkan (sorting) berdasarkan  nilai distance. 
	# data terurut dari yg terkecil ke yg terbesar
	# nilai distance paling kecil menunjukkan kedekatan test row dengan baris di codebooks yg paling baik
	# nilai distance terkecil tersebut dapat diakses dengan menggunakan distances[0][0]
	# print "Dist total: ",distances
	# print "Distance:",distances[0][1] #Nilai distance diwakili dng data distances[0][1]
	# print "Codebook Before:",distances[0][0] #Nilai codebook dng nilai distance terkecil diwakili dengan distances[0][0]
	# print "Data uji:", test_row
	
	# print "*******************************"
	# print "Batas perbandingan 1 baris data uji ke 38 baris codebooks"
	# print "*******************************"
	if distances[0][1] > 1.0:
		if distances[0][0][-1] == 0:
			distances[0][0][-1] = 1
		else:
			distances[0][0][-1] = 0
	# print "Codebook After:",distances[0][0]
	return distances[0][0]

def get_best_matching_unit_training(codebooks, test_row):
	distances = list()
	# print len(codebooks[0])
	# print len(test_row)
	i = 0
	for codebook in codebooks:
		# print len(codebooks)
		# print len(test_row)
		# print "codebook:-",i,": ", codebook
		# print "codebook:", codebook
		# print "test_row: ", test_row
		dist = euclidean_distance(codebook, test_row) #Nilai distance dihitung untuk setiap baris codebook (20 line) ke 1 baris data uji
		distances.append((codebook, dist))
		i+=1
		
	distances.sort(key=lambda tup: tup[1])
	# jarak ED antara 1 baris test uji terhadap semua baris di codeboooks disimpan di list distances.
	# list distances kemudian di urutkan (sorting) berdasarkan  nilai distance. 
	# data terurut dari yg terkecil ke yg terbesar
	# nilai distance paling kecil menunjukkan kedekatan test row dengan baris di codebooks yg paling baik
	# nilai distance terkecil tersebut dapat diakses dengan menggunakan distances[0][0]
	# print "Dist total: ",distances
	
	# print "*******************************"
	# print "Batas perbandingan 1 baris data uji ke 38 baris codebooks"
	# print "*******************************"

	return distances[0][0]


# Make a prediction with codebook vectors
def predict(codebooks, test_row):
	bmu = get_best_matching_unit(codebooks, test_row)
	# print "bmu - 1: ", bmu[-1]
	return bmu[-1]

# Create a random codebook vector / ini bikin Data Training
def random_codebook(train):
	n_records = len(train)
	n_features = len(train[0])
	codebook = [train[randrange(n_records)][i] for i in range(n_features)]
	return codebook

# Train a set of codebook vectors / ini Data Trainingnya, pake codebook library
def train_codebooks(train, n_codebooks, lrate, epochs):
	# codebooks = [random_codebook(train) for i in range(n_codebooks)]
	codebooks = train
	# print "Codebooks Before Train:", codebooks
	# print "************* Batas Awal Iterasi ***********"
	
	for epoch in range(epochs):
		rate = lrate * (1.0-(epoch/float(epochs)))
		j = 1
		# print "Codebooks:", codebooks
		codebooks_train=list()
		for row in train:
			# print "Row-",j,":", row
			bmu = get_best_matching_unit_training(codebooks, row)
			# print "bmu-",j,":", bmu
			j+=1
			for i in range(len(row)-1):
				error = row[i] - bmu[i]
				if bmu[-1] == row[-1]:
					bmu[i] += rate * error
				else:
					bmu[i] -= rate * error
			codebooks_train.append(bmu)
	# print "Codebooks After Train:", codebooks
	# print "BMU After Train:", codebooks_train
	# print "************* Batas Akhir Iterasi ***********"
	

	# return codebooks
	return codebooks_train

# LVQ Algorithm
def learning_vector_quantization(train, test, n_codebooks, lrate, epochs):
	# print "Train Data: ", train
	# print "Test data: ", test


	codebooks = train_codebooks(train, n_codebooks, lrate, epochs)
	# print "Ini data code books: ", codebooks
	# print "Ini data test: ", test
	predictions = list()
	for row in test:
		# print "Test in Loop: ", row
		output = predict(codebooks, row)
		predictions.append(output)
	# print "Hasil Prediksi:",predictions
	return(predictions)

# def readInputData(trainData, testData):
# 	print "Di MyLVQ: ", len(trainData)
# 	print "Di MyLVQ: ", len(testData)
# 	n_folds = 5
# 	learn_rate = 0.3
# 	n_epochs = 100
# 	n_codebooks = 20

# 	# scores = evaluate_algorithm(dataset, learning_vector_quantization, n_folds, n_codebooks, learn_rate, n_epochs)
# 	scores = evaluate_algorithm(trainData, learning_vector_quantization, n_folds, n_codebooks, learn_rate, n_epochs)
# 	# print('Scores: %s' % scores)
	# print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

## Load Data Dictionary
stop_words_glob = loadDict()

## Data Training Processing
train_df_raw_glob = loadDataTraining()
docs_train_glob = extractTweetTraining(train_df_raw_glob)
y_train_glob = extractTweetTrainingLabel(train_df_raw_glob)

# print docs_train_glob
# print y_train_glob

## Data Testing Processing
test_df_raw_glob = loadDataTest()
docs_test_glob = extractTweetTest(test_df_raw_glob)
y_test_glob = extractTweetTestLabel(test_df_raw_glob)

# print docs_test_glob
# print y_test_glob
# 
vectorizer_glob = vectorProcessing(stop_words_glob)

X_train=vectorizer_glob.fit_transform(docs_train_glob).toarray()
X_test=vectorizer_glob.transform(docs_test_glob).toarray()	
X_train_list=X_train.tolist()
X_test_list=X_test.tolist()


for i in range(len(X_train_list)):
	X_train_list[i].append(y_train_glob[i])
# print X_train_list[200]

dataset = X_train_list
# print "data set1: ",dataset
for i in range(len(X_test_list)):
	X_test_list[i].append(y_test_glob[i])
# print X_train_list[200]

datatest = X_test_list
# print "data test1: ",datatest
str_column_to_int(dataset, len(dataset[0])-1)
str_column_to_int_datatest(datatest, len(datatest[0])-1)
# print "data set2: ",dataset
# print "data test2: ",datatest
# evaluate algorithm
n_folds = 2
learn_rate = 0.3
# learn_rate = 0.5
n_epochs = 100
# n_codebooks = 20
n_codebooks = 5

scores = evaluate_algorithm(dataset, datatest, learning_vector_quantization, n_folds, n_codebooks, learn_rate, n_epochs)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
