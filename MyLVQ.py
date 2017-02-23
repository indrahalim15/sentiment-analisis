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
	lookup = dict()
	for i, value in enumerate(unique):
		if i == 0:
			lookup[value] = 1
		else:
			lookup[value] = 0
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

# Split a dataset into k folds
# def cross_validation_split(dataset, n_folds):
# 	dataset_split = list()
# 	dataset_copy = list(dataset)
# 	fold_size = int(len(dataset) / n_folds) #membagi data total sejumlah n_folds
# 	for i in range(n_folds):
# 		fold = list()
# 		while len(fold) < fold_size:
# 			index = randrange(len(dataset_copy)) 
# 			# print index
# 			fold.append(dataset_copy.pop(index)) #data fold diisi sebanyak 50 dari dataset dan data dipilih acak
# 		dataset_split.append(fold)
# 	return dataset_split

def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds) #membagi data total sejumlah n_folds
	for i in range(n_folds):
		fold = list()
		if i == 0:
			index = 0
		else:
			index = 50
		# print "Kelompok data: ", i
		while len(fold) < fold_size:
			# print index
			fold.append(dataset_copy[index]) #data fold diisi sebanyak 50 dari dataset dan data dipilih acak
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
# 		# print "actual: ", actual
# 		# print "predict: ", predicted
# 		accuracy = accuracy_metric(actual, predicted)
# 		scores.append(accuracy)
	
# 	return scores

	# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, datatest, algorithm, n_folds, *args):
	# print args
	folds_train = cross_validation_split(dataset, n_folds)
	folds_test = cross_validation_split(datatest, n_folds)
	

	# print "Len folds: ", len(folds)
	# print "Len fold atas: ", len(folds[0])
	
	count =0
	for x in dataset:
		# print len(x)
		count+=1
	# print "Jum Data", count
	scores = list()
	
	for fold_test in folds_test:
		test_set_fold = list(folds_test)
		test_set_fold.remove(fold_test)
		
		test_set = list()

		for row in test_set_fold:
				row_copy = list(row)
				# row_copy[-1] = None
				test_set.append(row_copy)
				# row_copy[-1] = None
		
		test_set_final = list(test_set[0])
		# print "Data Uji: ", test_set_final

		for fold_train in folds_train:
			train_set = list(folds_train)
			train_set.remove(fold_train) #fungsi untuk menghapus populasi total dengan populasi data yg lg di observasi (fold)
			train_set = sum(train_set, [])
			# print len(train_set[0]) #50 x 234
			# i = 0
			# for t in train_set:
			# 	print "data ke-",i," : ", len(t) # Cari tau kenapa jadi sama semua
			# 	i+=1
			
			# test_set = list()
			# # print len(folds)
			
			# for row in fold_train:
			# 	row_copy = list(row)
			# 	test_set.append(row_copy)
			# 	row_copy[-1] = None

			# print "Test:", len(test_set) #Jumlah Data Uji = 70 baris
			# print "Train:", len(train_set) #Jumlah Data Training = 280 baris
			# print len(train_set)
			# print len(test_set_final)
			# print len(datatest)
			# print "Data training: ", train_set
			predicted = algorithm(train_set, test_set_final, *args)
			# predicted = algorithm(train_set, datatest, *args)
			# print "row: ", row
			# print "Len fold: ", len(fold)
			# print "fold: ", fold
			# actual = [row[-1] for row in fold_train]
			# print "test: ", test_set_final
			# print "train: ", train_set
			actual = list()
			for row in fold_train:
				actual.append(row[-1])


			print "actual: ", actual
			print "predict: ", predicted
			accuracy = accuracy_metric(actual, predicted)
			# print "Akurasi: ", accuracy
			scores.append(accuracy)

	
	return scores

# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	# print len(row1)
	for i in range(len(row1)-1):
		# print "row1: ",row1[i]
		# print "row2: ",row2[i]
		# try:
		distance += (row1[i] - row2[i])**2
		# except Exception as e:
		# 	print "Error: ", e
		# 	print "data ke: ", i
		# 	print "row1: ", row1[i-1]
		# 	print "row2: ", row2[i-1]
	# print distance
	return sqrt(distance)

# Locate the best matching unit
def get_best_matching_unit(codebooks, test_row):
	distances = list()
	# print len(codebooks)
	for codebook in codebooks:
		dist = euclidean_distance(codebook, test_row) #Nilai distance dihitung untuk setiap baris codebook (20 line) ke 1 baris data uji
		distances.append((codebook, dist))
	distances.sort(key=lambda tup: tup[1])
	# print distances[0][0] #Nilai distance diwakili dng data distances[0][0]
	return distances[0][0]

# Make a prediction with codebook vectors
def predict(codebooks, test_row):
	bmu = get_best_matching_unit(codebooks, test_row)
	return bmu[-1]

# Create a random codebook vector / ini bikin Data Training
def random_codebook(train):
	n_records = len(train)
	n_features = len(train[0])
	codebook = [train[randrange(n_records)][i] for i in range(n_features)]
	return codebook

# Train a set of codebook vectors / ini Data Trainingnya, pake codebook library
def train_codebooks(train, n_codebooks, lrate, epochs):
	codebooks = [random_codebook(train) for i in range(n_codebooks)]
	
	for epoch in range(epochs):
		rate = lrate * (1.0-(epoch/float(epochs)))
		for row in train:
			bmu = get_best_matching_unit(codebooks, row)
			for i in range(len(row)-1):
				error = row[i] - bmu[i]
				if bmu[-1] == row[-1]:
					bmu[i] += rate * error
				else:
					bmu[i] -= rate * error
	return codebooks

# LVQ Algorithm
def learning_vector_quantization(train, test, n_codebooks, lrate, epochs):
	# print "data: ", train

	codebooks = train_codebooks(train, n_codebooks, lrate, epochs)
	predictions = list()
	for row in test:
		output = predict(codebooks, row)
		predictions.append(output)

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

vectorizer_glob = vectorProcessing(stop_words_glob)

X_train=vectorizer_glob.fit_transform(docs_train_glob).toarray()
X_test=vectorizer_glob.transform(docs_test_glob).toarray()	
X_train_list=X_train.tolist()
X_test_list=X_test.tolist()


for i in range(len(X_train_list)):
	X_train_list[i].append(y_train_glob[i])
# print X_train_list[200]

dataset = X_train_list

for i in range(len(X_test_list)):
	X_test_list[i].append(y_test_glob[i])
# print X_train_list[200]

datatest = X_test_list

str_column_to_int(dataset, len(dataset[0])-1)
str_column_to_int(datatest, len(datatest[0])-1)

# evaluate algorithm
n_folds = 2
learn_rate = 0.3
n_epochs = 100
n_codebooks = 20

scores = evaluate_algorithm(dataset, datatest, learning_vector_quantization, n_folds, n_codebooks, learn_rate, n_epochs)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
