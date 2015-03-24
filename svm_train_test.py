import lmdb
import caffe.proto.caffe_pb2 as cpb
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import scipy.io as sio

fc7_train_alexnet= []
fc7_test_alexnet= []
fc8_train_alexnet= []
fc8_test_alexnet= []
fc7_train_caffenet= []
fc7_test_caffenet= []
fc8_train_caffenet= []
fc8_test_caffenet= []

train_labels= []
test_labels= []


#####################
##   Load Labels   ##
#####################

print("Loading labels from files...")
rdfile1= open("train_data.txt", "r")
rdfile2= open("test_data.txt", "r")
write_file= open("main_output.txt", "w")

#to load train_labels from file
for line in rdfile1:
	line= line[:-1]
	train_labels.append(int(line.split()[1]))

train_labels= np.array(train_labels)

#to load test labels from file
for line in rdfile2:
	line= line[:-1]
	test_labels.append(int(line.split()[1]))

test_labels= np.array(test_labels)
print("\t\t"+ str(len(test_labels))+" labels loaded.\n")


#######################
##   Load Features   ##
#######################

###alexnet
##fc7
#train
print("Loading features from 'train_fc7_alexnet' database...")
db1 = lmdb.open('train_fc7_alexnet')
txn1 = db1.begin()
kvpairs1 = list(txn1.cursor().iternext(keys=True, values=True))

blob1 = cpb.Datum()
for key, value in kvpairs1:
    blob1.ParseFromString(value)
    feature_vector = np.array(blob1.float_data)
    fc7_train_alexnet.append(feature_vector)

fc7_train_alexnet= np.array(fc7_train_alexnet)		#converting back to np array
#print(type(fc7_train_alexnet))
print("\t\tfeatures successfully saved in fc7_train_alexnet\n")

#test
print("Loading features from 'test_fc7_alexnet' database...")
db2 = lmdb.open('test_fc7_alexnet')
txn2 = db2.begin()
kvpairs2 = list(txn2.cursor().iternext(keys=True, values=True))

blob2 = cpb.Datum()
for key, value in kvpairs2:
    blob2.ParseFromString(value)
    feature_vector = np.array(blob2.float_data)
    fc7_test_alexnet.append(feature_vector)

fc7_test_alexnet= np.array(fc7_test_alexnet)		#converting back to np array
#print(type(fc7_test_alexnet))
print("\t\tfeatures successfully saved in fc7_test_alexnet\n")


##fc8
#train
print("Loading features from 'train_fc8_alexnet' database...")
db3 = lmdb.open('train_fc8_alexnet')
txn3 = db3.begin()
kvpairs3 = list(txn3.cursor().iternext(keys=True, values=True))

blob3 = cpb.Datum()
for key, value in kvpairs3:
    blob3.ParseFromString(value)
    feature_vector = np.array(blob3.float_data)
    fc8_train_alexnet.append(feature_vector)

fc8_train_alexnet= np.array(fc8_train_alexnet)		#converting back to np array
#print(type(fc8_train_alexnet))
print("\t\tfeatures successfully saved in fc8_train_alexnet\n")

#test
print("Loading features from 'test_fc8_alexnet' database...")
db4 = lmdb.open('test_fc8_alexnet')
txn4 = db4.begin()
kvpairs4 = list(txn4.cursor().iternext(keys=True, values=True))

blob4 = cpb.Datum()
for key, value in kvpairs4:
    blob4.ParseFromString(value)
    feature_vector = np.array(blob4.float_data)
    fc8_test_alexnet.append(feature_vector)

fc8_test_alexnet= np.array(fc8_test_alexnet)		#converting back to np array
#print(type(fc8_test_alexnet))
print("\t\tfeatures successfully saved in fc8_test_alexnet\n")



###caffenet
##fc7
#train
print("Loading features from 'train_fc7_caffenet' database...")
db5 = lmdb.open('train_fc7_caffenet')
txn5 = db5.begin()
kvpairs5 = list(txn5.cursor().iternext(keys=True, values=True))

blob5 = cpb.Datum()
for key, value in kvpairs5:
    blob5.ParseFromString(value)
    feature_vector = np.array(blob5.float_data)
    fc7_train_caffenet.append(feature_vector)

fc7_train_caffenet= np.array(fc7_train_caffenet)		#converting back to np array
#print(type(fc7_train_caffenet))
print("\t\tfeatures successfully saved in fc7_train_caffenet\n")

#test
print("Loading features from 'test_fc7_caffenet' database...")
db6 = lmdb.open('test_fc7_caffenet')
txn6 = db6.begin()
kvpairs6 = list(txn6.cursor().iternext(keys=True, values=True))

blob6 = cpb.Datum()
for key, value in kvpairs6:
    blob6.ParseFromString(value)
    feature_vector = np.array(blob6.float_data)
    fc7_test_caffenet.append(feature_vector)

fc7_test_caffenet= np.array(fc7_test_caffenet)		#converting back to np array
#print(type(fc7_test_caffenet))
print("\t\tfeatures successfully saved in fc7_test_caffenet\n")


##fc8
#train
print("Loading features from 'train_fc8_caffenet' database...")
db7 = lmdb.open('train_fc8_caffenet')
txn7 = db7.begin()
kvpairs7 = list(txn7.cursor().iternext(keys=True, values=True))

blob7 = cpb.Datum()
for key, value in kvpairs7:
    blob7.ParseFromString(value)
    feature_vector = np.array(blob7.float_data)
    fc8_train_caffenet.append(feature_vector)

fc8_train_caffenet= np.array(fc8_train_caffenet)		#converting back to np array
#print(type(fc8_train_caffenet))
print("\t\tfeatures successfully saved in fc8_train_caffenet\n")

#test
print("Loading features from 'test_fc8_caffenet' database...")
db8 = lmdb.open('test_fc8_caffenet')
txn8 = db8.begin()
kvpairs8 = list(txn8.cursor().iternext(keys=True, values=True))

blob8 = cpb.Datum()
for key, value in kvpairs8:
    blob8.ParseFromString(value)
    feature_vector = np.array(blob8.float_data)
    fc8_test_caffenet.append(feature_vector)

fc8_test_caffenet= np.array(fc8_test_caffenet)		#converting back to np array
#print(type(fc8_test_caffenet))
print("\t\tfeatures successfully saved in fc8_test_caffenet\n")



##############
#SVM Training#
##############

print('Taking all the features of fc7 layer of alexnet... to train svm')
C=1.0
svc = svm.SVC(kernel='linear', C=C).fit(fc7_train_alexnet, train_labels)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(fc7_train_alexnet, train_labels)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(fc7_train_alexnet, train_labels)
lin_svc = svm.LinearSVC(C=C).fit(fc7_train_alexnet, train_labels)

for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    print("Classifier: ", clf)
    Z = clf.predict(fc7_test_alexnet)
    scores = (Z==test_labels)
    summation = np.sum(scores)
    print("Accuracy",summation/536.0)


print('Taking all the features of fc8 layer of alexnet... to train svm')
C=1.0
svc = svm.SVC(kernel='linear', C=C).fit(fc8_train_alexnet, train_labels)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(fc8_train_alexnet, train_labels)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(fc8_train_alexnet, train_labels)
lin_svc = svm.LinearSVC(C=C).fit(fc8_train_alexnet, train_labels)

for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    print("Classifier: ", clf)
    Z = clf.predict(fc8_test_alexnet)
    scores = (Z==test_labels)
    summation = np.sum(scores)
    print("Accuracy",summation/536.0)


print('Taking all the features of fc7 layer of caffenet... to train svm')
C=1.0
svc = svm.SVC(kernel='linear', C=C).fit(fc7_train_caffenet, train_labels)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(fc7_train_caffenet, train_labels)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(fc7_train_caffenet, train_labels)
lin_svc = svm.LinearSVC(C=C).fit(fc7_train_caffenet, train_labels)

for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    print("Classifier: ", clf)
    Z = clf.predict(fc7_test_caffenet)
    scores = (Z==test_labels)
    summation = np.sum(scores)
    print("Accuracy",summation/536.0)


print('Taking all the features of fc8 layer of caffenet... to train svm')
C=1.0
svc = svm.SVC(kernel='linear', C=C).fit(fc8_train_caffenet, train_labels)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(fc8_train_caffenet, train_labels)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(fc8_train_caffenet, train_labels)
lin_svc = svm.LinearSVC(C=C).fit(fc8_train_caffenet, train_labels)

for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    print("Classifier: ", clf)
    Z = clf.predict(fc8_test_caffenet)
    scores = (Z==test_labels)
    summation = np.sum(scores)
    print("Accuracy",summation/536.0)
