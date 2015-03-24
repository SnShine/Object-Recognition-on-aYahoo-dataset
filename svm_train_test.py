import lmdb
import caffe.proto.caffe_pb2 as cpb
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import scipy.io as sio

#Global parameters
net= "caffenet"         #values are "alexnet", "caffenet", "googlenet"
layer= "fc7"            #values are "fc7", "fc8", "fc7+fc8"
top_x= 3                #compare the results with the top x predicted models      


train_features= []
test_features= []

train_labels= []
test_labels= []


#####################
##   Load Labels   ##
#####################

print("Loading labels from files...")
rdfile1= open("train_data.txt", "r")
rdfile2= open("test_data.txt", "r")
write_file= open("output_svmTrainTest.txt", "w")

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
print("\t\t"+ str(len(test_labels))+" successfully labels loaded.\n")


#######################
##   Load Features   ##
#######################

#train
print("Loading features from train_"+layer+"_"+net+ " database...")
db1 = lmdb.open("train_"+layer+"_"+net)
txn1 = db1.begin()
kvpairs1 = list(txn1.cursor().iternext(keys=True, values=True))

blob1 = cpb.Datum()
for key, value in kvpairs1:
    blob1.ParseFromString(value)
    feature_vector = np.array(blob1.float_data)
    train_features.append(feature_vector)

train_features= np.array(train_features)        #converting back to np array
print("\t\tfeatures successfully saved!\n")

#test
print("Loading features from test_"+layer+"_"+net+ " database...")
db2 = lmdb.open("test_"+layer+"_"+net)
txn2 = db2.begin()
kvpairs2 = list(txn2.cursor().iternext(keys=True, values=True))

blob2 = cpb.Datum()
for key, value in kvpairs2:
    blob2.ParseFromString(value)
    feature_vector = np.array(blob2.float_data)
    test_features.append(feature_vector)

test_features= np.array(test_features)          #converting back to np array
print("\t\tfeatures successfully saved!\n")


######################
##   SVM Training   ##
######################

print("Taking all the features of "+layer+" layer of "+net+" to train svm...")
C=0.7
print("\tfitting rbf_svc...")
rbf_svc = svm.SVC(kernel= "rbf", cache_size=1000, gamma= 0.4, C= C).fit(train_features, train_labels)
print("\tfitting poly_svc...")
poly_svc = svm.SVC(kernel= "poly", cache_size=1000, degree= 2, C= C).fit(train_features, train_labels)
print("\tfitting sigm_svc...")
sigm_svc = svm.SVC(kernel= "sigmoid", cache_size=1000, gamma= 0.4, C= C).fit(train_features, train_labels)
print("\tfitting lin_svc...")
lin_svc = svm.LinearSVC(C= C).fit(train_features, train_labels)

print("\n\nPredicting values...")
for i, clf in enumerate((rbf_svc, poly_svc, sigm_svc, lin_svc)):
    print("\nClassifier: ")
    print(clf)
    Z = clf.predict(test_features)
    scores = (Z==test_labels)
    total = np.sum(scores)
    print("Accuracy: ")
    print("\t\t"+str(total/536.0))
