import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import time

caffe_root = "../../"  
import sys
sys.path.insert(0, caffe_root + "python")

import caffe

####################
##   Parameters   ##
####################

#model net to extract features
mynet= "alexnet"        #accepted values are alexnet, caffenet, googlenet
if mynet== "alexnet":
    deploy= "models/bvlc_alexnet/deploy.prototxt"
    model= "models/bvlc_alexnet/bvlc_alexnet.caffemodel"
elif mynet== "caffenet":
    deploy= "models/bvlc_reference_caffenet/deploy.prototxt"
    model= "models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"
elif mynet== "googlenet":
    deploy= "models/bvlc_googlenet/deploy.prototxt"
    model= "models/bvlc_googlenet/bvlc_googlenet.caffemodel"
else:
    print("Please give a proper net to extract!")

#layer of CNN to extract
mylayer= "fc7"         #accepted values are fc7, fc8, fc7+fc8


#########################
##   Global settings   ##
#########################

plt.rcParams["figure.figsize"] = (10, 10)
plt.rcParams["image.interpolation"] = "nearest"
plt.rcParams["image.cmap"] = "gray"

caffe.set_mode_cpu()
net = caffe.Net(caffe_root + deploy,
                caffe_root + model,
                caffe.TEST)

########################
##   Pre-Processing   ##
########################

transformer = caffe.io.Transformer({"data": net.blobs["data"].data.shape})
transformer.set_transpose("data", (2,0,1))
transformer.set_mean("data", np.load(caffe_root + "python/caffe/imagenet/ilsvrc_2012_mean.npy").mean(1).mean(1)) # mean pixel
transformer.set_raw_scale("data", 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap("data", (2,1,0))  # the reference model has channels in BGR order instead of RGB
net.blobs["data"].reshape(1,3,227,227)

train_feats= []

test_feats= []

#####################
##   Load Labels   ##
#####################

sttime= time.clock()

train_labels= []
test_labels= []

print("Loading labels from files...")
rdfile1= open("train_data.txt", "r")
rdfile2= open("test_data.txt", "r")

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
print("\t\t"+ str(len(test_labels))+" labels successfully loaded in "+str(time.clock()- sttime)+" Secs.\n")


##########################
##   Extract Features   ##
##########################

#train features
print("Extracting train features...")
sttime= time.clock()
train_file= open("train_data.txt","r")

if mylayer== "fc7":
    for line in train_file:
        print(line)
        line1= line.split()
        net.blobs["data"].data[...]= transformer.preprocess("data", caffe.io.load_image(line1[0]))
        out= net.forward()   
        feat= np.array(net.blobs["fc7"].data[0])
        train_feats.append(feat) 
    train_feats= np.array(train_feats)
elif mylayer== "fc8":
    for line in train_file:
        line1= line.split()
        net.blobs["data"].data[...]= transformer.preprocess("data", caffe.io.load_image(line1[0]))
        out= net.forward()
        feat= np.array(net.blobs["fc8"].data[0])
        train_feats.append(feat)
    train_feats= np.array(train_feats)
elif mylayer== "fc7+fc8":
    for line in train_file:
        line1= line.split()
        net.blobs["data"].data[...]= transformer.preprocess("data", caffe.io.load_image(line1[0]))
        out= net.forward()
        feat1= np.array(net.blobs["fc7"].data[0])
        feat2= np.array(net.blobs["fc8"].data[0])
        feat= np.append(feat1, feat2)
        train_feats.append(feat)
    train_feats= np.array(train_feats)

print("\t\tsuccessfully extracted in "+str(time.clock()- sttime)+" Secs.")


#test features
print("Extracting test features...")
sttime= time.clock()
test_file= open("test_data.txt", "r")

if mylayer== "fc7":
    for line in test_file:
        print(line)
        line1= line.split()
        net.blobs["data"].data[...]= transformer.preprocess("data", caffe.io.load_image(line1[0]))
        out= net.forward()   
        feat= np.array(net.blobs["fc7"].data[0])
        test_feats.append(feat) 
    test_feats= np.array(test_feats)
elif mylayer== "fc8":
    for line in train_file:
        line1= line.split()
        net.blobs["data"].data[...]= transformer.preprocess("data", caffe.io.load_image(line1[0]))
        out= net.forward()
        feat= np.array(net.blobs["fc8"].data[0])
        test_feats.append(feat) 
    test_feats= np.array(test_feats)
elif mylayer== "fc7+fc8":
    for line in train_file:
        line1= line.split()
        net.blobs["data"].data[...]= transformer.preprocess("data", caffe.io.load_image(line1[0]))
        out= net.forward()
        feat1= np.array(net.blobs["fc7"].data[0])
        feat2= np.array(net.blobs["fc8"].data[0])
        feat= np.append(feat1, feat2)
        test_feats.append(feat) 
    test_feats= np.array(test_feats)

print("\t\tsuccessfully extracted in "+str(time.clock()- sttime)+" Secs.")


######################
##   SVM Training   ##
######################

print("Starting to classify with svm")
C=1.0

print("\tfitting svc...")
sttime= time.clock()
svc = svm.SVC(kernel='linear', C=C).fit(train_feats[:,:500], train_labels)
print("\t\ttook "+str(time.clock()- sttime)+ " Secs.")

print("\tfitting rbf_svc...")
sttime= time.clock()
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(train_feats[:,:500], train_labels)
print("\t\ttook "+str(time.clock()- sttime)+ " Secs.")

print("\tfitting poly_svc...")
sttime= time.clock()
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(train_feats[:,:500], train_labels)
print("\t\ttook "+str(time.clock()- sttime)+ " Secs.")

print("\tfitting lin_svc...")
sttime= time.clock()
lin_svc = svm.LinearSVC(C=C).fit(train_feats[:,:500], train_labels)
print("\t\ttook "+str(time.clock()- sttime)+ " Secs.")


print("\n\nPredicting values...")
for i, clf in enumerate((svc, rbf_svc, poly_svc, lin_svc)):
    sttime= time.clock()
    print("\nClassifier: ")
    print(clf)
    Z = clf.predict(test_feats[:,:500])
    #print(Z)
    #print(test_labels)
    scores = (Z== test_labels)
    total = np.sum(scores)
    print("Accuracy: ")
    print("\t\t"+str(total/500.0))
    print("Time: ")
    print("\t\t"+ str(time.clock()- sttime)+ " Secs.")