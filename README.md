##Object-Recognition-on-aYahoo-dataset
Object recognition algorithm to classify 12 categories of aYahoo Image dataset trained in Caffe framework.


###Categories
Total Images- 2152

|Class|Train Images|Test Images|
|-----|------------:|-----------:|
|Bag|222|71|
|Building|174|52|
|Carriage|108|46|
|Centaur|38|13|
|Donkey|103|31|
|Goat|118|39|
|Jetski|134|41|
|Monkey|129|36|
|Mug|177|50|
|Statue|163|36|
|Wolf|142|41|
|Zebra|144|43|
|*All Classes*|*1652*|*500*|


###Results
####Using alexnet model
Accuracy in percentage obtained by taking various features of images and various training methods 
while using alexnet model.

|alexnet|Linear Kernel|RBF Kernel|Polynomial Kernel|LinearSVC|
|-------|------------:|---------:|----------------:|--------:|
|**First 500 features of fc7 layer**|91.6|35.6|84.4|90.2|
|**First 2048 features of fc7 layer**|93.8|35.6|87.8|93.4|
|**All features of fc7 layer**|**94.4**|35.6|88.4|94.2|
|**First 500 features of fc8 layer**|92.8|35.6|91.6|92.6|
|**All features of fc8 layer**|92.2|35.6|92.2|92.6|
|**All features of fc7+fc8 layer**|93.4|35.6|92.8|93.8|

####Using caffenet model
Accuracy in percentage obtained by taking various features of images and various training methods 
while using caffenet model.

|caffenet|Linear Kernel|RBF Kernel|Polynomial Kernel|LinearSVC|
|--------|------------:|---------:|----------------:|--------:|
|**First 500 features of fc7 layer**|90.2|35.6|84.4|90.2|
|**First 2048 features of fc7 layer**|92.8|35.6|87.8|93.4|
|**All features of fc7 layer**|93.4|35.6|88.4|**94.2**|
|**First 500 features of fc8 layer**|90.0|35.6|91.6|92.6|
|**All features of fc8 layer**|90.6|35.6|92.2|92.6|
|**All features of fc7+fc8 layer**|91.4|35.6|92.8|93.8|
