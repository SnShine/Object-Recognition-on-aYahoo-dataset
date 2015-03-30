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
||Linear Kernel|RBF Kernel|Polynomial Kernel|LinearSVC|
|First 500 features of fc7 layer|||||
|First 2048 features of fc7 layer|||||
|All features of fc7 layer|||||
|First 500 features of fc8 layer|||||
|All features of fc8 layer|||||
|All features of fc7+fc8 layer|||||
####Using caffenet model