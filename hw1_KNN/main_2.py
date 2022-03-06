import keras.datasets.mnist
from Load_the_Data import *
from Handmade_KNN import *
import tensorflow.keras

def trans_3D_to_list(train_data):
    flatten_list=[]
    for data in train_data:
        flatten_list.append(data.flatten().tolist())
    return flatten_list

(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()
x_test=x_test[:500]
y_test=y_test[:500]
dataset=trans_3D_to_list(x_test)
labelset=y_test.tolist()
acc_1=test_acc(dataset,labelset,euclidean_distance,5)
acc_2=test_acc(dataset,labelset,manhattan_distance,5)
acc_3=test_acc(dataset,labelset,infinite_L_distance,5)




