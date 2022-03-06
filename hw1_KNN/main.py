from Load_the_Data import *
from Handmade_KNN import *
(dataset,labelset)=load_the_csv(r"D:\COMPUTER\Python\UESTC_Machine_Learning\hw1_KNN\iris.csv")
# get the data
for i in range(len(dataset[0])):
    conv_string_to_float(dataset,i)
normalize(dataset)

type_dic=conv_string_map_integer(labelset)
# process data

acc_1=test_acc(dataset,labelset,euclidean_distance,10,20)
acc_2=test_acc(dataset,labelset,euclidean_distance,15,20)
acc_3=test_acc(dataset,labelset,euclidean_distance,20,20)




