# there are three ways to measure the distance
from math import sqrt

def euclidean_distance(x_1,x_2):
    distance=0.0
    for i in range(len(x_1)):
        distance+=(x_1[i]-x_2[i])**2
    return sqrt(distance)

def manhattan_distance(x_1,x_2):
    distance=0.0
    for i in range(len(x_1)):
        distance+=abs(x_1[i]-x_2[i])
    return distance

def infinite_x_distance(x_1,x_2):
    distance = 0.0
    for i in range(len(x_1)):
        temp=abs(x_1[i]-x_2[i])
        if temp>distance :
            distance=temp
    return distance

# get the k-nearest neighbors

def get_k_neighbors(train_datas,test_sample,k,distance_method):
    distances=[]
    for i,train_sample in enumerate(train_datas):
        distance=distance_method(train_sample,test_sample)
        distances.append((i,distance))
#     get the distance
    distances.sort(key=lambda x:x[1])
#     using the second index to sort
    neighbors=[]
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

def predict_sample(train_datas,test_sample,k,distance_method):
    neighbors=get_k_neighbors(train_datas,test_sample,distance_method=distance_method)
    out_puts=[neighbor[-1] for neighbor in neighbors ]
    prediction= max(out_puts,key=out_puts.count)
#     count 作为参数函数传递给 max的排序方法中




