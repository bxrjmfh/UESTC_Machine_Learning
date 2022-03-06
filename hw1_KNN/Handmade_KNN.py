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

def infinite_L_distance(x_1,x_2):
    distance = 0.0
    for i in range(len(x_1)):
        temp=abs(x_1[i]-x_2[i])
        if temp>distance :
            distance=temp
    return distance

# get the k-nearest neighbors

def get_k_neighbors(train_data,test_sample,k,distance_method):
    distances=[]
    for i,train_sample in enumerate(train_data):
        distance=distance_method(train_sample,test_sample)
        distances.append((i,distance))
#     get the distance
    distances.sort(key=lambda x:x[1])
#     using the second index to sort
    neighbors=[]
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors

def predict_sample(train_datas,train_label,test_sample,k,distance_method):
    neighbors=get_k_neighbors(train_datas,test_sample,k,distance_method=distance_method)
    out_puts=[train_label[neighbor] for neighbor in neighbors ]
    return max(out_puts,key=out_puts.count)
#     count 作为参数函数传递给 max的排序方法中

def caculate_acc(results,test_label):
    counter =0
    length =len(results)
    for i in range(length):
        if results[i]==test_label[i]:
            counter+=1

    return float(counter/length)

# to use the mini data set
def n_fold_validation(dataset,labelset,n_fold,i):
    fold_length=len(dataset)//n_fold
    test_data=dataset[i*fold_length:(i+1)*fold_length]
    train_data=dataset[:i*fold_length]+dataset[(i+1)*fold_length:]
    test_label=labelset[i*fold_length:(i+1)*fold_length]
    train_label=labelset[:i*fold_length]+labelset[(i+1)*fold_length:]
    return train_data , test_data,train_label,test_label

def test_acc(dataset,labelset,distance_method,k,n_fold=10):
    accs = []
    for i in range(10):
        (train_data, test_data,
         train_label, test_label) = n_fold_validation(dataset, labelset, n_fold, i)
        results = []
        for test_sample in test_data:
            results.append(predict_sample(train_data, train_label, test_sample,k, distance_method))

        acc_each_batch = caculate_acc(results, test_label)
        accs.append(acc_each_batch)
    return accs
