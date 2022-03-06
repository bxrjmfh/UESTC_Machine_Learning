import matplotlib.pyplot as plt
plt.plot(range(len(acc_1)),acc_1,'b',label='euclidean , ave ='+str('%.4f'%float(sum(acc_1)/len(acc_1))))
plt.plot(range(len(acc_2)),acc_2,'g',label='manhattan , ave ='+str('%.4f'%float(sum(acc_2)/len(acc_2))))
plt.plot(range(len(acc_3)),acc_3,'y',label='infinite_L , ave ='+str('%.4f'%float(sum(acc_3)/len(acc_3))))
plt.xlabel("batches")
plt.ylabel("acc")
plt.legend()
plt.title("Acc on different distance function")
plt.show()
# 测试训练精度


import matplotlib.pyplot as plt
plt.plot(range(len(acc_1)),acc_1,'r',label='k=10 , ave ='+str('%.4f'%float(sum(acc_1)/len(acc_1))))
plt.plot(range(len(acc_2)),acc_2,'m',label='k=15 , ave ='+str('%.4f'%float(sum(acc_2)/len(acc_2))))
plt.plot(range(len(acc_3)),acc_3,'c',label='k=20 , ave ='+str('%.4f'%float(sum(acc_3)/len(acc_3))))
plt.xlabel("batches")
plt.ylabel("acc")
plt.legend()
plt.title("Acc on different k value")
plt.show()
# 测试训练精度