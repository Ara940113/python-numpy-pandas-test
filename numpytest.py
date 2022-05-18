# python -m pip install numpy 
# python -m pip install pandas
# python -m pip install matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier



# csv 불러오기
bream_length = [pd.read_csv("bream_length.csv",header=None)]
bream_weight = [pd.read_csv("bream_weight.csv",header=None)]
smelt_length = [pd.read_csv("smelt_length.csv",header=None)]
smelt_weight = [pd.read_csv("smelt_weight.csv",header=None)]

# matplot으로 시각화
# plt.scatter(bream_length, bream_weight)
# plt.scatter(smelt_length, smelt_weight)
# plt.xlabel('length')
# plt.ylabel('weight')

# 1차원 배열로 만들기
length = np.concatenate((bream_length, smelt_length),axis=None)
weight = np.concatenate((bream_weight, smelt_weight),axis=None)
# print(length)
# print(weight)

#2차원 배열 만들기
fish_data = np.column_stack((length, weight))
# print(fish_data)

# 타겟데이터 만들기
fish_target = [-1]*35 + [0]*14

# print(fish_target)


# 훈련데이터 만들기
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

# 셔플을 이용해 섞기
np.random.seed(42)
index = np.arange(49)
np.random.shuffle(index)
#print(index)

#테스트 데이터 훈련데이터 구분하기
train_input = input_arr [index[:35]]
train_target = target_arr [index[:35]]

test_input = input_arr [index[35:]]
test_target = target_arr [index[35:]]

# 훈련데이터 matplot 시각화
plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(test_input[:, 0], test_input[:, 1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()