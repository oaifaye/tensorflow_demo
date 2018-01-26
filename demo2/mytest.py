'''
Created on 2017年12月27日

@author: Administrator
'''


# 矩阵加法
def madd(M1, M2):
    if isinstance(M1, (tuple, list)) and isinstance(M2, (tuple, list)):
        return [[m+n for m,n in zip(i,j)] for i, j in zip(M1,M2)]

a = [[1,1],[2,2],[3,3]]
b = [2,2]
print(a,b)