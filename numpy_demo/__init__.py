import numpy as np

a = np.array([[-1,2],[2,3]])
b = np.array([[3,4],[4,5]])
print ('\n a:\n',a)
print ('\n b:\n',b)

##转置
print ('\n a transpose:\n',a.T)

##共扼矩阵
#print '\n a H:\n',a.I

##逆矩阵
print ('\n a inv:\n',np.linalg.inv(a)) # 求逆

##转置
print ('\n a transpose:\n',a.T)

# a + b，矩阵相加
print ("\n a+b: \n",a+b)

# a - b，矩阵相减
print ("\n a-b: \n",a-b)

#2x2 矩阵，矩阵相乘
print ("\n a mul b:\n",a.dot(b.T))

#2x2 矩阵，乘以数字
print ("\n a mul数字\n",a.dot([-1,-1]))

#2x3矩阵，矩阵点乘
print ("\n a dot b: \n",a*b)

#2x3矩阵，矩阵点除
print ("\n a/b \n:",a/np.linalg.inv(b))

#求迹
print ("\n a trace",np.trace(a) )

#特征，特征向量
eigval,eigvec = np.linalg.eig(a) 
#eigval = np.linalg.eigvals(a) #直接求解特征值

print ("\n a eig value:\n",eigval,)
print ('\n a eig vector:\n',eigvec)

c = [[ 0.27158719]
 [ 0.33287081]
 [ 0.36050054]
 [ 0.36116984]
 [ 0.36008123]
 [ 0.3884801 ]
 [ 0.2436851 ]
 [ 0.24764147]
 [ 0.2742694 ]
 [ 0.29212099]
 [ 0.30045867]
 [ 0.28454551]]
print(np.round(c, 5))