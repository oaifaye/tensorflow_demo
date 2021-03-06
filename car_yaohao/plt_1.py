'''
Created on 2018年1月20日

@author: Administrator
'''
import matplotlib.pyplot as plt 

y_ave_data = [[0.16340],[0.14262],[0.11891],
          [0.11144],[0.10877],[0.10757],
          [0.11162],[0.12950],[0.16892],
          [0.26201],[0.33987],
          
          [0.25861],[0.13729],[0.15485],
          [0.18996],[0.22682],[0.23539],
          [0.24490],[0.24537],[0.22391],
          [0.19093],[0.18588],[0.19599],
          
          [0.19207],[0.20103],[0.24344],
          [0.28478],[0.32225],[0.34766],
          [0.35522],[0.35356],[0.32422],
          [0.26656],[0.27150],[0.28916],
          
          [0.28643],[0.31321],[0.34662],
          [0.37498],[0.39484],[0.35561],
          [0.22857],[0.25087],[0.28978],
          [0.30504],[0.29461],[0.27542]
        ]

y_min_data = [[0.10000],[0.10000],[0.10000],
          [0.10000],[0.10000],[0.10000],
          [0.10500],[0.12000],[0.16000],
          [0.24400],[0.25000],
          
          [0.10000],[0.11600],[0.14200],
          [0.18000],[0.20100],[0.21100],
          [0.22100],[0.20100],[0.15000],
          [0.15500],[0.17000],[0.18500],
          
          [0.18000],[0.19200],[0.23500],
          [0.27200],[0.30100],[0.31600],
          [0.32000],[0.30400],[0.20500],
          [0.23100],[0.25700],[0.27700],
          
          [0.27500],[0.30100],[0.33600],
          [0.36000],[0.34800],[0.25000],
          [0.19000],[0.23500],[0.27200],
          [0.27000],[0.25500],[0.24000]
        ]

#画图  
seed_plt = 0.042
x_plt = [[-seed_plt*23],[-seed_plt*22],[-seed_plt*21],[-seed_plt*20],[-seed_plt*19],[-seed_plt*18],[-seed_plt*17],[-seed_plt*16],[-seed_plt*15],[-seed_plt*14],[-seed_plt*13],
         [-seed_plt*12],[-seed_plt*11],[-seed_plt*10],[-seed_plt*9],[-seed_plt*8],[-seed_plt*7],[-seed_plt*6],[-seed_plt*5],[-seed_plt*4],[-seed_plt*3],[-seed_plt*2],[-seed_plt*1],
         [seed_plt*1],[seed_plt*2],[seed_plt*3],[seed_plt*4],[seed_plt*5],[seed_plt*6],[seed_plt*7],[seed_plt*8],[seed_plt*9],[seed_plt*10],[seed_plt*11],[seed_plt*12],
         [seed_plt*13],[seed_plt*14],[seed_plt*15],[seed_plt*16],[seed_plt*17],[seed_plt*18],[seed_plt*19],[seed_plt*20],[seed_plt*21],[seed_plt*22],[seed_plt*23],[seed_plt*24]
         ]
plt.figure()  
plt.plot(x_plt, y_ave_data, 'r-', lw = 5)#画预测的实线，红色  
plt.plot(x_plt, y_min_data, 'b-', lw = 5)#画预测的实线，红色  

x_year_first_mo = [[-seed_plt*12],[seed_plt*1],[seed_plt*13]]
y_year_first_mo = [[0.25861],[0.19207],[0.28643]]
plt.scatter(x_year_first_mo, y_year_first_mo)#画散点图  

plt.show()