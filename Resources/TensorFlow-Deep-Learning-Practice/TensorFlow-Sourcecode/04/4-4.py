import numpy as np									
import pylab										
import scipy.stats as stats								

data = np.mat([[1,200,105,3,False],[2,165,80,2,False],
               [3,184.5,120,2,False],[4,116,70.8,1,False],[5,270,150,4,True]])	  

col1 = []											
for row in data:										
    col1.append(row[0,1])							

stats.probplot(col1,plot=pylab)							
pylab.show()			
