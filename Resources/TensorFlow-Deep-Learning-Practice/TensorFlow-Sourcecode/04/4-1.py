import numpy as np									
data = np.mat([[1,200,105,3,False],[2,165,80,2,False],
               [3,184.5,120,2,False],[4,116,70.8,1,False],[5,270,150,4,True]])	  
row = 0											
for line in data:										
    row += 1										
print( row	)										
print( data.size	)	
