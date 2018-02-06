from pylab import *											
import pandas as pd											
import matplotlib.pyplot as plot									
filePath = ("c://dataTest.csv")									
dataFile = pd.read_csv(filePath,header=None, prefix="V")				

summary = dataFile.describe()									
minRings = -1												
maxRings = 99												
nrows = 10												
for i in range(nrows):											
    dataRow = dataFile.iloc[i,1:10]								
    labelColor = (dataFile.iloc[i,10] - minRings) / (maxRings - minRings)	
    dataRow.plot(color=plot.cm.RdYlBu(labelColor), alpha=0.5)		
plot.xlabel("Attribute")										
plot.ylabel("Score")											
show()			
