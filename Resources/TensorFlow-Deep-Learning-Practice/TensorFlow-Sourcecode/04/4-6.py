import pandas as pd											
import matplotlib.pyplot as plot									
filePath = ("c://dataTest.csv")									
dataFile = pd.read_csv(filePath,header=None, prefix="V")				

dataRow1 = dataFile.iloc[100,1:300]							
dataRow2 = dataFile.iloc[101,1:300]							
plot.scatter(dataRow1, dataRow2)								
plot.xlabel("Attribute1")										
plot.ylabel("Attribute2")										
plot.show()	
