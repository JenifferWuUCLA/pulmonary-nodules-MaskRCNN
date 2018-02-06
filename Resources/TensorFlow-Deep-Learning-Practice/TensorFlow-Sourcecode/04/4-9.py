from pylab import *											
import pandas as pd											
import matplotlib.pyplot as plot									
filePath = ("c://dataTest.csv")									
dataFile = pd.read_csv(filePath,header=None, prefix="V")				

print(dataFile.head())										
print((dataFile.tail())											

summary = dataFile.describe()									
print(summary)												

array = dataFile.iloc[:,10:16].values								
boxplot(array)												
plot.xlabel("Attribute")										
plot.ylabel(("Score"))											
show()	
