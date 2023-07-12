#Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

#Reading the CSV file
data = pd.read_csv("Depth.csv")

#Creating a mask to get the underwater values only
depth_mask = data['OceanDepth'] <= 0

#Creating a box plot
mag_df = pd.melt(data[['MagX','MagY','MagZ']])
box = sns.boxplot(data=mag_df,x='variable', y='value')
plt.show()

#Applying the mask to the magnetic field data
magX = (data['MagX'])[depth_mask]
magY = (data['MagY'])[depth_mask]
magZ = (data['MagZ'])[depth_mask]
depth = (data['OceanDepth'])[depth_mask]


#Filtering the magnetic field data,to reduce noise
data['median_MagX'] = magX.rolling(3000).median()
data['median_MagZ'] = magZ.rolling(3000).median()
data['median_MagY'] = magY.rolling(3000).median()

#Calculating the resultant magnetic field
Mag_Res = (magX**2 + magY**2 + magZ**2)**0.5


#Creating a scatter plot for magX and depth
plt.scatter(magX,depth,alpha=0.07)
plt.title("Magnetic Feild X")
plt.ylabel("Depth")
plt.xlabel("Magnetic Feild Strength/T")
plt.show()


#Creating a scatter plot for magY and depth
plt.scatter(magY,depth,alpha=0.07)
plt.title("Magnetic Feild Y")
plt.ylabel("Depth")
plt.xlabel("Magnetic Feild Strength/T")
plt.show()


#Creating a scatter plot for magZ and depth
plt.scatter(magZ,depth,alpha=0.07)
plt.title("Magnetic Feild Z")
plt.ylabel("Depth")
plt.xlabel("Magnetic Feild Strength/T")
plt.show()

#Creating a scatter plot for the resultant magnetic field and depth
plt.scatter(Mag_Res,depth,alpha=0.07)
plt.title("Magnetic Feild Resultant")
plt.ylabel("Depth")
plt.xlabel("Magnetic Feild Strength/T")
plt.show()