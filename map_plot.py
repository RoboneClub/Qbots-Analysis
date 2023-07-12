#Importing libraries
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#Reading the csv file
data = pd.read_csv("Depth.csv")

map = Basemap()
map.bluemarble()

#Plotting a map of the mission path
plt.scatter(data['Longitude'],data['Latitude'],s=2,c='y')
plt.title("Mission path",fontsize=20,weight='bold')
plt.show()