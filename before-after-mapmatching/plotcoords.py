import csv
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

class Plotter():
    def __init__(self, csvfile):
        self.file = open('small_data.csv')
        csv_file = csv.reader(self.file)
        self.before_points, self.after_points = self.get_mapmatching_coords(csv_file)
        self.plotCoords(self.before_points)
    def get_mapmatching_coords(self,csv):
        before_points = []
        after_points = []
        for row in csv:
            #pair is (long,lat)
	        before_points_pair = [row[1],row[2]]
	        after_points_pair = [row[8], row[9]]
	        #print 'before' + str(before_points_pair)
	        #print after_points_pair
	        before_points.append(before_points_pair)
	        after_points.append(after_points_pair)
        return before_points, after_points
    def plotCoords(self,points):
        fig=plt.figure()
        ax=fig.add_axes([0.1,0.1,0.8,0.8])
        # setup mercator map projection.
        m = Basemap(llcrnrlon=112.,llcrnrlat=20.,urcrnrlon=115.,urcrnrlat=24.,\
                    resolution='i',projection='merc',\
                    lat_0=22.5431,lon_0=114.062996)
        m.drawcoastlines()
        m.fillcontinents()
        m.fillcontinents(color = 'coral')
        m.drawmapboundary()
        # draw parallels
        m.drawparallels(np.arange(10,90,20),labels=[1,1,0,1])
        # draw meridians
        m.drawmeridians(np.arange(-180,180,30),labels=[1,1,0,1])
        ax.set_title('Before MapMatching')
        for point in points:
            lon = float(point[0])
            lat = float(point[1])
            x,y = m(lon, lat)
            m.plot(x, y, 'bo', markersize=4)
        plt.show()



plotter = Plotter(".....")

