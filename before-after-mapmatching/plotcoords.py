import csv
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

class Plotter():
    def __init__(self, csvfile):
        self.file = open('all_data.csv')
        csv_file = csv.reader(self.file)
        self.before_points, self.after_points = self.get_mapmatching_coords(csv_file)
        self.plot_coords_before(self.before_points)

    def get_mapmatching_coords(self,csv):
        before_points = []
        after_points = []
        rowCounter = 0
        for row in csv:
        	rowCounter += 1
        	print rowCounter
        	if rowCounter == 10000:
        		break
        	before_points_pair = [row[1],row[2]]
	        after_points_pair = [row[8], row[9]]
	        #print 'before' + str(before_points_pair)
	        #print after_points_pair
	        before_points.append(before_points_pair)
	        after_points.append(after_points_pair)
        print "done get mapmatching coords"
        return before_points, after_points  

    def plot_coords_before(self,points):
        fig=plt.figure()
        ax=fig.add_axes([0.1,0.1,0.8,0.8])
        # setup mercator map projection.
        m = Basemap(projection='cass',
                resolution='i',
              lon_0 = 114.173847,
              lat_0 = 22.543179,
              width = 200000,
              height = 100000);
        m.bluemarble(scale = 0.25);
        m.drawcoastlines();
        ax.set_title('Before MapMatching')
        for point in points:
            lon = float(point[0])
            lat = float(point[1])
            x,y = m(lon, lat)
            m.plot(x, y, 'bo', markersize=2)
        plt.show()



plotter = Plotter(".....")

