import csv
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np

class Plotter():

    DEFAULT_LINES_READ = 10000;

    def __init__(self, csvfile, lines_read=DEFAULT_LINES_READ):
        self.file = open('all_data.csv')
        self.lines_read = lines_read
        csv_file = csv.reader(self.file)
        self.before_points, self.after_points = self.get_mapmatching_coords(csv_file, self.lines_read)
        self.plot_coords(self.before_points, self.after_points)
        #self.plot_coords_before(self.before_points) //shows only before points map
        #self.plot_coords_after(self.after_points) //shows only after points map

    def get_mapmatching_coords(self,csv,lines_read):
        before_points = []
        after_points = []
        rowCounter = 0
        for row in csv:
            rowCounter += 1
            print rowCounter
            if rowCounter == lines_read:
                break
            before_points_pair = [row[1],row[2]]
            after_points_pair = [row[8], row[9]]
            before_points.append(before_points_pair)
            print "before points" + str(before_points_pair)
            print "after points" + str(after_points_pair)
            after_points.append(after_points_pair)
        return before_points, after_points  

    def plot_coords(self, before_points, after_points):
        fig, axes = plt.subplots(2, 1)
        m = Basemap(projection='cass',
                    resolution='i',
                    lon_0 = 114.077183,
                    lat_0 = 22.662677,
                    width = 80000,
                    height = 40000,
                    ax=axes[0]);
        axes[0].set_title('Before MapMatching')
        m.bluemarble(scale = 0.5);
        m.drawcoastlines();
        for point in before_points:
            lon = float(point[0])
            lat = float(point[1])
            x,y = m(lon, lat)
            m.plot(x, y, 'bo', markersize=.25)
        m = Basemap(projection='cass',
                    resolution='i',
                    lon_0 = 114.077183,
                    lat_0 = 22.662677,
                    width = 80000,
                    height = 40000,
                    ax=axes[1]);
        axes[1].set_title('After MapMatching')
        m.bluemarble(scale = 0.5);
        m.drawcoastlines();
        for point in after_points:
            lon = float(point[0])
            lat = float(point[1])
            x,y = m(lon, lat)
            m.plot(x, y, 'yo', markersize=.25)
        plt.show()

    def plot_coords_before(self,points):
        fig=plt.figure()
        ax=fig.add_axes([0.1,0.1,0.8,0.8])
        # setup map projection.
        m = Basemap(projection='cass',
                    resolution='i',
                    lon_0 = 114.077183,
                    lat_0 = 22.662677,
                    width = 120000,
                    height = 50000);
        m.bluemarble(scale = 0.5);
        m.drawcoastlines();
        ax.set_title('Before MapMatching')
        for point in points:
            lon = float(point[0])
            lat = float(point[1])
            x,y = m(lon, lat)
            m.plot(x, y, 'bo', markersize=.25)
        plt.show()

    def plot_coords_after(self,points):
        fig=plt.figure()
        ax=fig.add_axes([0.1,0.1,0.8,0.8])
        # setup map projection.
        m = Basemap(projection='cass',
                    resolution='i',
                    lon_0 = 114.077183,
                    lat_0 = 22.662677,
                    width = 120000,
                    height = 50000);
        m.bluemarble(scale = 0.5);
        m.drawcoastlines();
        ax.set_title('After MapMatching')
        for point in points:
            lon = float(point[0])
            lat = float(point[1])
            x,y = m(lon, lat)
            m.plot(x, y, 'yo', markersize=.25)
        plt.show()












plotter = Plotter(".....")

