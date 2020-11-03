import matplotlib.pyplot as plt 
import geopandas

class ElectionResultsMap():
    def __init__(self):
        super(ElectionResultsMap, self).__init__()

        self.states = geopandas.read_file('usa-states-census-2014.shp')
        self.states = self.states.to_crs("EPSG:3395")
        
    def show(self):
        fig = plt.figure(1, figsize=(12, 12))
        ax = fig.add_subplot()

        self.states.boundary.plot(ax=ax, color='Black', linewidth=0.4)
        self.states.plot(ax=ax, color='whitesmoke', figsize=(12, 12))

        self.states[self.states['NAME']=='Texas'].plot(ax = ax, color='tab:red')
        self.states[self.states['NAME']=='Ohio'].plot(ax = ax, color='tab:blue')
        
        plt.savefig('map.png')
        plt.show()

if __name__ == '__main__':
    map = ElectionResultsMap()
    map.show()