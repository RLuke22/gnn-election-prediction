import matplotlib.pyplot as plt 
import geopandas

""" class ElectionResultsMap():
    def __init__(self):
        super(ElectionResultsMap, self).__init__()

        self.states = geopandas.read_file('usa-states-census-2014.shp')
        
    def show(self):
        print("show")
        self.states.plot()
        plt.savefig('map.png')
        plt.show()

if __name__ == '__main__':
    map = ElectionResultsMap()
    map.show() """

states = geopandas.read_file('usa-states-census-2014.shp')
states.plot()
plt.savefig('map.png')
plt.show()