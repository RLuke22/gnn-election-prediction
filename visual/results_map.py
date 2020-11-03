import matplotlib.pyplot as plt 
import geopandas

class ElectionResultsMap():
    def __init__(self):
        super(ElectionResultsMap, self).__init__()

        self.states = geopandas.read_file('usa-states-census-2014.shp')
        #self.states = self.states.to_crs("EPSG:3395")
        self.fig = plt.figure(1, figsize=(12, 12))
        self.ax = self.fig.add_subplot() 

        self.states.boundary.plot(ax=self.ax, color='Black', linewidth=0.4, alpha=0.3)
        self.states.plot(ax=self.ax, color='whitesmoke', figsize=(12, 12))

    def color_decided_states(self):
        # not Arizona, Florida, Iowa, Georgia, Ohio, Texas, North Carolina
        self.states[self.states['NAME']=='Alabama'].plot(ax=self.ax, color='tab:red')
        #self.states[self.states['NAME']=='Alaska'].plot(ax=self.ax, color='tab:red')
        self.states[self.states['NAME']=='Arkansas'].plot(ax=self.ax, color='tab:red')
        self.states[self.states['NAME']=='California'].plot(ax=self.ax, color='tab:blue')
        self.states[self.states['NAME']=='Colorado'].plot(ax=self.ax, color='tab:blue')
        self.states[self.states['NAME']=='Connecticut'].plot(ax=self.ax, color='tab:blue')
        self.states[self.states['NAME']=='Delaware'].plot(ax=self.ax, color='tab:blue')
        self.states[self.states['NAME']=='District of Columbia'].plot(ax=self.ax, color='tab:blue')
        #self.states[self.states['NAME']=='Hawaii'].plot(ax=self.ax, color='tab:blue')
        self.states[self.states['NAME']=='Idaho'].plot(ax=self.ax, color='tab:red')
        self.states[self.states['NAME']=='Illinois'].plot(ax=self.ax, color='tab:blue')
        self.states[self.states['NAME']=='Indiana'].plot(ax=self.ax, color='tab:red')
        self.states[self.states['NAME']=='Kansas'].plot(ax=self.ax, color='tab:red')
        self.states[self.states['NAME']=='Kentucky'].plot(ax=self.ax, color='tab:red')
        self.states[self.states['NAME']=='Louisiana'].plot(ax=self.ax, color='tab:red')
        self.states[self.states['NAME']=='Maine'].plot(ax=self.ax, color='tab:blue')
        self.states[self.states['NAME']=='Maryland'].plot(ax=self.ax, color='tab:blue')
        self.states[self.states['NAME']=='Massachusetts'].plot(ax=self.ax, color='tab:blue')
        self.states[self.states['NAME']=='Michigan'].plot(ax=self.ax, color='tab:blue')
        self.states[self.states['NAME']=='Minnesota'].plot(ax=self.ax, color='tab:blue')
        self.states[self.states['NAME']=='Mississippi'].plot(ax=self.ax, color='tab:red')
        self.states[self.states['NAME']=='Missouri'].plot(ax=self.ax, color='tab:red')
        self.states[self.states['NAME']=='Montana'].plot(ax=self.ax, color='tab:red')
        self.states[self.states['NAME']=='Nebraska'].plot(ax=self.ax, color='tab:red')
        self.states[self.states['NAME']=='Nevada'].plot(ax=self.ax, color='tab:blue')
        self.states[self.states['NAME']=='New Hampshire'].plot(ax=self.ax, color='tab:blue')
        self.states[self.states['NAME']=='New Jersey'].plot(ax=self.ax, color='tab:blue')
        self.states[self.states['NAME']=='New Mexico'].plot(ax=self.ax, color='tab:blue')
        self.states[self.states['NAME']=='New York'].plot(ax=self.ax, color='tab:blue')
        self.states[self.states['NAME']=='North Dakota'].plot(ax=self.ax, color='tab:red')
        self.states[self.states['NAME']=='Oklahoma'].plot(ax=self.ax, color='tab:red')
        self.states[self.states['NAME']=='Oregon'].plot(ax=self.ax, color='tab:blue') 
        self.states[self.states['NAME']=='Pennsylvania'].plot(ax=self.ax, color='tab:blue')
        self.states[self.states['NAME']=='Rhode Island'].plot(ax=self.ax, color='tab:blue')
        self.states[self.states['NAME']=='South Carolina'].plot(ax=self.ax, color='tab:red')
        self.states[self.states['NAME']=='South Dakota'].plot(ax=self.ax, color='tab:red')
        self.states[self.states['NAME']=='Tennessee'].plot(ax=self.ax, color='tab:red')
        self.states[self.states['NAME']=='Utah'].plot(ax=self.ax, color='tab:red')
        self.states[self.states['NAME']=='Vermont'].plot(ax=self.ax, color='tab:blue')
        self.states[self.states['NAME']=='Virginia'].plot(ax=self.ax, color='tab:blue')
        self.states[self.states['NAME']=='Washington'].plot(ax=self.ax, color='tab:blue')
        self.states[self.states['NAME']=='West Virginia'].plot(ax=self.ax, color='tab:red')
        self.states[self.states['NAME']=='Wisconsin'].plot(ax=self.ax, color='tab:blue')
        self.states[self.states['NAME']=='Wyoming'].plot(ax=self.ax, color='tab:red')

        #additional_states = '\n'.join(('* Alaska is Republican (Red)', '* Hawaii is Democrat (Blue)'))
        
        self.ax.text(0.02, 0.20, '* Alaska (3) is Republican', transform=self.ax.transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='square', facecolor='tab:red', alpha=0.9))
        self.ax.text(0.02, 0.10, '* Hawaii (4) is Democrat', transform=self.ax.transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='square', facecolor='tab:blue', alpha=0.9))

    def write_electoral_college_votes(self):
        electoral_votes_dict = { 'California': 55,
                                 'Texas': 38,
                                 'Florida': 29,
                                 'New York': 29,
                                 'Pennsylvania': 20,
                                 'Illinois': 20,
                                 'Ohio': 18,
                                 'Michigan': 16,
                                 'Georgia': 16,
                                 'North Carolina': 15,
                                 'New Jersey': 14,
                                 'Virginia': 13,
                                 'Washington': 12,
                                 'Arizona': 11,
                                 'Indiana': 11,
                                 'Massachusetts': 11,
                                 'Tennessee': 11,
                                 'Maryland': 10,
                                 'Minnesota': 10,
                                 'Missouri': 10,
                                 'Wisconsin': 10,
                                 'Alabama': 9,
                                 'Colorado': 9,
                                 'South Carolina': 9,
                                 'Kentucky': 8,
                                 'Louisiana': 8,
                                 'Connecticut': 7,
                                 'Oklahoma': 7,
                                 'Oregon': 7,
                                 'Arkansas': 6,
                                 'Iowa': 6,
                                 'Kansas': 6,
                                 'Mississippi': 6,
                                 'Nevada': 6,
                                 'Utah': 6,
                                 'Nebraska': 5,
                                 'New Mexico': 5,
                                 'West Virginia': 5,
                                 'Hawaii': 4,
                                 'Idaho': 4,
                                 'Maine': 4,
                                 'New Hampshire': 4,
                                 'Rhode Island': 4,
                                 'Alaska': 3,
                                 'Delaware': 3,
                                 'Montana': 3,
                                 'North Dakota': 3,
                                 'South Dakota': 3,
                                 'Vermont': 3,
                                 'Wyoming': 3 ,
                                 'District of Columbia': 3}

        self.states.apply(lambda x: self.ax.annotate(text="{}: {}".format(x.NAME, electoral_votes_dict[x.NAME]), xy=x.geometry.centroid.coords[0], fontsize=8, fontweight='bold'),axis=1)
    

    def color_predicted_states(self):
        self.states[self.states['NAME']=='Arizona'].plot(ax=self.ax, color='tab:blue')
        self.states[self.states['NAME']=='Iowa'].plot(ax=self.ax, color='tab:red')
        self.states[self.states['NAME']=='Florida'].plot(ax=self.ax, color='tab:blue')
        self.states[self.states['NAME']=='Georgia'].plot(ax=self.ax, color='tab:red')
        self.states[self.states['NAME']=='Ohio'].plot(ax=self.ax, color='tab:blue')
        self.states[self.states['NAME']=='Texas'].plot(ax=self.ax, color='tab:red')
        self.states[self.states['NAME']=='NorthCarolina'].plot(ax=self.ax, color='tab:blue')
        
    def show(self):        
        plt.savefig('7_states_map.png')
        plt.show()

if __name__ == '__main__':
    map = ElectionResultsMap()
    map.color_decided_states()
    map.write_electoral_college_votes()
    map.show()