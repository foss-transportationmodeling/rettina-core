import networkx as nx
import fiona
import pickle as pkl
import pandas as pd
import os
#import matplotlib.pyplot as plt
from fiona.crs import from_epsg

#this_dir =  os.path.dirname(__file__)

def create_multiGraph(road_net_shp):
    '''
    converts a multigraph object from the line shapefile.
    Use this one not the multiDigraph one.
    After conversion dumps the multigraph as a pickle file in folder with other
    important input files
    '''
    G = nx.MultiGraph()
    link_id_len = {}
    with fiona.open(road_net_shp,crs= from_epsg(32618)) as shp:
        #driver='ESRI Shapefile'
        node_coord_to_id_dict = {}
        node_id = 0
        for elem in shp:
            strt = elem['geometry']['coordinates'][0]
            len_coords = len(elem['geometry']['coordinates'])
            end = elem['geometry']['coordinates'][len_coords-1]
            for node in [strt, end]:
                if node not in node_coord_to_id_dict:
                    node_coord_to_id_dict[node] = node_id
                    node_id += 1
            length_meter = elem['properties']['Length_met']
            link_id_len[int(elem['id'])] = length_meter
            G.add_edge(strt, end,
                       len = length_meter,
                       key= int(elem['id']))
    link_id_ser = pd.Series(link_id_len.values(), index=link_id_len.keys())
    #node_id_to_coord_reverse = dict(zip(node_coord_to_id_dict.values(),
    #                                 node_coord_to_id_dict.keys()))
    nx.set_node_attributes(G, 'node_id', node_coord_to_id_dict)
    return G, link_id_ser

def create_multiDiGraph(road_net_shp):
    '''
    Use multigraph, not this one. This function was written for experimentation
    '''
    G = nx.MultiDiGraph()
    with fiona.open(road_net_shp,crs= from_epsg(32618)) as shp:
        #driver='ESRI Shapefile'
        for elem in shp:
            strt = elem['geometry']['coordinates'][0]
            len_coords = len(elem['geometry']['coordinates'])
            end = elem['geometry']['coordinates'][len_coords-1]
            length_meter = elem['properties']['Length_met']
            G.add_edge(strt, end, weight = length_meter)
            G.add_edge(end, strt, weight = length_meter)
    '''print len(G), G.number_of_nodes(),G.number_of_edges()
    nx.draw_graphviz(G)
    plt.show()'''
    return G

if __name__ == '__main__':
    road_net_shp = os.path.join(r'../../../data/Relevant_files',
                                'LineString_Road_Network_UTM.shp')

    G, link_id_ser = create_multiGraph(road_net_shp)
    pkl.dump(G, open('../../../data/Relevant_files/MultiGraph.p', 'wb'))
    pkl.dump(link_id_ser,
             open('../../../data/Relevant_files/LinkIdLenSer.p', 'wb'))
    #MDG = create_multiDiGraph(road_net_shp)
    #pkl.dump(MDG, open('../Relevant_files/MultiDiGraph.p', 'wb'))
    #print nx.dijkstra_path(G, (732805.0271117099, 4623614.515877712),
    #                              (729658.6308007092, 4630691.33944068), 'weight')

    #print G.neighbors((729658.6308007092, 4630691.33944068))
    #print G.neighbors((732805.0271117099, 4623614.515877712))

    #print G.degree([(732805.0271117099, 4623614.515877712),(732805.0271117099, 4623614.515877712)])
