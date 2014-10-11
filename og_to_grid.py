''' http://scriptogr.am/jdp/post/pathfinding-with-python-graphs-and-a-star '''

from astargrid import AStarGrid, AStarGridNode
from itertools import product
from mapdef import mapdef
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def make_graph(mapinfo):
    nodes = [[AStarGridNode(x, y) for y in range(mapinfo['height'])] 
                            for x in range(mapinfo['width'])]
    graph = {}
    for x, y in product(range(mapinfo['width']), range(mapinfo['height'])):
        node = nodes[x][y]
        graph[node] = []
        for i, j in product([-1, 0, 1], [-1, 0, 1]):
            if not (0 <= x + i < mapinfo['width']): continue
            if not (0 <= y + j < mapinfo['height']): continue
            if not mapinfo['occ_grid'][y+j][x+i]: continue
            graph[nodes[x][y]].append(nodes[x+i][y+j])
    return graph, nodes
    
    
def plot_path(path
            , this_map
            ):
    
    start_x, start_y = path[0].x, path[0].y
    end_x, end_y = path[-1].x, path[-1].y
    path_x = [node.x for node in path]
    path_y = [node.y for node in path]
    plt.subplot(111)
    plt.cla()
    plt.plot(start_x, start_y, '*', color = 'y', markersize = 10)
    plt.plot(end_x, end_y, '*', color = 'r', markersize = 10)
    plt.imshow(this_map.grid
                ,cmap=cm.Greens_r
                ,interpolation = 'none'
                , origin='lower'
                )
    plt.xlim(0, this_map.N)
    plt.ylim(0, this_map.N)
    plt.plot(path_x, path_y, color = 'b')
    plt.draw() 

    
if __name__ == "__main__":
    this_map = mapdef()
    graph, nodes = make_graph({"width": this_map.N, "height": this_map.N, "occ_grid": this_map.grid})
    paths = AStarGrid(graph)
    start, end = nodes[25][25], nodes[95][95]
    path = paths.search(start, end)
    # path = [(node.x, node.y) for node in path]
    if path is None:
        print "No path found"
    else:
        print "Path found:", [(node.x, node.y) for node in path]
        plt.ion()
        plot_path(path, this_map)