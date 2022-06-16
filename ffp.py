import random
import math
from collections import defaultdict
# Provides the methods to create and solve the firefighter problem

# Function to find the shortest
# path between two nodes of a graph
def BFS_SP(graph, start, goal):
    explored = []
     
    # Queue for traversing the
    # graph in the BFS
    queue = [[start]]
     
    # If the desired node is
    # reached
    if start == goal:
        print("Same Node")
        return queue
     
    # Loop to traverse the graph
    # with the help of the queue
    while queue:
        path = queue.pop(0)
        node = path[-1]
         
        # Condition to check if the
        # current node is not visited
        if node not in explored:
            neighbours = graph[node]
             
            # Loop to iterate over the
            # neighbours of the node
            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)
                 
                # Condition to check if the
                # neighbour node is the goal
                if neighbour == goal:
                    # print("Shortest path = ", *new_path)
                    return new_path
            explored.append(node)
 
    # Condition when the nodes
    # are not connected
    print("So sorry, but a connecting"\
                "path doesn't exist :(")
    return None


#Provide a class to save the Node information to increase readibility
class Node:
    def __init__(self, id) -> None:
        self.id = id    # Node Id [String]
        self.degree = 0 # Degree of the Node
        self.state = None   # State of the Node
                            #   -1 On fire
                            #   0 Available for analysis
                            #   1 Protected
        self.adjacent_nodes = None  # List of the adjacent nodes
        self.distance_to_burning_node = -1  # Distance to the closest burning node
        self.next_to_burning_node = False
        pass
    
    def update_state(self,new_state):
        self.state = new_state
    
    def update_degree(self, new_degree):
        self.degree = new_degree
    
    def update_adjacent_nodes(self, n_adjacent_nodes):
        self.adjacent_nodes = n_adjacent_nodes
    
    def update_distance_to_burning_node(self, new_distance):
        self.distance_to_burning_node = new_distance

# Provides the methods to create and solve the firefighter problem
class FFP:

    # Constructor
    #   fileName = The name of the file that contains the FFP instance
    def __init__(self, fileName):
        file = open(fileName, "r")
        text = file.read()
        tokens = text.split()
        seed = int(tokens.pop(0))
        self.n = int(tokens.pop(0))
        self.nodes = []
        self.debug_position = list(range(self.n))
        model = int(tokens.pop(0))
        int(tokens.pop(0))  # Ignored
        # self.state contains the state of each node
        #    -1 On fire
        #     0 Available for analysis
        #     1 Protected
        self.state = [0] * self.n
        nbBurning = int(tokens.pop(0))
        for i in range(nbBurning):
            b = int(tokens.pop(0))
            self.state[b] = -1
        self.graph_m = []   # graph as an adjacency matrix
        for i in range(self.n):
            self.graph_m.append([0] * self.n)
        while tokens:
            x = int(tokens.pop(0))
            y = int(tokens.pop(0))
            self.graph_m[x][y] = 1
            self.graph_m[y][x] = 1
        self.graph_l = self.convert_adjacency_matrix_to_list()    # graph as an adjacency list
        # Create backbone to use the Heuristics
        self.backbone = []
        self.DFS(0)
        self.create_nodes()
        
        
    def create_nodes(self):
        for index in range(self.n):
            degree = sum(self.graph_m[index])
            node = Node(index)
            node.update_degree(degree)
            if self.state[index] != 0:
                node.update_state(self.state[index])
            node.update_adjacent_nodes(self.graph_l[index])
            self.nodes.append(node)
    
    def calculate_node_burning_distance(self, node):
        # Save burning nodes in a list
        burning_nodes = []
        for index in range(self.n):
            if self.state[index] == -1:
                burning_nodes.append(index)
        minimun_distance = self.n
        # Start checking paths to burning nodes
        for burning_node in burning_nodes:
            path = BFS_SP(self.graph_l, node.id, burning_node)
            if len(path) < minimun_distance:
                minimun_distance = len(path)
        return minimun_distance


    def update_nodes(self):
        for index in range(self.n):
            node = self.nodes[index]
            state = self.state[index]
            # Update the state of the node if there is an update in the state
            if node.state != state:
                node.state = state
                if node.state == 0:
                    # Check if node is next to burning node
                    for next_node in node.adjacent_nodes:
                        if next_node == -1:
                            node.next_to_burning_node = True
                    # Check distance to new burning nodes
                    min_dist = self.calculate_node_burning_distance(node)
                    node.update_distance_to_burning_node(min_dist)

    def convert_adjacency_matrix_to_list(self):
        graph_m = self.graph_m
        adjacency_l = defaultdict(list)
        for i in range(len(graph_m)):
            for j in range(len(graph_m[i])):
                if graph_m[i][j] != 0:
                    adjacency_l[i].append(j)
        return adjacency_l

    def print_adjacency_list(self):
        for i in self.graph_l:
            print(i, end ="")
            for j in self.graph_l[i]:
                print(" -> {}".format(j), end ="")
            print()
    
    '''A recursive function to print all paths from 'u' to 'd'.
    visited[] keeps track of vertices in current path.
    path[] stores actual vertices and path_index is current
    index in path[]'''
    def printAllPathsUtil(self, u, d, visited, path):
 
        # Mark the current node as visited and store in path
        visited[u]= True
        path.append(u)
 
        # If current vertex is same as destination, then print
        # current path[]
        if u == d:
            print (path)
        else:
            # If current vertex is not destination
            # Recur for all the vertices adjacent to this vertex
            for i in self.graph_l[u]:
                if visited[i]== False:
                    self.printAllPathsUtil(i, d, visited, path)
                     
        # Remove current vertex from path[] and mark it as unvisited
        path.pop()
        visited[u]= False

    # Prints all paths from 's' to 'd'
    def printAllPaths(self, s, d):
 
        # Mark all the vertices as not visited
        visited =[False]*(self.n)
 
        # Create an array to store paths
        path = []
 
        # Call the recursive helper function to print all paths
        self.printAllPathsUtil(s, d, visited, path)

    # A function used by DFS
    def DFSUtil(self, v, visited):
 
        # Mark the current node as visited
        # and print it
        visited.add(v)
        # print(v, end=' ')
        # print(" -> {}".format(v), end ="")
        # print(" -> {}".format(j), end ="")
        self.backbone.append(v)
 

        # Recur for all the vertices
        # adjacent to this vertex
        for neighbour in self.graph_l[v]:
            if neighbour not in visited:
                self.DFSUtil(neighbour, visited)
 
    # The function to do DFS traversal. It uses
    # recursive DFSUtil()
    def DFS(self, v):
 
        # Create a set to store visited vertices
        visited = set()
 
        # Call the recursive helper function
        # to print DFS traversal
        self.DFSUtil(v, visited)

    # Solves the FFP by using a given method and a number of firefighters
    #   method = Either a string with the name of one available heuristic or an object of class HyperHeuristic
    #   nbFighters = The number of available firefighters per turn
    #   debug = A flag to indicate if debugging messages are shown or not
    def solve(self, method, nbFighters, debug=False):
        spreading = True
        if (debug):
            print("Initial state:" + str(self.state))
        t = 0
        while (spreading):
            self.update_nodes()
            if (debug):
                print("Pos:  " + str(self.debug_position))
                print("State:" + str(self.state))
                print("Features")
                print("")
                print("Graph density: %1.4f" %
                      (self.getFeature("EDGE_DENSITY")))
                print("Average degree: %1.4f" %
                      (self.getFeature("AVG_DEGREE")))
                print("Burning nodes: %1.4f" %
                      self.getFeature("BURNING_NODES"))
                print("Burning edges: %1.4f" %
                      self.getFeature("BURNING_EDGES"))
                print("Nodes in danger: %1.4f" %
                      self.getFeature("NODES_IN_DANGER"))
            # It protects the nodes (based on the number of available firefighters)
            for i in range(nbFighters):
                heuristic = method
                if (isinstance(method, HyperHeuristic)):
                    heuristic = method.nextHeuristic(self)
                node = self.__nextNode(heuristic)
                if (node >= 0):
                    # The node is protected
                    self.state[node] = 1
                    # The node is disconnected from the rest of the graph
                    for j in range(len(self.graph_m[node])):
                        self.graph_m[node][j] = 0
                        self.graph_m[j][node] = 0
                    if (debug):
                        print("\tt" + str(t) +
                              ": A firefighter protects node " + str(node))
            # It spreads the fire among the unprotected nodes
            spreading = False
            state = self.state.copy()
            for i in range(len(state)):
                # If the node is on fire, the fire propagates among its neighbors
                if (state[i] == -1):
                    for j in range(len(self.graph_m[i])):
                        if (self.graph_m[i][j] == 1 and state[j] == 0):
                            spreading = True
                            # The neighbor is also on fire
                            self.state[j] = -1
                            # The edge between the nodes is removed (it will no longer be used)
                            self.graph_m[i][j] = 0
                            self.graph_m[j][i] = 0
                            if (debug):
                                print("\tt" + str(t) +
                                      ": Fire spreads to node " + str(j))
            t = t + 1
            if (debug):
                print("---------------")
        if (debug):
            print("Final state: " + str(self.state))
            print("Solution evaluation: " +
                  str(self.getFeature("BURNING_NODES")))
        return self.getFeature("BURNING_NODES")

    # Selects the next node to protect by a firefighter
    #   heuristic = A string with the name of one available heuristic
    def __nextNode(self, heuristic):
        index = -1
        best = -1
        for i in range(len(self.state)):
            if (self.state[i] == 0):
                index = i
                break
        value = -1
        for i in range(len(self.state)):
            if (self.state[i] == 0):
                if (heuristic == "LDEG"):
                    print("LDEG")
                    # It prefers the node with the largest degree, but it only considers
                    # the nodes directly connected to a node on fire
                    for j in range(len(self.graph_m[i])):
                        if (self.graph_m[i][j] == 1 and self.state[j] == -1):
                            value = sum(self.graph_m[i])
                            break
                elif (heuristic == "GDEG"):
                    value = sum(self.graph_m[i])
                elif (heuristic == "BBG"):
                    print("BBG\n")
                    if i in self.backbone:
                        value = sum(self.graph_m[i])
                else:
                    print("=====================")
                    print("Critical error at FFP.__nextNode.")
                    print("Heuristic " + heuristic +
                          " is not recognized by the system.")
                    print("The system will halt.")
                    print("=====================")
                    exit(0)
            if (value > best):
                best = value
                index = i
        return index

    # Returns the value of the feature provided as argument
    #   feature = A string with the name of one available feature
    def getFeature(self, feature):
        f = 0
        if (feature == "EDGE_DENSITY"):
            n = len(self.graph_m)
            for i in range(len(self.graph_m)):
                f = f + sum(self.graph_m[i])
            f = f / (n * (n - 1))
        elif (feature == "AVG_DEGREE"):
            n = len(self.graph_m)
            count = 0
            for i in range(len(self.state)):
                if (self.state[i] == 0):
                    f += sum(self.graph_m[i])
                    count += 1
            if (count > 0):
                f /= count
                f /= (n - 1)
            else:
                f = 0
        elif (feature == "BURNING_NODES"):
            for i in range(len(self.state)):
                if (self.state[i] == -1):
                    f += 1
            f = f / len(self.state)
        elif (feature == "BURNING_EDGES"):
            n = len(self.graph_m)
            for i in range(len(self.graph_m)):
                for j in range(len(self.graph_m[i])):
                    if (self.state[i] == -1 and self.graph_m[i][j] == 1):
                        f += 1
            f = f / (n * (n - 1))
        elif (feature == "NODES_IN_DANGER"):
            for j in range(len(self.state)):
                for i in range(len(self.state)):
                    if (self.state[i] == -1 and self.graph_m[i][j] == 1):
                        f += 1
                        break
            f /= len(self.state)
        else:
            print("=====================")
            print("Critical error at FFP._getFeature.")
            print("Feature " + feature + " is not recognized by the system.")
            print("The system will halt.")
            print("=====================")
            exit(0)
        return f

    # Returns the string representation of this problem
    def __str__(self):
        text = "n = " + str(self.n) + "\n"
        text += "state = " + str(self.state) + "\n"
        for i in range(self.n):
            for j in range(self.n):
                if (self.graph_m[i][j] == 1 and i < j):
                    text += "\t" + str(i) + " - " + str(j) + "\n"
        return text

# Provides the methods to create and use hyper-heuristics for the FFP
# This is a class you must extend it to provide the actual implementation


class HyperHeuristic:

    # Constructor
    #   features = A list with the names of the features to be used by this hyper-heuristic
    #   heuristics = A list with the names of the heuristics to be used by this hyper-heuristic
    def __init__(self, features, heuristics):
        if (features):
            self.features = features.copy()
        else:
            print("=====================")
            print("Critical error at HyperHeuristic.__init__.")
            print("The list of features cannot be empty.")
            print("The system will halt.")
            print("=====================")
            exit(0)
        if (heuristics):
            self.heuristics = heuristics.copy()
        else:
            print("=====================")
            print("Critical error at HyperHeuristic.__init__.")
            print("The list of heuristics cannot be empty.")
            print("The system will halt.")
            print("=====================")
            exit(0)

    # Returns the next heuristic to use
    #   problem = The FFP instance being solved
    def nextHeuristic(self, problem):
        print("=====================")
        print("Critical error at HyperHeuristic.nextHeuristic.")
        print("The method has not been overriden by a valid subclass.")
        print("The system will halt.")
        print("=====================")
        exit(0)

    # Returns the string representation of this hyper-heuristic
    def __str__(self):
        print("=====================")
        print("Critical error at HyperHeuristic.__str__.")
        print("The method has not been overriden by a valid subclass.")
        print("The system will halt.")
        print("=====================")
        exit(0)

# A dummy hyper-heuristic for testing purposes.
# The hyper-heuristic creates a set of randomly initialized rules.
# Then, when called, it measures the distance between the current state and the
# conditions in the rules
# The rule with the condition closest to the problem state is the one that fires


class DummyHyperHeuristic(HyperHeuristic):

    # Constructor
    #   features = A list with the names of the features to be used by this hyper-heuristic
    #   heuristics = A list with the names of the heuristics to be used by this hyper-heuristic
    #   nbRules = The number of rules to be contained in this hyper-heuristic
    def __init__(self, features, heuristics, nbRules, seed):
        super().__init__(features, heuristics)
        random.seed(seed)
        self.conditions = []
        self.actions = []
        for i in range(nbRules):
            self.conditions.append([0] * len(features))
            for j in range(len(features)):
                self.conditions[i][j] = random.random()
            self.actions.append(
                heuristics[random.randint(0, len(heuristics) - 1)])

    # Returns the next heuristic to use
    #   problem = The FFP instance being solved
    def nextHeuristic(self, problem):
        minDistance = float("inf")
        index = -1
        state = []
        for i in range(len(self.features)):
            state.append(problem.getFeature(self.features[i]))
        print("\t" + str(state))
        for i in range(len(self.conditions)):
            distance = self.__distance(self.conditions[i], state)
            if (distance < minDistance):
                minDistance = distance
                index = i
        heuristic = self.actions[index]
        print("\t\t=> " + str(heuristic) + " (R" + str(index) + ")")
        return heuristic

    # Returns the string representation of this dummy hyper-heuristic
    def __str__(self):
        text = "Features:\n\t" + \
            str(self.features) + "\nHeuristics:\n\t" + \
            str(self.heuristics) + "\nRules:\n"
        for i in range(len(self.conditions)):
            text += "\t" + \
                str(self.conditions[i]) + " => " + self.actions[i] + "\n"
        return text

    # Returns the Euclidian distance between two vectors
    def __distance(self, vectorA, vectorB):
        distance = 0
        for i in range(len(vectorA)):
            distance += (vectorA[i] - vectorB[i]) ** 2
        distance = math.sqrt(distance)
        return distance

# Tests
# =====================


fileName = "instances/BBGRL/50_ep0.2_0_gilbert_1.in"
# Solves the problem using heuristic LDEG and one firefighter
# problem = FFP(fileName)
# print("LDEG = " + str(problem.solve("LDEG", 1, True)))

# Solves the problem using heuristic GDEG and one firefighter
problem = FFP(fileName)
# problem.print_adjacency_list()

print(problem.backbone)

print("BBG = " + str(problem.solve("BBG", 1, True)))


# print("GDEG = " + str(problem.solve("GDEG", 1, True)))

# # Solves the problem using a randomly generated dummy hyper-heuristic
# problem = FFP(fileName)
# seed = random.randint(0, 1000)
# print(seed)
# hh = DummyHyperHeuristic(["EDGE_DENSITY", "BURNING_NODES", "NODES_IN_DANGER"], [
#                          "LDEG", "GDEG"], 2, seed)
# print(hh)
# print("Dummy HH = " + str(problem.solve(hh, 1, True)))
