# James Cooper
# CS475
# 2/20/2022
# This is incomplete.
# I have no idea what I am doing.
# I am sure none of it is correct.
# Pretty sure IDS is completely wrong.
# As well as DFS.
# :(


"""
Search (Chapters 3-4)

The way to use this code is to subclass Problem to create a class of problems,
then create problem instances and solve them with calls to the various search
functions.
"""

import sys
from collections import deque

from utils import *


class Problem:
    """The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initialx, initialy, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        self.initialx = initialx
        self.initialy = initialy
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, statex, statey):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if statex in statey:
          return True
        else:
          return False

    # this is horrible
    def goal_test_arr(self, arrx, arry):
        for m in range(0, len(arrx)):
            frontLenx = len(arrx[m])
            arrElementx = arrx[m]
            for n in range(0, len(arry)):
                frontLeny= len(arry[n])
                arrElementy = arry[n]
                if arrx[n] == arry[m]:
                    return True
                elif m == n and arrELementx == arrElementy:
                    return True
        return False
    
    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value. Hill Climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError

# ______________________________________________________________________________

class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state. Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node. Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        # We use the hash value of the state
        # stored in the node instead of the node
        # object itself to quickly search a node
        # with the same state in a Hash Table
        return hash(self.state)

# ______________________________________________________________________________
# Uninformed Search algorithms

def depth_first_graph_search(problem, choice):
    """
    [Figure 3.7]
    Search the deepest nodes in the search tree first.
    Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    Does not get trapped by loops.
    If two paths reach a state, only use the first one.
    """
    frontierx = [(Node(problem.initialx))]  # Stack for x
    frontiery = [(Node(problem.initialy))]  # Stack for y
    exploredx = set()
    exploredy = set()
    temp = set()
    iter = 0
    while frontierx and frontiery:
        nodex = frontierx.pop()
        nodey = frontiery.pop()
        print("Iteration: ", iter, "selected node with ( x ) state", nodex.state)
        print("Iteration: ", iter, "selected node with ( y ) state", nodey.state)
        exploredx.add(nodex.state)
        exploredy.add(nodey.state)
        frontierx.extend(child for child in nodex.expand(problem)
                        if child.state not in exploredx and child not in frontierx)
        if (bool(frontierx)):
            temp = frontierx.pop()
            print("    Added", temp, "to frontier ( x )")
            frontierx.append(temp)

        frontiery.extend(child for child in nodey.expand(problem)
                        if child.state not in exploredy and child not in frontiery)
        if (bool(frontiery)):
            temp = frontiery.pop()
            print("    Added", temp, "to frontier ( y )")
            frontiery.append(temp)
        if nodex.state == nodey.state:
            if choice == 0:
                return nodex
            elif choice == 1:
                return nodey
        iter = iter + 1

    print("\n--No solution found--")
    print("--All possible nodes visisted--")
    print("--Expect an AttributeError--\n\n")
    return None


def breadth_first_graph_search(problem, choice):
    """[Figure 3.11]
    Note that this function can be implemented in a
    single line as below:
    return graph_search(problem, FIFOQueue())
    """
    nodex = Node(problem.initialx)
    nodey = Node(problem.initialy)
    if problem.goal_test(nodex.state, nodey.state):
        return nodex
    frontierx = deque([nodex])
    frontiery = deque([nodey])
    exploredx = set()
    exploredy = set()
    temp = set()
    iter = 0
    while frontierx:
        
        temp.clear()
        nodex = frontierx.popleft()
        nodey = frontiery.popleft()
        print("Iteration: ", iter, "selected node with ( x ) state", nodex.state)
        print("Iteration: ", iter, "selected node with ( y ) state", nodey.state)
        exploredx.add(nodex.state)
        exploredy.add(nodey.state)
        for childx in nodex.expand(problem):
            if childx.state not in exploredx and childx not in frontierx:
                print("    child x node with state -- if goal return", childx.state)
                frontierx.append(childx)
                print("    Added", childx, "to frontier ( x )")
                temp.add(childx)
        for childy in nodey.expand(problem):
            if childy.state not in exploredy and childy not in frontiery:
                print("    child y node with state -- if goal return", childy.state)
                frontiery.append(childy)
                print("    Added", childy, "to frontier ( y )")
            if problem.goal_test(childy, temp):
                for childx in temp:
                    if childy == childx:
                        if choice == 0:
                            return childx
                        elif choice == 1:
                            return childy
            
        iter = iter + 1
    return None

def best_first_graph_search(choice, problem, f, display=False):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    nodex = Node(problem.initialx)
    nodey = Node(problem.initialy)
    frontierx = PriorityQueue('min', f)
    frontiery = PriorityQueue('min' ,f)
    frontierx.append(nodex)
    frontiery.append(nodey)
    exploredx = set()
    exploredy = set()
    while frontierx:
        nodex = frontierx.pop()
        nodey = frontiery.pop()
        if goal_test_arr(frontierx, frontiery):
            if display:
                print(len(exploredx), "( x ) paths have been expanded and", len(frontierx), "( x ) paths remain in the frontier")
                print(len(exploredy), "( y ) paths have been expanded and", len(frontiery), "( y ) paths remain in the frontier")
            return choiceHelper(nodex, nodey, choice)
        exploredx.add(nodex.state)
        exploredy.add(nodey.state)
        for child in nodex.expand(problem):
            if child.state not in exploredx and child not in frontierx:
                frontierx.append(child)
            elif child in frontierx:
                if f(child) < frontierx[child]:
                    del frontierx[child]
                    frontierx.append(child)
        for child in nodey.expand(problem):
            if child.state not in exploredy and child not in frontiery:
                frontiery.append(child)
            elif child in frontiery:
                if f(child) < frontiery[child]:
                    del frontiery[child]
                    frontiery.append(child)
        
    return None

def uniform_cost_search(choice, problem, display=False):
    """[Figure 3.14]"""
    return best_first_graph_search(choice, problem, lambda node: node.path_cost, display)

def choiceHelper(x, y, choice):
  if choice == 0:
    return x
  elif choice == 1:
    return y

def depth_limited_search(problem, choice, limit=50):
    """[Figure 3.17]"""
    def recursive_dls(nodex, nodey, problem, choice, limit):
        #if problem.goal_test(node.state):
        print()
        print("Iteration on node ( x )", nodex.state)
        print("Iteration on node ( y )", nodey.state)
        if nodex.state == nodey.state:
            return choiceHelper(nodex, nodey, choice)
        elif limit == 0:
            return 'cutoff'
        else:
            cutoff_occurred = False
            for childx in nodex.expand(problem):
                for childy in nodey.expand(problem):
                    result = recursive_dls(childx, childy, problem, choice, limit - 1)
                    if result == 'cutoff':
                        cutoff_occurred = True
                    elif result is not None:
                        return result
            return 'cutoff' if cutoff_occurred else None

    # Body of depth_limited_search:
    return recursive_dls(Node(problem.initialx), Node(problem.initialy), problem, choice, limit)


def iterative_deepening_search(problem, choice):
    """[Figure 3.18]"""
    for depth in range(sys.maxsize):
        result = depth_limited_search(problem, choice, depth)
        if result != 'cutoff':
            return result


# ______________________________________________________________________________
# Informed (Heuristic) Search

greedy_best_first_graph_search = best_first_graph_search

# Greedy best-first search is accomplished by specifying f(n) = h(n).

def astar_search(choice, problem, h=None, display=False):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(choice, problem, lambda n: n.path_cost + h(n), display)

# _____________________________________________________________________________
# The remainder of this file implements examples for the search algorithms.

# ______________________________________________________________________________
# Graphs and Graph Problems

class Graph:
    """A graph connects nodes (vertices) by edges (links). Each edge can also
    have a length associated with it. The constructor call is something like:
        g = Graph({'A': {'B': 1, 'C': 2})
    this makes a graph with 3 nodes, A, B, and C, with an edge of length 1 from
    A to B,  and an edge of length 2 from A to C. You can also do:
        g = Graph({'A': {'B': 1, 'C': 2}, directed=False)
    This makes an undirected graph, so inverse links are also added. The graph
    stays undirected; if you add more links with g.connect('B', 'C', 3), then
    inverse link is also added. You can use g.nodes() to get a list of nodes,
    g.get('A') to get a dict of links out of A, and g.get('A', 'B') to get the
    length of the link from A to B. 'Lengths' can actually be any object at
    all, and nodes can be any hashable object."""

    def __init__(self, graph_dict=None, directed=True):
        self.graph_dict = graph_dict or {}
        self.directed = directed
        if not directed:
            self.make_undirected()

    def make_undirected(self):
        """Make a digraph into an undirected graph by adding symmetric edges."""
        for a in list(self.graph_dict.keys()):
            for (b, dist) in self.graph_dict[a].items():
                self.connect1(b, a, dist)

    def connect(self, A, B, distance=1):
        """Add a link from A and B of given distance, and also add the inverse
        link if the graph is undirected."""
        self.connect1(A, B, distance)
        if not self.directed:
            self.connect1(B, A, distance)

    def connect1(self, A, B, distance):
        """Add a link from A to B of given distance, in one direction only."""
        self.graph_dict.setdefault(A, {})[B] = distance

    def get(self, a, b=None):
        """Return a link distance or a dict of {node: distance} entries.
        .get(a,b) returns the distance or None;
        .get(a) returns a dict of {node: distance} entries, possibly {}."""
        links = self.graph_dict.setdefault(a, {})
        if b is None:
            return links
        else:
            return links.get(b)

    def nodes(self):
        """Return a list of nodes in the graph."""
        s1 = set([k for k in self.graph_dict.keys()])
        s2 = set([k2 for v in self.graph_dict.values() for k2, v2 in v.items()])
        nodes = s1.union(s2)
        return list(nodes)

def UndirectedGraph(graph_dict=None):
    """Build a Graph where every edge (including future ones) goes both ways."""
    return Graph(graph_dict=graph_dict, directed=False)

# Map of the graph in the problem with points A - H
# where A = locA, B = locB, etc,.
apalleco_map = UndirectedGraph(dict(
  locA = dict(locB = 5, locC = 6, locG = 8),
  locB = dict(locA = 5, locE = 4),
  locC = dict(locA = 6, locD = 3, locE = 5),
  locD = dict(locB = 6, locC = 3, locF = 5),
  locE = dict(locB = 4, locC = 5, locG = 3),
  locF = dict(locD = 5, locG = 2, locH = 1),
  locG = dict(locA = 8, locE = 3, locF = 2),
  locH = dict(locF = 1)
))

# Holds the straight line distance from all nodes to each other.
# Distance from the node to itself is 0
apalleco_map.locations = dict(
    locA = dict(locA = 0, locB = 5, locC = 6, locD = 7, locE = 6, locF = 9, locG = 8, locH = 10),
    locB = dict(locA = 5, locB = 0, locC = 5, locD = 6, locE = 4, locF = 6, locG = 6, locH = 8),
    locC = dict(locA = 6, locB = 5, locC = 0, locD = 3, locE = 5, locF = 7, locG = 6, locH = 7),
    locD = dict(locA = 7, locB = 6, locC = 3, locD = 0, locE = 5, locF = 5, locG = 6, locH = 5),
    locE = dict(locA = 6, locB = 4, locC = 5, locD = 5, locE = 0, locF = 4, locG = 3, locH = 5),
    locF = dict(locA = 9, locB = 6, locC = 7, locD = 5, locE = 4, locF = 0, locG = 2, locH = 1),
    locG = dict(locA = 8, locB = 6, locC = 6, locD = 6, locE = 3, locF = 2, locG = 0, locH = 2),
    locH = dict(locA = 10, locB = 8, locC =7 , locD = 5, locE = 5, locF = 1, locG = 2, locH = 0)
    )

""" [Figure 3.2]
Simplified road map of Romania
"""
romania_map = UndirectedGraph(dict(
    Arad=dict(Zerind=75, Sibiu=140, Timisoara=118),
    Bucharest=dict(Urziceni=85, Pitesti=101, Giurgiu=90, Fagaras=211),
    Craiova=dict(Drobeta=120, Rimnicu=146, Pitesti=138),
    Drobeta=dict(Mehadia=75),
    Eforie=dict(Hirsova=86),
    Fagaras=dict(Sibiu=99),
    Hirsova=dict(Urziceni=98),
    Iasi=dict(Vaslui=92, Neamt=87),
    Lugoj=dict(Timisoara=111, Mehadia=70),
    Oradea=dict(Zerind=71, Sibiu=151),
    Pitesti=dict(Rimnicu=97),
    Rimnicu=dict(Sibiu=80),
    Urziceni=dict(Vaslui=142)))

romania_map.locations = dict(
    Arad=(91, 492), Bucharest=(400, 327), Craiova=(253, 288),
    Drobeta=(165, 299), Eforie=(562, 293), Fagaras=(305, 449),
    Giurgiu=(375, 270), Hirsova=(534, 350), Iasi=(473, 506),
    Lugoj=(165, 379), Mehadia=(168, 339), Neamt=(406, 537),
    Oradea=(131, 571), Pitesti=(320, 368), Rimnicu=(233, 410),
    Sibiu=(207, 457), Timisoara=(94, 410), Urziceni=(456, 350),
    Vaslui=(509, 444), Zerind=(108, 531))



class GraphProblem(Problem):
    """The problem of searching a graph from one node to another."""
    
#    def __init__(self, initial, goal, graph):
    def __init__(self, initialx, initialy, goal, graph):
        super().__init__(initialx, initialy, goal)
        self.graph = graph

    def actions(self, A):
        """The actions at a graph node are just its neighbors."""
        return list(self.graph.get(A).keys())

    def result(self, state, action):
        """The result of going to a neighbor is just that neighbor."""
        return action

    def path_cost(self, cost_so_far, A, action, B):
        return cost_so_far + (self.graph.get(A, B) or np.inf)

    def find_min_edge(self):
        """Find minimum value of edges."""
        m = np.inf
        for d in self.graph.graph_dict.values():
            local_min = min(d.values())
            m = min(m, local_min)

        return m

    def h(self, node):
        """h function is straight-line distance from a node's state to goal."""
        locs = getattr(self.graph, 'locations', None)
        if locs:
            if type(node) is str:
                
                return int(distance(locs[node], locs[self.goal]))

            return int(distance(locs[node.state], locs[self.goal]))
        else:
            return np.inf
    
    def h_1(self, node): 
        locs = getattr(self.graph, 'locations', None)
        min = 0
        if locs:
            if type(node) is str:
                # chooses lowest straight line cost to the other nodes and returns it
                # not sure if this is how it's done, I couldn't complete astar
                if node.state == 'locA':
                    A = locs[locA]
                    for key, value in A.items():
                        if value < min and value > 0:
                            min = value
                elif node.state == 'locB':
                    B = locs[locB]
                    for key, value in B.items():
                        if value < min and value > 0:
                            min = value
                elif node.state == 'locC':
                    C = locs[locC]
                    for key, value in C.items():
                        if value < min and value > 0:
                            min = value
                elif node.state == 'locD':
                    D = locs[locD]
                    for key, value in D.items():
                        if value < min and value > 0:
                            min = value
                elif node.state == 'locE':
                    E = locs[locE]
                    for key, value in E.items():
                        if value < min and value > 0:
                            min = value
                elif node.state == 'locF':
                    F = locs[locF]
                    for key, value in F.items():
                        if value < min and value > 0:
                            min = value
                elif node.state == 'locG':
                    G = locs[locG]
                    for key, value in G.items():
                        if value < min and value > 0:
                            min = value
                elif node.state == 'locH':
                    H = locs[locH]
                    for key, value in H.items():
                        if value < min and value > 0:
                            min = value
              
        return min
        