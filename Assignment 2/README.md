Made by James Cooper
CS 475

: Please read :

Requirements:

    - search.py
    - utils.py
    - sys
    - collections

Syntax:

    - The points A - H are labeled locA - locH in code.
    - If a search algorithm prints out "--No solution found.--" by default with .solution()
        > Running .solution() on it will yeild an AttributeError
	- Running .solution() will show the path the agent took to get to a certian node.
		> If .solution() was not added, by default it will print out the end node
		> based on the choice selected, either (x = 0, y = 1)

Variables:

    - apalleco_map
        > UndirectedGraph of problem with locations A - H
    - initialx
        > Starting position for x
        > Available Positions:
            {('locA'),('locB'),('locC'),('locD'),('locE'),('locF'),('locG'),('locH')}
    - initialx
        > Starting position for y
        > Available Positions:
            {('locA'),('locB'),('locC'),('locD'),('locE'),('locF'),('locG'),('locH')}
    - goal
        > None
    - graph
        > apalleco_map
    - problem
        > name of GraphProblem instance
	- choice
		> Pick either X or Y to expand upon.
		> No default value.
		> 0 is x
		> 1 is y
    
Usage:
    
	- Run: python3 or python in the terminal / command prompt
	- Run: from search import *
    - Create a GraphProblem instance using GraphProblem()
        > Syntax: GraphProblem(initialx, initialy, goal, graph)
			+ Ex: path = GraphProblem('locA', 'locH', None, apalleco_map)
        
    - Call one of the search algorithms and pass in the GraphProblem instance
        > Available searches:
            breadth_first_graph_search(problem, choice)
				+ Ex: breadth_first_graph_search(path, 0)
				+ Ex: breadth_first_graph_search(path, 1).solution()
			depth_first_graph_search(problem, choice)
				+ Ex: depth_first_graph_search(path, 1)
			iterative_deepening_search(problem, choice)
				+ Ex: iterative_deepening_search(path, 0)
				+ Ex: iterative_deepening_search(path, 1).solution()
			astar_search(choice, problem, problem.h_1)
				+ No example as it does not work.

Notes:
	
	- I am pretty sure DFS and IDS and Astar do not work...
	- These can all be ran with romania_map as well
	- The goal state is when each node.state is the same
	- Possible states are the location of each x and y and the node they are currently on