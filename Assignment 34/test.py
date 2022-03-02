#from games import *
import utils


import copy
import itertools
import random
from collections import namedtuple

import numpy as np

from utils import vector_add
"""
succs = dict()
  
# key with 2 values
# first value is for min second is for max
# for a given choice
utils = dict(one=[-1,1], two=[-2,2], three=[-3,3], four=[-4,4], five=[-5,5], six=[-6,6], seven=[-7,7], eight=[-8,8], nine=[-9,9])
# utils will not be needed right now

# set equivalents for later
equival = dict(one='1', two='2', three='3', four='4', five='5', six='6', seven='7', eight='8', nine='9')

# Need to have multiple subsets from sigma
# This will help me visualize the problem
# and the actions being taken

# initial / sigma
sigma = {'1','2','3','4','5','6','7','8','9'}

# previous >= 5
bo = {'1','3','5','7','9'}
be = {'2','4','6','8'}

# previous < 5
so = {'5','7','9'}
se = {'6','8'}

# < 5
sf = {'1', '2', '3', '4'}

# >= 5
gf = {'5', '6', '7', '8', '9'}


# terminal state
term = set()
"""
class Game:
    """A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement actions,
    result, utility, and terminal_test. You may override display and
    successors or you can inherit their default methods. You will also
    need to set the .initial attribute to the initial state; this can
    be done in the constructor."""

    def actions(self, state):
        """Return a list of the allowable moves at this point."""
        raise NotImplementedError

    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        raise NotImplementedError

    def utility(self, state, player):
        """Return the value of this final state to player."""
        raise NotImplementedError

    def terminal_test(self, state):
        """Return True if this is a final state for the game."""
        return not self.actions(state)

    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    def display(self, state):
        """Print or otherwise display the state."""
        print(state)

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    def play_game(self, *players):
        """Play an n-person, move-alternating game."""
        state = self.initial
        while True:
            for player in players:
                move = player(self, state)
                state = self.result(state, move)
                if self.terminal_test(state):
                    self.display(state)
                    return self.utility(state, self.to_move(self.initial))


class agame(Game):
	succs = {}
	# key with 2 values
	# first value is for min second is for max
	# for a given choice
	utils = (1, 2, 3, 4, 5, 6, 7, 8, 9)
	# initial / sigma
	sigma = {1,2,3,4,5,6,7,8,9}
	# previous >= 5
	bo = {1,3,5,7,9}
	be = {2,4,6,8}
	# previous < 5
	so = {5,7,9}
	se = {6,8}
	# < 5
	sf = {1, 2, 3, 4}
	# >= 5
	gf = {5, 6, 7, 8, 9}
	# terminal state
	term = []
	# keep track of visited values
	visited = set()
	counter = 0
	def visit(state):
		self.visited.add(state)
		return
	
	# give result or state after taking a path
	def result(self, e):
		return self.succs[e]
	
	# give actions that can be taken from current state
	def actions(self, state):
		self.succs.clear()
		if isinstance(state, int):
			self.succs[state] = self.check(state)
		else:
			for i in state:
				self.succs[i] = self.check(i)
			
		return list(self.succs.keys())
	# checks contraints
	def check(self, e):
		print('element: ', e)
		print('visited: ', list(self.visited))
		# even
		if e % 2 == 0 and e < 5:
			# because its even so next must be odd and less than 5 so next >= 5
			return list((self.so.difference(self.visited)))
			# checkMin(so, store)
		elif e % 2 == 0 and e >= 5:
			# because its even so next must be odd
			return list((self.bo.difference(self.visited)))
		# odd
		elif e % 2 != 0 and e < 5:
			# becuase its odd so next must be even and less than 5 so next >= 5
			return list((self.se.difference(self.visited)))
		elif e % 2 != 0 and e >= 5:
			# because its even so next must be even
			return list((self.be.difference(self.visited)))
		
	# returns available actions given previous moves and rules
	def utility(self, state, player):
		#tmp = int(state)
		if player == 'MAX':
			return self.utils[equival[state - 1]]
		else:
			return -self.utils[equival[state - 1]]
	
	# check if state is empty, indicating a terminal state
	def terminal_test(self, state):
		return True if len(state) == 0 else False
  
	def to_move(self, state):
		return 'MAX' if state in self.sigma else 'MIN'


# end agame(Game)

def minmax_decision(state, game):

	player = game.to_move(state)
	# visited = set()
	#game.visit(state)
	
	def max_value(state):
		if game.terminal_test(state):
			return game.utility(state, player)
		v = -np.inf
		for a in game.actions(state):
			v = max(v, min_value(game.actions(state)))
			print('v max: ', list(v))
		#game.visit(v)
		print('visitedmax: ', list(game.visited))
		return v
	def min_value(state):
		if game.terminal_test(state):
			return game.utility(state, player)
		v = np.inf
		tmp = game.actions(state)
		for a in tmp:
			v = min(v, max_value(game.actions(state)))
			print('v min:', list(v))
		#game.visit(v)
		print('visitedmin: ', list(game.visited))
		return v
	# body of minmax_decision:
	return max(game.actions(state), key=lambda a: min_value(game.actions(a)))
		