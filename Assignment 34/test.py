from games import *

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

class agame(Game):
	succs = dict()
	# key with 2 values
	# first value is for min second is for max
	# for a given choice
	utils = dict(one=1, two=2, three=3, four=4, five=5, six=6, seven=7, eight=8, nine=9)
	# equivalences
	equival = ['one', 'two', 'three', 'four', 'five', 'six', 'seve', 'eight', 'nine']
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
	# keep track of visited values
	visited = set()
	
	# returns available actions given previous moves and rules
	def actions(self, state, visited):
		istate = int(state)
		# even
		if istate % 2 == 0 and istate < 5:
			# because its even so next must be odd and less than 5 so next >= 5
			return list((so.difference(visited)).sort())
			# checkMin(so, store)
		elif istate % 2 == 0 and istate >= 5:
			# because its even so next must be odd
			return list((bo.difference(visited)).sort())
		# odd
		elif istate % 2 != 0 and istate < 5:
			# becuase its odd so next must be even and less than 5 so next >= 5
			return list((se.difference(visited)).sort())
		elif istate % 2 != 0 and istate >= 5:
			# because its even so next must be even
			return list((be.difference(visited)).sort())
		
	def utility(self, state, player):
		tmp = int(state)
		if player == 'MAX':
			return self.utils[equival[istate]]
		else:
			return -self.utils[equival[istate]]
	
	def terminal_test(self, state):
		return True if state in term else False


	
	
	
		