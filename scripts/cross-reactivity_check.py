#This code attempts to do a proof of principle calucation that our idea works in high dimensions
#Import standard python packages:
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#import ploting packages
import os

os.environ["PATH"] += ':/usr/local/texlive/2015/bin/x86_64-darwin'
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.tick_params(labelsize=16)
plt.clf()

#declare simulation paramaters:
#basel proliferation rate
rho = 1.0

#death rate
b = 1.0

#the number of Tregs
Num_treg = 10

#the number of Tcells
Num_tcell = 5000

#the number of sites
Num_sites = 9000

#the Treg- antigen site binding strenth
c = 1.0

 #generate a Treg-Tcell graph
#######################################

#the vx densities
max_v = 1.0
vx = max_v * np.ones(Num_sites) + np.random.uniform(-max_v/2,max_v/2,Num_sites) 


#shuffle the array around
labeling1A = np.arange(1  , Num_tcell)
labeling2A = np.arange(1 , Num_sites)
from sklearn.utils import shuffle
#labeling1A = shuffle( labeling1A , random_state=True)
#labeling2A = shuffle( labeling2A , random_state=True)

#defining the cross reactivity funtions
scale1 =  float(Num_sites) / float(Num_tcell) 
def tcell_Block( x , y , scale):


	xp =  labeling1A[ x - 1 ]
	yp =  labeling2A[y - 1]


	A = ( (xp - yp/scale1 ) % Num_tcell)
	B = ( ( yp/scale1  - xp)% Num_tcell)

	#A = (x - y/scale1 )% Num_tcell
	#B = ( y/scale1  - x)% Num_tcell

	dist = min(A,B)

	val = 0

	if ( dist  < scale ):

		val = 1

	#val = np.exp( - np.abs( dist  )**2 / scale    )

	return val



#shuffle the array around
labeling1 = np.arange(1  , Num_treg)
labeling2 = np.arange(1 , Num_sites)
from sklearn.utils import shuffle
labeling1 = shuffle( labeling1, random_state=True)
labeling2 = shuffle( labeling2, random_state=True)

scale2 =  float(Num_sites) /float(Num_treg) 
def treg_Block(x,y , scale):

	xp =  labeling1[ x - 1 ]
	yp =  labeling2[y - 1]

	print( x , y)
	print( xp , yp )

	A = ( (xp - yp/scale2 ) % Num_treg)
	B = ( ( yp/scale2  - xp)% Num_treg)


	dist = min(A,B)

	val = 0

	if (dist < scale ):

		val = 1

	#val = np.exp( - np.abs( dist )**2 / scale    )

	return val


#layer 1 is the conectivity of the Tcells and antigen sequences
layer1 = np.zeros([Num_tcell,Num_sites])
 
#let us try to put in some 'cross reactivy' information:
for i in range(Num_tcell):

	for j in range(Num_sites):

		val = tcell_Block(i,j , 20)

		layer1[i,j] = val



#layer 2 is the connectivvity of the binding of the antigen sites to the number of Tregs
layer2 = np.zeros([Num_sites,Num_treg])

for i in range(Num_sites):

	for j in range(Num_treg):

		val = treg_Block(j, i, 1.3 )

		layer2[i,j] = val


plt.imshow(layer1 , aspect = 'auto')
plt.show()

plt.imshow(layer2 , aspect = 'auto')
plt.show()


#compute the mean binding, this will be used when we look at how well our aproxomation works
mean_reg = np.zeros([Num_treg])

for i in range(Num_treg):

	mean_reg[i] = np.sum( layer1[i,:] * layer2[:,i]     )

#now compute the r_{i} and \phi quantities
######################################################################

#compute the matrix overlaps
phi_reg_reg = np.zeros([Num_treg,Num_treg])

for i in range(Num_treg):
	for j in range(Num_treg):

		phi_reg_reg[i,j] = np.sum( vx[:]*layer2[:,i]*layer2[:,j]  )


plt.imshow(phi_reg_reg, aspect = 'auto')
plt.show()

#compute the matrix overlaps
phi_cell_reg = np.zeros([Num_tcell,Num_treg])


for i in range(Num_tcell):
	for j in range(Num_treg):

		phi_cell_reg[i,j] = np.sum( vx[:]*layer1[i,:]*layer2[:,j]  )


plt.imshow(phi_cell_reg, aspect = 'auto')
plt.show()


rvals = np.zeros([Num_tcell])

for i in range(Num_tcell):

	rvals[i] = np.sum( vx[:]*layer1[i,:]    )


plt.plot(rvals)
plt.show()



####################################################################################
#compute the Treg steady state and active set of constraints
#QP is done with CVXOPT packages
from cvxopt import matrix, solvers
import numpy as np
solvers.options['show_progress'] = False 

#Set up the quadratic part of QP matrix
Qmat = np.zeros([ Num_treg , Num_treg ])

for i in range(Num_treg):
	for j in range(Num_treg):
		
		Qmat[i,j] = phi_reg_reg[i,j]


#Convert to CVXOPT format
Q = matrix(Qmat)

p = np.zeros(Num_treg)
p = matrix(p)


G = np.zeros([Num_tcell + Num_treg , Num_treg ])

for i in range(Num_tcell):

	for j in range(Num_treg):

		G[i,j] = -1.0* phi_cell_reg[i,j] * (rvals[i]**(-1.0) ) 

#enforce positivity
for i in range(Num_tcell, Num_tcell + Num_treg):

	G[i, i - Num_tcell - Num_treg ] = -1.0


G = matrix(G)

h = np.zeros([Num_tcell + Num_treg])
for i in range(Num_tcell):
	h[i] = -1.0 * (rho) 

for i in range(Num_tcell,   Num_treg):

	h[i] = 0.0


h = matrix(h)

sol =  solvers.qp(Q, p, G, h)
#the QP solution, this is Treg part
vals = np.array( sol['x'] )


#the dual varibles
dual = np.array( sol['z'] )
dual = dual[0:Num_tcell]

Treg = vals[:,0]

#compute the distance to the specalist point
distance = np.sqrt( np.sum( (Treg - (rho/c)*np.ones(Num_treg) )**2        ) )
print('distance')
print(distance)



#now plot what the constrains look like
##########################################################################################################################################

import matplotlib.pyplot as plt
#import ploting packages
import os

os.environ["PATH"] += ':/usr/local/texlive/2015/bin/x86_64-darwin'
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.tick_params(labelsize=16)
plt.clf()

fontsize1 = 22



graph_space = np.linspace(0, 5*rho/c, 200)

for i in range(Num_tcell):

	if( phi_cell_reg[i,0] != 0 ):


		plt.plot(  graph_space,  (rvals[i]*rho -  phi_cell_reg[i,1] * graph_space)/(phi_cell_reg[i,0]), 'b' )


	if( phi_cell_reg[i,0] == 0 ):
		
		plt.axvline(x=rho/c , color = 'b')



plt.grid()
plt.plot(Treg[1],Treg[0],'*',markersize = 25,markeredgecolor='k', markerfacecolor='g' )
plt.ylim(0,5)
plt.xlim(0,5)
plt.tight_layout()
plt.show()
plt.clf()
