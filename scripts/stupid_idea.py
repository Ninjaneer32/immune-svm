
#this code attempts

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


#number of ammino acids
amminos = 22

#the number of nmers
nmers = 9

#the 'ammino acid' binding matrix
M = np.random.uniform( -10 , 10  ,size = (amminos,amminos) )

M = 0.5*( M + np.transpose(M)  )

#number of thymocites
th_Samples = 1000

thymocytes = np.zeros([th_Samples,nmers])

for i in range(th_Samples):

	thymocytes[i,:] = np.random.randint(0,nmers , size = nmers)


#most naive thing:
#probility ammino acid sequence is fixed

self_examples = 100

probs = np.random.uniform(0 , 1 , amminos )
probs = probs / (np.sum(probs))


self_sites = np.zeros( [self_examples , nmers] )

ammino_array = np.arange(amminos)

for i in range(self_examples):

	self_sites[i,:] = np.random.choice( ammino_array , size = (nmers), replace = True ,  p = probs  ) 



Beta = 1000.0
E = 21
bindings = np.zeros([ th_Samples , self_examples ])
#computing the binding energies of self and forign
for i in range(th_Samples):

	for j in range(self_examples):

		val = 0

		for k in range(nmers):

	
			val = val + M[ int(thymocytes[i,k]) , int(self_sites[j,k])  ]

		bindings[i , j] = 1/( 1 + np.exp( - Beta*( val - E) )   )



#Positive and negative selection?
Ep = + 0.7
En = + 1.0
delta = 0.2

Tregs = np.array([])
Tcells = np.array([] )

for i in range(th_Samples):


	if (  Ep <= np.max(bindings[i,:]) <= En   ):

		p = np.random.uniform(0 , 1 )

		if ( p < delta ):
			print(1)

			Tregs = np.append( Tregs , [i] ,  axis=None)

		if ( p > delta ):

			Tcells = np.append(  Tcells , [i] ,axis = None )


#fuck this does not work: Let us try a Euclidan shape space approch:
#dimension of shape space

d = 10






#############################################

#declare simulation paramaters:
#basel proliferation rate
rho = 1.0

#death rate
b = 1.0

#the number of Tregs
Num_treg = len(  Tregs )

#the number of Tcells
Num_tcell = len(  Tcells )

#the number of sites
Num_sites = self_examples

#the Treg- antigen site binding strenth
c = 1.0

 #generate a Treg-Tcell graph
#######################################

#the vx densities
max_v = 1.0
vx = max_v * np.ones(Num_sites) + np.random.uniform(-max_v/2,max_v/2,Num_sites) 



#layer 1 is the conectivity of the Tcells and antigen sequences
layer1 = np.zeros([Num_tcell,Num_sites])
 #let us try to put in some 'cross reactivy' information:
for i in range(Num_tcell):

	for j in range(self_examples):


		val = bindings[ int( Tcells[i] ) - 1  , j    ]

		layer1[i,j] = val

layer2 = np.zeros([Num_sites , Num_treg])
#let us try to put in some 'cross reactivy' information:
for i in range(Num_sites):

	for j in range(Num_treg):

		val = bindings[ int( Tregs[j] ) - 1 , i    ]

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
