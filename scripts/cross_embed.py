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

#draw from the shape space
#dimension of shape space
d = 2

#number of self samples
self_samples = 1000

#draw the antigen distribution from a gaussain
mu_vect = np.zeros([d])

# sigma = np.random.uniform( 1 , 1 , size = (d,d) )
# sigma = 0.5*(sigma + np.transpose(sigma) )
# sigma  = sigma.dot(sigma)

sigma = np.eye(d)

#shape is self, d
#self_dist = np.random.multivariate_normal( mu_vect , sigma , size = ( self_samples )  )
self_dist = np.random.uniform( -1 , +1 , size = ( self_samples , d )  )

#defining the ball function, need the adapive scale!!!
def ball( x, y , scale ):

	e = 0

	val = np.linalg.norm(x - y)

	if ( val < scale ):

		e = 1

	return e


samples = 9500

thymocites = np.zeros([samples , d])
sizes = np.random.uniform( 0.1 , 0.7 , samples )
bindings = np.zeros([samples, self_samples])
totals = np.zeros([samples])

for i in range(samples):

	point = np.random.uniform( -2 , 2 , size = d  )

	thymocites[i, :] = point

	for j in range(self_samples):

		distance = ball(  self_dist[j , :]  , point  , sizes[i] )

		bindings[ i , j ] = distance

	totals[i] = np.sum( bindings[i,:] )



plt.hist( totals  )
plt.show()

Tregs = np.array( [] )
Tcells = np.array( [] )

#treg/tcell fraction
delta = 0.1

for i in range(samples):

	total = np.sum(bindings[i,:])

	
	if (  160*sizes[i] < total < 281*sizes[i] ):

		p = np.random.uniform(0,1)


		if ( p < delta ):
			

				Tregs = np.append( Tregs , [i] ,  axis=None)

		if ( p > delta ):

			Tcells = np.append(  Tcells , [i] ,axis = None )


#the number of Tregs
Num_treg = len(  Tregs )

#the number of Tcells
Num_tcell = len(  Tcells )

#the number of sites
Num_sites = self_samples

#plt.plot( self_dist[:,0] , self_dist[:,1] , '.' , color = 'b' )
#plt.plot( self_dist[ Tregs  ,0] , self_dist[  Tregs ,1] , '.' , color ='r' )
#plt.plot(  thymocites[:,0] ,  thymocites[:,1] , '.' , alpha = 0.6 )

fig, ax = plt.subplots()

for i in range( self_samples ):

	plt.plot( self_dist[:,0] , self_dist[:,1] , '.' , color = 'b' )


for i in range(Num_treg):

	#plt.scatter( thymocites[ int( Tregs[i] ) , 0 ] , thymocites[ int( Tregs[i] ) , 1 ]  , color = 'r' , ms = sizes[ int( Tregs[ i ] ) ]  )

	circle = plt.Circle(   ( thymocites[ int( Tregs[i] ) , 0 ] , thymocites[ int( Tregs[i] ) , 1 ] ) , sizes[ int(Tregs[i])  ] , color = 'r' , edgecolor= 'k' )        

	ax.add_artist(circle)

for i in range(Num_tcell):


	circle = plt.Circle(   ( thymocites[ int( Tcells[i] ) , 0 ] , thymocites[ int( Tcells[i] ) , 1 ] ) , sizes[ int(Tcells[i])  ]  , color = 'r' , edgecolor= 'k' )        

	ax.add_artist(circle)

plt.ylim(-10,10)
plt.xlim(-10,10)
plt.grid()
plt.show()



#layer 1 is the conectivity of the Tcells and antigen sequences
layer1 = np.zeros([Num_tcell,Num_sites])
 #let us try to put in some 'cross reactivy' information:
for i in range(Num_tcell):

	for j in range(self_samples):


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





#generate a Treg-Tcell graph
#######################################

#the vx densities
max_v = 1.0
vx = max_v * np.ones(Num_sites) + np.random.uniform(-max_v/2,max_v/2,Num_sites) 


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
distance = np.sqrt( np.sum( (Treg - (rho)*np.ones(Num_treg) )**2        ) )
print('distance')
print(distance)

