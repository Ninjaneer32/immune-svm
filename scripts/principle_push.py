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
Num_treg = 95

#the number of Tcells
Num_tcell = 17000

#the number of sites
Num_sites = 25000

#the Treg- antigen site binding strenth
c = 1.0

#the varience in Treg-antigen binding around zero
sigma_c = 0.01

#the varience in Treg-antigen binding around c
sigma_cp = 0.01
 

#the binding
pval_cell = 0.3
pval_reg = 0.6

#mean number of sites that a tcell binds to
print(pval_cell*Num_sites)

#mean number of sites that a treg binds to
print( pval_reg * Num_sites )

#the probability that at least one Tcell binds to the new site
print(  1- (pval_cell)**(Num_tcell) )

#probability that at least one Treg binds to the new site
print( 1 - ( pval_reg )**( Num_treg )  )


#generate a Treg-Tcell graph
#######################################

#the vx densities
max_v = 1.0
vx = max_v * np.ones(Num_sites) + np.random.uniform(-max_v/2,max_v/2,Num_sites) 


#layer 1 is the conectivity of the Tcells and antigen sequences
layer1 = np.zeros([Num_tcell,Num_sites])
layer1 = np.random.choice( [0,1] , size= (Num_tcell,Num_sites), replace = True ,  p= [1- pval_cell,pval_cell]) 


#each site needs at least one t cell to bind to it
for i in range(Num_sites):
	
	if ( np.sum(layer1[:,i]) <= 0  ):

		num = np.random.randint(0,Num_tcell)

		layer1[num,i] = 1

#each tcell should bind to at least site
for i in range(Num_tcell):

	if ( np.sum( layer1[i,:] <= 0  ) ):

		num = np.random.randint(0,Num_sites)

		layer1[ i  , num ] = 1


plt.hist( layer1.sum( axis = 0) )
plt.show()


plt.hist( layer1.sum( axis = 1) )
plt.show()




#layer 2 is the connectivvity of the binding of the antigen sites to the number of Tregs
layer2 = np.zeros([Num_sites,Num_treg])
layer2 = (c + np.random.normal(0,sigma_cp,(Num_sites,Num_treg) ) )* np.random.choice( [0,1] , size = (Num_sites , Num_treg), replace = True ,  p=[1-pval_reg, pval_reg]) 


#make sure that all of the elements have probility that is less than 1
for i in range(Num_sites):
	for j in range(Num_treg):

		if ( layer2[i,j] > 1):

			layer2[i,j] = 1


#add some noise to each site, still make sure each element is less than 1
for i in range(Num_sites):

	for j in range(Num_treg):

		if ( layer2[i,j] == 0 ):

			val = np.abs( np.random.normal(0,sigma_c ) )

			layer2[i,j] = np.random.uniform(0,1)

			if (val < 1):

				layer2[i,j] = val




#add some noise around c to each site
for i in range(Num_sites):

	if ( np.sum(layer2[i,:]) <= c - 0.01   ):

		num = np.random.randint(0,Num_treg)

		layer2[i,num] = c  + np.random.normal(0,sigma_cp )



plt.hist( layer2.sum( axis = 0) )
plt.show()

plt.hist( layer2.sum( axis = 1) )
plt.show()


#compute the mean binding, this will be used when we look at how well our aproxomation works
mean_reg = np.zeros([Num_treg])

for i in range(Num_treg):

	mean_reg[i] = np.sum( layer1[i,:] * layer2[:,i]     )



#compute just graphical overlap
# +1 for each treg connected to tcell
connectivity_count = np.zeros([Num_tcell , Num_treg])

for i in range(Num_tcell):

	for j in range(Num_treg):


		val = 0
		for k in range(Num_sites):
	
			if (  layer1[i,k] > c/10 and layer2[k,j] > c/10 ):
			
				val = val + 1

				connectivity_count[i,j] = 1


#now compute the r_{i} and \phi quantities
######################################################################

#compute the matrix overlaps
phi_reg_reg = np.zeros([Num_treg,Num_treg])

for i in range(Num_treg):
	for j in range(Num_treg):

		phi_reg_reg[i,j] = np.sum( vx[:]*layer2[:,i]*layer2[:,j]  )



#compute the matrix overlaps
phi_cell_reg = np.zeros([Num_tcell,Num_treg])

for i in range(Num_tcell):
	for j in range(Num_treg):

		phi_cell_reg[i,j] = np.sum( vx[:]*layer1[i,:]*layer2[:,j]  )


rvals = np.zeros([Num_tcell])

for i in range(Num_tcell):

	rvals[i] = np.sum( vx[:]*layer1[i,:]    )

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


plt.hist(Treg)
plt.show()


#compute the aproxomation
aprox_values = np.zeros([Num_sites])

for i in range(Num_sites):

	aprox_values[i] = np.sum( Treg[:] * layer2[i,:] )


plt.hist(aprox_values)
plt.show()


quit()
##########################################################################################################################################
#Now, consider the addition of new antigen sites


#the number of new antigen sites introduced, ie number of 'testing' samples
N_trials = 200

#the space of Treg- antigen site binding
cspace = np.linspace( 0 , 1 , 5 )

#count the accuracy, number of tcells binding
count_pos_test = np.zeros([ len(cspace) , N_trials ])

for cpoint in range(len(cspace)):
	
	#the testing val
	cnew = c*cspace[cpoint]

	for run in range(N_trials):

		#the value of the new site density,
		#in this case density is very low compared to prexisting sites
		vx_mag = 1/4.0
	
		#######################################################################################################################################
		#add new site
		vx_new = np.zeros(Num_sites + 1)

		vx_new[0:Num_sites] = vx

		vx_new[Num_sites] = vx_mag 


		#the new Tcell- antigen site binding
		new_layer1 = np.zeros([Num_tcell,Num_sites + 1])
		new_layer1[:,0:Num_sites] = layer1
		new_layer1[:,Num_sites] = np.random.choice( [0,1] , size = (Num_tcell) ,  replace = True ,  p = [(1 - pval_cell ), pval_cell ]) 


		#the new Treg-antigen site binding
		new_layer2 = np.zeros([Num_sites+1,Num_treg])
		new_layer2[0:Num_sites,:] = layer2
		new_layer2[Num_sites, :] = cnew * np.random.choice( [0,1] , size = (Num_treg) ,  replace = True ,  p = [( 1 - pval_reg ), pval_reg ]) 


		#now compute the new r_{i} and matrix elements
		################################################################

		for i in range(Num_treg):
			for j in range(Num_treg):

				phi_reg_reg[i,j] = np.sum( vx_new[:]*new_layer2[:,i]*new_layer2[:,j]  )


		#compute the matrix overlaps
		phi_cell_reg = np.zeros([Num_tcell,Num_treg])


		for i in range(Num_tcell):
			for j in range(Num_treg):

				phi_cell_reg[i,j] = np.sum( vx_new[:]*new_layer1[i,:]*new_layer2[:,j]  )


		rvals = np.zeros([Num_tcell])

		for i in range(Num_tcell):

			rvals[i] = np.sum( vx_new[:]*new_layer1[i,:]    )




		#compute which have positive growth rates
		count = 0
		for i in range(Num_tcell):

			val = ( rvals[i]*rho  -  np.sum( phi_cell_reg[i,:].dot( Treg[:] )  ) )
			if(val > 0):
				count = count + 1


		count_pos_test[cpoint,run] = count



#the mean and error of number of tcells with positve growth rate
avg_test_low = np.zeros([len(cspace)])
test_low_err = np.zeros( [len(cspace)]  )

for i in range(len(cspace)):

	avg_test_low[i] = np.sum(count_pos_test[i,:])/N_trials
	test_low_err[i] =  np.sum(     count_pos_test[i,:]**2      )/N_trials -   ( np.sum(     count_pos_test[i,:]      ) / N_trials )**2 


test_low_err = np.sqrt(test_low_err)

#start again, now for a new v_{x} that is the same as mean of the prexisting graph
#########################################################################################################
count_pos_test = np.zeros([ len(cspace) , N_trials ])

for cpoint in range(len(cspace)):

	#the testing val
	cnew = c*cspace[cpoint]

	
	for run in range(N_trials):

		vx_mag = 1.0

		
		#######################################################################################################################################
		#add new site	
		vx_new = np.zeros(Num_sites + 1)

		vx_new[0:Num_sites] = vx

		vx_new[Num_sites] = vx_mag 


		new_layer1 = np.zeros([Num_tcell,Num_sites + 1])
		new_layer1[:,0:Num_sites] = layer1
		new_layer1[:,Num_sites] = np.random.choice( [0,1] , size = (Num_tcell) ,  replace = True ,  p = [(1-pval_cell ), pval_cell ]) 


		new_layer2 = np.zeros([Num_sites+1,Num_treg])
		new_layer2[0:Num_sites,:] = layer2
		new_layer2[Num_sites, :] = cnew * np.random.choice( [0,1] , size = (Num_treg) ,  replace = True ,  p = [(1 - pval_reg ), pval_reg ]) 


		for i in range(Num_treg):
			for j in range(Num_treg):

				phi_reg_reg[i,j] = np.sum( vx_new[:]*new_layer2[:,i]*new_layer2[:,j]  )


		#compute the matrix overlaps
		phi_cell_reg = np.zeros([Num_tcell,Num_treg])


		for i in range(Num_tcell):
			for j in range(Num_treg):

				phi_cell_reg[i,j] = np.sum( vx_new[:]*new_layer1[i,:]*new_layer2[:,j]  )


		rvals = np.zeros([Num_tcell])

		for i in range(Num_tcell):

			rvals[i] = np.sum( vx_new[:]*new_layer1[i,:]    )


		#compute which have positive growth rates
		count = 0
		for i in range(Num_tcell):

			val = ( rvals[i]*rho  -  np.sum( phi_cell_reg[i,:].dot( Treg[:] )  ) )
			if(val > 0):
				count = count + 1

		count_pos_test[cpoint,run] = count


avg_test_med = np.zeros([len(cspace)])
test_med_err = np.zeros( [len(cspace)]  )

for i in range(len(cspace)):

	avg_test_med[i] = np.sum(count_pos_test[i,:])/N_trials
	test_med_err[i] =  np.sum(     count_pos_test[i,:]**2      )/N_trials -   ( np.sum(     count_pos_test[i,:]      ) / N_trials )**2 


test_med_err = np.sqrt(  test_med_err )



#start again
#########################################################################################################


count_pos_train = np.zeros([ len(cspace) , N_trials ])
count_pos_test = np.zeros([ len(cspace) , N_trials ])

for cpoint in range(len(cspace)):


	#the testing val
	cnew = c*cspace[cpoint]


	for run in range(N_trials):

		vx_mag = 4.0

		
		#######################################################################################################################################
		#add new site	
		vx_new = np.zeros(Num_sites + 1)

		vx_new[0:Num_sites] = vx

		vx_new[Num_sites] = vx_mag 


		new_layer1 = np.zeros([Num_tcell,Num_sites + 1])
		new_layer1[:,0:Num_sites] = layer1
		new_layer1[:,Num_sites] = np.random.choice( [0,1] , size = (Num_tcell) ,  replace = True ,  p = [(1-pval_cell ), pval_cell ]) 


		new_layer2 = np.zeros([Num_sites+1,Num_treg])
		new_layer2[0:Num_sites,:] = layer2
		new_layer2[Num_sites, :] = cnew * np.random.choice( [0,1] , size = (Num_treg) ,  replace = True ,  p = [(1- pval_reg ), pval_reg ]) 


		for i in range(Num_treg):
			for j in range(Num_treg):

				phi_reg_reg[i,j] = np.sum( vx_new[:]*new_layer2[:,i]*new_layer2[:,j]  )


		#compute the matrix overlaps
		phi_cell_reg = np.zeros([Num_tcell,Num_treg])


		for i in range(Num_tcell):
			for j in range(Num_treg):

				phi_cell_reg[i,j] = np.sum( vx_new[:]*new_layer1[i,:]*new_layer2[:,j]  )


		rvals = np.zeros([Num_tcell])

		for i in range(Num_tcell):

			rvals[i] = np.sum( vx_new[:]*new_layer1[i,:]    )




		#compute which have positive growth rates
		count = 0
		for i in range(Num_tcell):

			val = ( rvals[i]*rho  -  np.sum( phi_cell_reg[i,:].dot( Treg[:] )  ) )
			if(val > 0):
				count = count + 1

		count_pos_test[cpoint,run] = count

		#print(count)

avg_test_high = np.zeros([len(cspace)])

test_high_err = np.zeros( [len(cspace)]  )

for i in range(len(cspace)):

	avg_test_high[i] = np.sum(count_pos_test[i,:])/N_trials
	test_high_err[i] =  np.sum(     count_pos_test[i,:]**2      )/N_trials -   ( np.sum(     count_pos_test[i,:]      ) / N_trials )**2 




test_high_err = np.sqrt( test_high_err )



# plt.plot( cspace,  avg_test_high / Num_tcell  , '-.', label="$ v_{x} ="+str(3.0)+"$", color = 'r')
# plt.plot( cspace,  avg_test_med / Num_tcell  , '-.', label="$ v_{x}  ="+str(1.0)+"$" , color = 'g')
# plt.plot( cspace,  avg_test_low / Num_tcell , '-.', label="$ v_{x} ="+str(0.3)+"$", color = 'b')


plt.errorbar( cspace , avg_test_high , yerr = test_high_err, label="$ v_{x} ="+str(4.0)+"$", color = 'r' ,fmt = 'o' ,linestyle='dashed')
plt.errorbar( cspace , avg_test_med , yerr = test_med_err, label="$ v_{x} ="+str(1.0)+"$", color = 'g' , fmt = 'o', linestyle='dashed')
plt.errorbar( cspace , avg_test_low , yerr = test_low_err, label="$ v_{x} ="+str(0.25)+"$", color = 'b' , fmt = 'o', linestyle='dashed')
plt.grid()
#plt.ylim(0,.14)
plt.legend( prop={'size': 21})
plt.tight_layout()
plt.savefig("Principle")
plt.show()
plt.clf()


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

plt.xlabel("$ c'  / c $", fontsize = fontsize1 + 10)
plt.ylabel("$ \\textrm{ Fraction } \\lambda_{i} \\textrm{ with } \\frac{ d \\lambda_{i} }{ dt }  > 0  $" ,fontsize=fontsize1 + 10)


plt.grid()

graph_space = np.linspace(0, 5*rho/c, 200)

for i in range(Num_tcell):

	if( phi_cell_reg[i,0] != 0 ):


		plt.plot(  graph_space,  (rvals[i]*rho -  phi_cell_reg[i,1] * graph_space)/(phi_cell_reg[i,0]), 'b' )


	if( phi_cell_reg[i,0] == 0 ):
		
		plt.axvline(x=rho/c , color = 'b')


plt.plot(Treg[1],Treg[0],'*',markersize = 25,markeredgecolor='k', markerfacecolor='g' )
#plt.legend()

plt.ylim(0,5)
plt.xlim(0,5)

plt.grid()
plt.tight_layout()
plt.show()
plt.clf()