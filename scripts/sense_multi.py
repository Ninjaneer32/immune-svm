from immune_functions import *
import pickle

#declare simulation paramaters:
nw_vec = np.linspace(5,20,10)
t0 = 0
tf = 100
Num_rep = 1
thresh=1e-5
params={
'sampling' : 'Multidimensional',
'shape_dim' : 5,
#the number of Tregs
'Num_treg' : 100,
#the number of Tcells
'Num_tcell' : 100,
#the number of sites
'Num_sites' : 1000,
#the Treg- antigen site binding strenth
'c' : 1.0,
#the varience in Treg-antigen binding around zero
'sigma_c' : 0.0,
#the varience in Treg-antigen binding around c
'sigma_cp' : 0.0,
#the binding
'pval_cell' : 0.1,
'pval_treg' : 0.1,
'max_v' : 1.0}

#generate a Treg-Tcell graph
#######################################


#Define antigen concentrations
vx = np.ones(params['Num_sites']) 
y0 = np.ones(params['Num_treg']+params['Num_tcell'])*0.01
####################################################################################
#compute the Treg steady state and active set of constraints
output = []
spectra = []
for k in range(len(nw_vec)):
    params['sigma'] = nw_vec[k]
    print(k)
    
    for j in range(Num_rep):
        #try:
        pix, palphax = MakeAffinities(params)
        phi_reg_reg, phi_cell_reg, rvals = MakeOverlaps(pix,palphax,vx)
        #Tcell, Treg = TrainNetwork(phi_reg_reg,phi_cell_reg,rvals)
        out = solve_ivp(lambda t,y: ddt_simple(t,y,phi_reg_reg,phi_cell_reg,rvals),(t0,tf),y0)
        Tcell = out.y[:params['Num_tcell'],-1]
        Treg = out.y[params['Num_tcell']:,-1]
        Qvar = ((1-palphax.T.dot(Treg))**2).mean()
        ILvar = ((1-pix.T.dot(Tcell)/(palphax.T.dot(Treg)))**2).mean()
        lam,u = np.linalg.eig(phi_reg_reg)
        alpha= params['Num_treg']/((np.sqrt(np.abs(lam))>thresh).sum())
        output.append([alpha,Qvar,ILvar])
        spectra.append(lam)
        #exc    ept:
        #    print(str(params['sigma'])+' failed')

    with open('../data/spectra.dat','wb') as f:
        pickle.dump(spectra,f)
    pd.DataFrame(output,columns=['alpha','Qvar','ILvar']).to_csv('../data/sensitivity_multi_2.csv')