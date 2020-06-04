from immune_functions import *
import pickle

#declare simulation paramaters:
nw_vec = np.linspace(1,10,100)
Num_rep = 10
thresh=1e-6
params={
'sampling' : '1D',
#the number of Tregs
'Num_treg' : 500,
#the number of Tcells
'Num_tcell' : 500,
#the number of sites
'Num_sites' : 5000,
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

####################################################################################
#compute the Treg steady state and active set of constraints
output = []
spectra = []
for k in range(len(nw_vec)):
    params['niche_width'] = nw_vec[k]*params['Num_sites']/params['Num_treg']
    print(k)
    
    for j in range(Num_rep):
        pix, palphax = MakeAffinities(params)
        phi_reg_reg, phi_cell_reg, rvals = MakeOverlaps(pix,palphax,vx)
        Tcell, Treg = TrainNetwork(phi_reg_reg,phi_cell_reg,rvals)
        Qvar = ((1-palphax.T.dot(Treg))**2).mean()
        ILvar = ((1-pix.T.dot(Tcell)/(palphax.T.dot(Treg)))**2).mean()
        lam,u = np.linalg.eig(phi_reg_reg)
        alpha= params['Num_treg']/((np.sqrt(np.abs(lam))>thresh).sum())
        output.append([alpha,Qvar,ILvar])
        spectra.append(lam)

    with open('../data/spectra.dat','wb') as f:
        pickle.dump(spectra,f)
    pd.DataFrame(output,columns=['alpha','Qvar','ILvar']).to_csv('../data/sensitivity_1D.csv')