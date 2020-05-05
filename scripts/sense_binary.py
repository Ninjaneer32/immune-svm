from immune_functions import *

#declare simulation paramaters:
M_vec = np.linspace(100,300,20,dtype=int)
pvec = [0.1,0.2,0.3]
Num_rep = 30
thresh=1e-3
params={
'sampling' : 'Binary',
#the number of Tregs
'Num_treg' : 10000,
#the number of Tcells
'Num_tcell' : 1000,
#the number of sites
'Num_sites' : 100,
#the Treg- antigen site binding strenth
'c' : 1.0,
#the varience in Treg-antigen binding around zero
'sigma_c' : 0.2,
#the varience in Treg-antigen binding around c
'sigma_cp' : 0.0,
#the binding
'pval_cell' : 0.1,
'pval_treg' : 0.1,
'max_v' : 1.0}
alpha_vec = M_vec/params['Num_sites']

#Define antigen concentrations
vx = np.ones(params['Num_sites']) 

output = []
for p in pvec:
    print(p)
    params['pval_cell'] = p
    params['pval_treg'] = p
    pix, palphax = MakeAffinities(params)
    phi_reg_reg_base, phi_cell_reg_base, rvals = MakeOverlaps(pix,palphax,vx)
    for k in range(len(M_vec)):
        print(k)
        M = M_vec[k]
    
        for j in range(Num_rep):
            Treg_list = np.random.choice(params['Num_treg'],size=M,replace=False)
            phi_reg_reg = phi_reg_reg_base[Treg_list,:]
            phi_reg_reg = phi_reg_reg[:,Treg_list]
            phi_cell_reg = phi_cell_reg_base[:,Treg_list]
            Tcell, Treg = TrainNetwork(phi_reg_reg,phi_cell_reg,rvals)
            dgdvx = ((pix*(1-palphax[Treg_list,:].T.dot(Treg)))**2).mean()
            output.append([p,alpha_vec[k],dgdvx])
            
pd.DataFrame(output,columns=['p','alpha','sensitivity']).to_csv('../data/sensitivity_binary.csv')