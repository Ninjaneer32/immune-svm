import numpy as np 
import matplotlib.pyplot as plt
import cvxpy as cvx
from scipy.linalg import circulant
from scipy.stats import norm
import seaborn as sns
import pandas as pd
from scipy.integrate import solve_ivp
from matplotlib.backends import backend_pdf as bpdf

def BinaryRandomMatrix(S,M,p):
    r = np.random.rand(S,M)
    m = np.zeros((S,M))
    m[r<p] = 1.0
    return m

def MakeAffinities(params):
    sampling = params['sampling']
    if sampling == 'Binary':
        pix = BinaryRandomMatrix(params['Num_tcell'],params['Num_sites'],params['pval_cell']) 
        palphax = (params['c'] + np.random.normal(0,params['sigma_cp'],(params['Num_treg'],params['Num_sites']) ) )* BinaryRandomMatrix(params['Num_treg'],params['Num_sites'],params['pval_treg']) 
    elif sampling == '1D':
        circ = circulant(norm.pdf(np.linspace(-params['Num_sites']/2,params['Num_sites']/2,params['Num_sites'])/params['niche_width'])/norm.pdf(0))
        Tcell_choice = np.random.choice(params['Num_sites'],size=params['Num_tcell'],replace=True)
        Treg_choice = np.random.choice(params['Num_sites'],size=params['Num_treg'],replace=True)
        pix = circ[Tcell_choice,:]
        palphax = params['c']*circ[Treg_choice,:]
    elif sampling == 'Circulant':
        circ = circulant(norm.pdf(np.linspace(-params['Num_sites']/2,params['Num_sites']/2,params['Num_sites'])/params['niche_width']))
        pix = circ[np.linspace(0,params['Num_sites']-1,params['Num_tcell'],dtype=int),:]
        palphax = params['c']*circ[np.linspace(0,params['Num_sites']-1,params['Num_treg'],dtype=int),:]
    elif sampling == 'Fixed_degree':
        pix = BinaryRandomMatrix(params['Num_tcell'],params['Num_sites'],params['pval_cell']) 
        palphax = np.zeros((params['Num_treg'],params['Num_sites']))
        degree = np.asarray(params['degree']+np.random.randn(params['Num_sites'])*params['sigma_degree'],dtype=int)
        for i in range(params['Num_sites']):
            palphax[:degree[i],i] = params['c']*np.ones(degree[i])+np.random.randn(degree[i])*params['sigma_c']
            np.random.shuffle(palphax[:,i])
    else:
        print('Invalid sampling choice. Valid choices are Binary, 1D, Circulant or Fixed_degree.')
        pix = np.nan
        palphax = np.nan
    return pix, palphax

def MakeOverlaps(pix,palphax,vx):
    phi_reg_reg = (palphax*vx).dot(palphax.T)
    phi_cell_reg = (pix*vx).dot(palphax.T)
    rvals = pix.dot(vx)
    return phi_reg_reg, phi_cell_reg, rvals

def TrainNetwork(phi_reg_reg,phi_cell_reg,rvals):
    Num_treg = len(phi_reg_reg)
    Num_tcell = len(phi_cell_reg)
    Treg = cvx.Variable(Num_treg)
    G = np.vstack((-(phi_cell_reg.T/rvals).T,-np.eye(Num_treg)))
    h = np.hstack((-np.ones(Num_tcell),np.zeros(Num_treg)))
    constraints = [G@Treg <= h]
    obj = cvx.Minimize((1/2)*cvx.quad_form(Treg,phi_reg_reg))
    prob = cvx.Problem(obj, constraints)
    prob.solve(solver=cvx.ECOS,abstol=1e-7,feastol=1e-7,abstol_inacc=1e-7,feastol_inacc=1e-7,max_iters=100,verbose=False)
    Tcell=constraints[0].dual_value[:Num_tcell]/rvals
    Treg=Treg.value
    return Tcell,Treg

def ddt_simple(t,y,phi_reg_reg,phi_cell_reg,rvals):
    Num_treg = len(phi_reg_reg)
    Num_tcell = len(phi_cell_reg)
    Tcell = y[:Num_tcell]
    Treg = y[Num_tcell:]
    
    dTcelldt = Tcell*(rvals-phi_cell_reg.dot(Treg))
    dTregdt = Treg*(phi_cell_reg.T.dot(Tcell) - phi_reg_reg.dot(Treg))
    
    return np.hstack((dTcelldt, dTregdt))

#declare simulation paramaters:
nw_vec = np.linspace(1,3.5,3)
Num_rep = 2
thresh=1e-4
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

for k in range(len(nw_vec)):
    params['niche_width'] = nw_vec[k]*params['Num_sites']/params['Num_treg']
    print(k)
    
    for j in range(Num_rep):
        pix, palphax = MakeAffinities(params)
        phi_reg_reg, phi_cell_reg, rvals = MakeOverlaps(pix,palphax,vx)
        Tcell, Treg = TrainNetwork(phi_reg_reg,phi_cell_reg,rvals)
        dgdvx = ((pix*(1-palphax.T.dot(Treg)))**2).mean()
        lam,u = np.linalg.eig(phi_reg_reg)
        alpha= params['Num_treg']/((lam>thresh).sum())
        output.append([alpha,dgdvx])
            
pd.DataFrame(output,columns=['alpha','sensitivity']).to_csv('../data/sensitivity_1D.csv')