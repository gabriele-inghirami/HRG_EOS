# prepare_fce_eos_smash.py - version 0.2.4 - 20/03/2022
# 
# it creates a tabulated EoS assuming full chemical equilibrium. In this version only the energy density and the net baryon density are considered.

import fileinput
import math
import numpy as np
import sys
import os
import pickle
from scipy import optimize
from scipy import special
from scipy import interpolate
import scipy.integrate as integrate
from get_info_smash_hadrons import *

particle_data_path="particles.txt"

plim_min=5 #minimum momentum integration limit in GeV
p_factor=10 #factor by which we multiply the energy density to get the momentum integration limit in GeV 
one_over_twopihbarc3 = 1/(2*((np.pi)**2)*(0.197326**3))
use_Boltzmann_approx=True #if True the +1/-1 Fermion/Boson terms in the distribution are set to 0

#intervals in the rho_B and energy density table
#rhoB_sp=[0,0.001,1,5]
#rhoB_p=[10,50,20]
rhoB_sp=[0.,0.5]
rhoB_p=[51]
#edens_sp=[0.,0.002,0.2,1,5,30]
#edens_p=[20,100,100,40,25]
edens_sp=[0.001,0.01,0.1,1.]
edens_p=[11,51,181]

#root finding tolerance error
root_tol=1.e-6

#fine calculation ratio over min energy density limit
fine_calc=0.1
#ratio over min energy density limit to skip the calculation
skip_calc=0.001
#fine calcolatin points
fine_points=3000

energy_guess_start=0.9
# the initial rhoB is 0
T_guess_start=0.177
muB_guess_start=0
muS_guess_start=0.
muQ_guess_start=0.

#space in messages
sp="    "


#safety check
if energy_guess_start > edens_sp[-1]:
    print("Error, the very first guess of the energy density is out of range; please, choose another one...")
    sys.exit(4)

#we get the hadron data
h=extract_data(particle_data_path,1)
num_hadrons=len(h.name)
stat_factor=[]
for i in range(num_hadrons):
    if(use_Boltzmann_approx):
        stat_factor.append(0)
    else:
        if(abs(h.spin[i]-math.floor(h.spin[i]))>1.e-10): #we have a half integer spin, it is a fermion
            stat_factor.append(1)
        else:
            stat_factor.append(-1)

#we get the mass of the particles with minimum baryon number, strangeness and electric charge
limit_B_mass=100
limit_Q_mass=100
limit_S_mass=100
for i in range(num_hadrons):
    if (h.baryon_number[i] != 0):
        if (limit_B_mass > h.mass[i]):
            limit_B_mass = h.mass[i]
    if (h.electric_charge[i] != 0):
        if (limit_Q_mass > h.mass[i]):
            limit_Q_mass = h.mass[i]
    if (h.strangeness[i] != 0):
        if (limit_S_mass > h.mass[i]):
            limit_S_mass = h.mass[i]

#if False it prints only error messages, if True it writes what it is doing at the moment and the intermediate results 
#verbose=False

def integrand_density(k,T,mu,m,sf):
    return k**2/(np.exp((np.sqrt(m**2+k**2)-mu)/T)+sf)

def integrand_energy_dens(k,T,mu,m,sf):
    en=np.sqrt(m**2+k**2)
    return en*k**2/(np.exp((en-mu)/T)+sf) 

def f_system_fce(x,inargs):
    T_fce=x[0]
    muB_fce=x[1]
    muS_fce=x[2]
    muQ_fce=x[3]
    energy_fce=inargs[0]
    rhoB_fce=inargs[1]
    rhoS_fce=inargs[2]
    rhoQ_fce=inargs[3]
    plim=inargs[4]
    int_energy=0
    int_rhoB=0
    int_rhoS=0
    int_rhoQ=0
#    if(verbose):
#        print("Entering f_system_fce. Guess values for T, mu_B, mu_S, mu_Q: "+str(T_fce)+",  "+str(muB_fce)+",  "+str(muS_fce)+",  "+str(muQ_fce))
#        print("Values of energy density, rhoB and rhoS to be matched: "+str(energy_fce)+",  "+str(rhoB_fce)+",  "+str(rhoS_fce)+",  "+str(rhoQ_fce))
    for i in range(num_hadrons):
        chempot=muB_fce*h.baryon_number[i]+h.strangeness[i]*muS_fce+h.electric_charge[i]*muQ_fce
        int_energy=int_energy+h.spin_deg[i]*(integrate.quad(integrand_energy_dens,0,plim,args=(T_fce,chempot,h.mass[i],stat_factor[i]),limit=300)[0])
        density_integral=(integrate.quad(integrand_density,0,plim,args=(T_fce,chempot,h.mass[i],stat_factor[i]),limit=300)[0])*h.spin_deg[i]
        int_rhoB=int_rhoB+h.baryon_number[i]*density_integral
        int_rhoS=int_rhoS+h.strangeness[i]*density_integral
        int_rhoQ=int_rhoQ+h.electric_charge[i]*density_integral

#    if(verbose):
#        print("Values computed after integration: "+str(int_energy*one_over_twopihbarc3)+",  "+str(int_rhoB*one_over_twopihbarc3)+",  "+str(int_rhoS*one_over_twopihbarc3)+",  "+str(int_rhoQ*one_over_twopihbarc3))
    int_energy=int_energy*one_over_twopihbarc3-energy_fce
    int_rhoB=int_rhoB*one_over_twopihbarc3-rhoB_fce
    int_rhoS=int_rhoS*one_over_twopihbarc3-rhoS_fce
    int_rhoQ=int_rhoQ*one_over_twopihbarc3-rhoQ_fce
    return [int_energy,int_rhoB,int_rhoS,int_rhoQ]

def write_results(of,x):
    fs='{:9.6f}'
    ff='{:9.6e}'
    sp="    "
    for k in x[0:2]:
        of.write(fs.format(k)+sp)
    for k in x[2:]:
        of.write(ff.format(k)+sp)
    of.write("\n")

#we get the name of input and output files
N_input_args=len(sys.argv)-1

if(N_input_args!=3):
   print ('Syntax: ./prepare_fce_tab_eos.py <outputfile_prefix> [rho_S] [rho_Q]')
   sys.exit(1)

outputfile=sys.argv[1]+"_s_"+sys.argv[2]+"_q_"+sys.argv[3]+".dat"
rhoS_input=float(sys.argv[2])
rhoQ_input=float(sys.argv[3])
        
tmp_arr=[]
for i in range(len(rhoB_p)-1):
    tmp_arr.append(np.linspace(rhoB_sp[i],rhoB_sp[i+1],rhoB_p[i],endpoint=False))
tmp_arr.append(np.linspace(rhoB_sp[-2],rhoB_sp[-1],rhoB_p[-1],endpoint=True))
rhoB_points=np.concatenate((tmp_arr[:]))
N_rhoB=len(rhoB_points)

tmp_arr=[]
for i in range(len(edens_p)-1):
    tmp_arr.append(np.linspace(edens_sp[i],edens_sp[i+1],edens_p[i],endpoint=False))
tmp_arr.append(np.linspace(edens_sp[-2],edens_sp[-1],edens_p[-1],endpoint=True))
energy_density_array=np.concatenate((tmp_arr[:]))
N_edens=len(energy_density_array)

T_arr=np.zeros((N_edens,N_rhoB),dtype=np.float64)
muB_arr=np.zeros((N_edens,N_rhoB),dtype=np.float64)
muS_arr=np.zeros((N_edens,N_rhoB),dtype=np.float64)
muQ_arr=np.zeros((N_edens,N_rhoB),dtype=np.float64)

ed_index_start=np.argmin(abs(energy_guess_start-energy_density_array))
energy_density_array_low=energy_density_array[:ed_index_start]
energy_density_array_high=energy_density_array[ed_index_start:]
energy_density_array_low_flipped=np.flip(energy_density_array_low)

T_guess=T_guess_start
muB_guess=muB_guess_start
muS_guess=muS_guess_start
muQ_guess=muQ_guess_start
rhoB_input=rhoB_points[0]
for i in range(len(energy_density_array_low)):
    ed=energy_density_array_low_flipped[i]
    plimit=ed*p_factor
    if (plimit < plim_min):
        plimit=plim_min
    if ((ed > abs(rhoB_input)*limit_B_mass) and (ed > abs(rhoQ_input)*limit_Q_mass) and (ed > abs(rhoS_input)*limit_S_mass)):
        print("Trying to solve energy_density_array_low_flipped for "+str(i)+sp+str(ed)+sp+str(rhoB_input)+sp+str(rhoS_input)+sp+str(rhoQ_input))
        print("Guesses are "+str(T_guess)+sp+str(muB_guess)+sp+str(muS_guess)+sp+str(muQ_guess))
        TMUFCE = optimize.root(f_system_fce, [T_guess,muB_guess,muS_guess,muQ_guess], args=[ed,rhoB_input,rhoS_input,rhoQ_input,plimit])
        if(TMUFCE.success):
            T_guess,muB_guess,muS_guess,muQ_guess=TMUFCE.x[0:4]
            T_arr[ed_index_start-i-1,0]=T_guess
            muB_arr[ed_index_start-i-1,0]=muB_guess
            muS_arr[ed_index_start-i-1,0]=muS_guess
            muQ_arr[ed_index_start-i-1,0]=muQ_guess
        else:
            print("Error for energy density "+str(ed))
    else:
            print("Discarded energy value below threshold: "+str(ed))
T_guess=T_guess_start
muB_guess=muB_guess_start
muS_guess=muS_guess_start
muQ_guess=muQ_guess_start
for i in range(len(energy_density_array_high)):
    ed=energy_density_array_high[i]
    print("Trying to solve energy_density_array_high for "+str(i)+sp+str(ed)+sp+str(rhoB_input)+sp+str(rhoS_input)+sp+str(rhoQ_input))
    print("Guesses are "+str(T_guess)+sp+str(muB_guess)+sp+str(muS_guess)+sp+str(muQ_guess))
    TMUFCE = optimize.root(f_system_fce, [T_guess,muB_guess,muS_guess,muQ_guess], args=[ed,rhoB_input,rhoS_input,rhoQ_input,plimit])
    if(TMUFCE.success):
        T_guess,muB_guess,muS_guess,muQ_guess=TMUFCE.x[0:4]
        T_arr[ed_index_start+i,0]=T_guess
        muB_arr[ed_index_start+i,0]=muB_guess
        muS_arr[ed_index_start+i,0]=muS_guess
        muQ_arr[ed_index_start+i,0]=muQ_guess
    else:
        print("Error for energy density "+str(ed))

print("Preliminary part at 0 baryon density done, continuing with other "+str(N_rhoB-1)+" different baryon density computions")
energy_array_reversed=np.flip(energy_density_array)
for k in range(1,N_rhoB):
    rhoB_input=rhoB_points[k]
    print("DBG k is : "+str(k) + " and rhoB_input is: "+str(rhoB_points[k]))
    elim=max(abs(rhoB_input)*limit_B_mass,abs(rhoQ_input)*limit_Q_mass,abs(rhoS_input)*limit_S_mass)
    print("DBG: elim is "+str(elim))
    elimit_index=np.argmin(abs(elim-energy_array_reversed))
    print("DBG: elimit_index is "+str(elimit_index))
    #if(energy_array_reversed[elimit_index]>elim):
    #    elimit_index=elimit_index+1
    print("DBG Check: en(elim_index): "+str(energy_array_reversed[elimit_index]))
    for i in range(elimit_index):
        ed=energy_array_reversed[i]
        j=N_edens-i-1
        if (i==0):
            T_guess=T_arr[-1,k-1]
            if ((ed==0) or (T_guess==0)):
                 continue
            muB_guess=muB_arr[-1,k-1]
            muS_guess=muS_arr[-1,k-1]
            muQ_guess=muQ_arr[-1,k-1]
        else:
            T_guess=T_arr[j+1,k]
            if ((ed==0) or (T_guess==0)):
                 T_guess=T_arr[j,k-1]
                 if T_guess==0:
                     continue
            muB_guess=muB_arr[j+1,k]
            muS_guess=muS_arr[j+1,k]
            muQ_guess=muQ_arr[j+1,k]
        if ((ed-elim)/elim < skip_calc):
            continue
        if ((ed-elim)/elim < fine_calc):
            num_cycles=int((ed-elim)/elim*fine_points/(elimit_index-i))
            envals=np.flip(np.linspace(energy_array_reversed[i],energy_array_reversed[i-1],num=num_cycles))
            print("Fine computations: "+str(ed)+"    "+str(rhoB_input)+"   "+str((ed-elim)/elim))
            print(str(envals))
            for l in range(num_cycles):
                plimit=envals[l]*p_factor
                if (plimit < plim_min):
                    plimit=plim_min
                print("Trying to solve envals for "+str(l)+sp+str(envals[l])+sp+str(rhoB_input)+sp+str(rhoS_input)+sp+str(rhoQ_input))
                print("Guesses are "+str(T_guess)+sp+str(muB_guess)+sp+str(muS_guess)+sp+str(muQ_guess))
                TMUFCE = optimize.root(f_system_fce, [T_guess,muB_guess,muS_guess,muQ_guess], args=[envals[l],rhoB_input,rhoS_input,rhoQ_input,plimit])
                if(TMUFCE.success):
                    T_guess,muB_guess,muS_guess,muQ_guess=TMUFCE.x[0:4]
                else:
                    break
        print("Trying to solve energy_density_reversed for "+str(i)+sp+str(k)+sp+str(ed)+sp+str(rhoB_input)+sp+str(rhoS_input)+sp+str(rhoQ_input))
        print("Guesses are "+str(T_guess)+sp+str(muB_guess)+sp+str(muS_guess)+sp+str(muQ_guess))
        TMUFCE = optimize.root(f_system_fce, [T_guess,muB_guess,muS_guess,muQ_guess], args=[ed,rhoB_input,rhoS_input,rhoQ_input,plimit])   
        if(TMUFCE.success):
             T_guess,muB_guess,muS_guess,muQ_guess=TMUFCE.x[0:4]
             T_arr[j,k]=T_guess
             muB_arr[j,k]=muB_guess
             muS_arr[j,k]=muS_guess
             muQ_arr[j,k]=muQ_guess
             print("Success: "+str(T_guess)+sp+str(muB_guess)+sp+str(muS_guess)+sp+str(muQ_guess))
        else:
             print("Error for energy density "+str(ed)+" and baryon density "+str(rhoB_input))
             break
         
with open(outputfile,"w") as outf:
    outf.write("#strangeness density\n")
    outf.write(sys.argv[2]+"\n")
    outf.write("#charge density\n")
    outf.write(sys.argv[3]+"\n")
    outf.write("#baryon density    energy density     T     muB     muS    muQ\n")
    for k in range(N_rhoB):
        for i in range(N_edens):
            vals=(rhoB_points[k], energy_density_array[i], T_arr[i,k],muB_arr[i,k],muS_arr[i,k],muQ_arr[i,k])
            write_results(outf,vals)
