# prepare_fce_eos_smash.py - version 0.5.0 - 04/04/2022
# 
# it creates a tabulated EoS assuming full chemical equilibrium. In this version only the energy density and the net baryon density are considered.
# from v 0.3 we compute also the pressure

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
from get_eos_data_single_point import *

particle_data_path="particles.txt"
helper_eos_file_path="hadgas_eos_SMASH.dat"

plim_min=6 #minimum momentum integration limit in GeV
p_factor=10 #factor by which we multiply the energy density to get the momentum integration limit in GeV 
one_over_twopihbarc3 = 1/(2*((np.pi)**2)*(0.197326**3))
use_Boltzmann_approx=True #if True the +1/-1 Fermion/Boson terms in the distribution are set to 0

#intervals in the rho_B and energy density table
rhoB_sp=[0.,0.5]
#rhoB_sp=[0.0001,0.03]
rhoB_p=[51]
edens_sp=[0.001,0.01,0.1,1.0]
edens_p=[11,51,181]
#edens_sp=[0.001,0.01,0.1]
#edens_p=[11,51]

#root finding tolerance error
root_tol=1.e-6

#ratio over min energy density limit to skip the calculation
skip_calc=1.01

#temperature ratio : maximum ratio between two computations of temperature to increase the resolution
max_temp_ratio=1.3

#levels of refinement
n_ref_max=5
points_per_ref_level=10

#space in messages
sp="    "

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

#0 = no messages, 1 = some info, 2 = full debug messages 
verbose=2

def integrand_density(k,T,mu,m,sf):
    return k**2/(np.exp((np.sqrt(m**2+k**2)-mu)/T)+sf)

def integrand_energy_dens(k,T,mu,m,sf):
    en=np.sqrt(m**2+k**2)
    return en*k**2/(np.exp((en-mu)/T)+sf) 

def integrand_pressure(k,T,mu,m,sf):
    en=np.sqrt(m**2+k**2)
    return k**4/(3*en*(np.exp((en-mu)/T)+sf))

def get_pressure(x,plim):
    T_fce=x[0]
    muB_fce=x[1]
    muS_fce=x[2]
    muQ_fce=x[3]
    int_pressure=0
    for i in range(num_hadrons):
        chempot=muB_fce*h.baryon_number[i]+h.strangeness[i]*muS_fce+h.electric_charge[i]*muQ_fce
        int_pressure=int_pressure+h.spin_deg[i]*(integrate.quad(integrand_pressure,0,plim,args=(T_fce,chempot,h.mass[i],stat_factor[i]),limit=300)[0])
    return one_over_twopihbarc3*int_pressure

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
    if(verbose > 1):
        print("Entering f_system_fce. Guess values for T, mu_B, mu_S, mu_Q: "+str(T_fce)+",  "+str(muB_fce)+",  "+str(muS_fce)+",  "+str(muQ_fce))
        print("Values of energy density, rhoB and rhoS to be matched: "+str(energy_fce)+",  "+str(rhoB_fce)+",  "+str(rhoS_fce)+",  "+str(rhoQ_fce))
    for i in range(num_hadrons):
        chempot=muB_fce*h.baryon_number[i]+h.strangeness[i]*muS_fce+h.electric_charge[i]*muQ_fce
        int_energy=int_energy+h.spin_deg[i]*(integrate.quad(integrand_energy_dens,0,plim,args=(T_fce,chempot,h.mass[i],stat_factor[i]),limit=300)[0])
        density_integral=(integrate.quad(integrand_density,0,plim,args=(T_fce,chempot,h.mass[i],stat_factor[i]),limit=300)[0])*h.spin_deg[i]
        int_rhoB=int_rhoB+h.baryon_number[i]*density_integral
        int_rhoS=int_rhoS+h.strangeness[i]*density_integral
        int_rhoQ=int_rhoQ+h.electric_charge[i]*density_integral

    if(verbose > 1):
        print("Values computed after integration: "+str(int_energy*one_over_twopihbarc3)+",  "+str(int_rhoB*one_over_twopihbarc3)+",  "+str(int_rhoS*one_over_twopihbarc3)+",  "+str(int_rhoQ*one_over_twopihbarc3))
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

nE_tmp,nB_tmp,nQ_tmp,en_list_helper_EoS_tmp,B_list_helper_EoS_tmp,Q_list_helper_EoS_tmp=count_points(helper_eos_file_path)
        
en_list_helper_EoS=np.array(en_list_helper_EoS_tmp,dtype=np.float64)
B_list_helper_EoS=np.array(B_list_helper_EoS_tmp,dtype=np.float64)
Q_list_helper_EoS=np.array(Q_list_helper_EoS_tmp,dtype=np.float64)

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

if energy_density_array[-1] < en_list_helper_EoS[0] :
    print("Sorry, but the maximum value of the energy density array must be at least "+str(en_list_helper_EoS[0])+ \
        "which is the minimum value of the tabulated Eos "+helper_eos_path)
    sys.exit(2)

energy_guess_start_index=np.argmin(abs(energy_density_array[-1]-en_list_helper_EoS))
edens_input=en_list_helper_EoS[energy_guess_start_index]
rhoQ_guess_start_index=np.argmin(abs(rhoQ_input-Q_list_helper_EoS)) #rhoQ provided as invocation argument
rhoQ_guess_start=Q_list_helper_EoS[rhoQ_guess_start_index]
rhoB_guess_start_index=np.argmin(abs(rhoB_points[0]-B_list_helper_EoS)) #rhoB is hardcoded, we pick up the smallest value
rhoB_guess_start=B_list_helper_EoS[rhoB_guess_start_index]

T_guess_start,press_tmp,muB_guess_start,muS_guess_start,muQ_guess_start=extract_eos_data(helper_eos_file_path,edens_input,rhoB_guess_start,rhoQ_guess_start)

if (verbose > 1):
    print("DBG initial guesses: "+str(T_guess_start)+"  "+str(muB_guess_start)+"  "+str(muS_guess_start)+"  "+str(muQ_guess_start))


T_arr=np.zeros((N_edens,N_rhoB),dtype=np.float64)
muB_arr=np.zeros((N_edens,N_rhoB),dtype=np.float64)
muS_arr=np.zeros((N_edens,N_rhoB),dtype=np.float64)
muQ_arr=np.zeros((N_edens,N_rhoB),dtype=np.float64)
p_arr=np.zeros((N_edens,N_rhoB),dtype=np.float64)

ed_index_start=np.argmin(abs(edens_input-energy_density_array))
energy_density_array_low=energy_density_array[:ed_index_start]
energy_density_array_high=energy_density_array[ed_index_start:]
energy_density_array_low_flipped=np.flip(energy_density_array_low)

T_guess=T_guess_start
muB_guess=muB_guess_start
muS_guess=muS_guess_start
muQ_guess=muQ_guess_start
rhoB_input=rhoB_points[0]
failure_in_refinement = False
for i in range(len(energy_density_array_low_flipped)):
    if failure_in_refinement:
        break
    ed=energy_density_array_low_flipped[i]
    plimit=ed*p_factor
    if (plimit < plim_min):
        plimit=plim_min
    elim=max(abs(rhoB_input)*limit_B_mass,abs(rhoQ_input)*limit_Q_mass,abs(rhoS_input)*limit_S_mass)
    if elim != 0:
        if ed/elim < skip_calc:
            break
    go_on = True
    ref_level = 0
    while (go_on):
        skip_because_failed = False
        if (ref_level == n_ref_max) :
            failure_in_refinement = True
            if (verbose > 0):
                print("Reached max level of refinement recursion in energy_density_array_low, I stop")
            break
        if (ref_level > 0):
            if (verbose > 0):
                print("Trying ref level :"+str(ref_level))
            num_cycles=int(points_per_ref_level**ref_level)
            envals=np.flip(np.linspace(energy_density_array_low_flipped[i],energy_density_array_low_flipped[i-1],num=num_cycles))
            if (verbose > 0):
                print("Fine computations: "+str(ed)+"    "+str(rhoB_input)+"   "+str((ed-elim)/elim))
            if (verbose > 1):
                print(str(envals))
            for l in range(num_cycles):
                plimit=envals[l]*p_factor
                if (plimit < plim_min):
                    plimit=plim_min
                if (verbose > 0):
                    print("Fine computations: "+str(ed)+"    "+str(rhoB_input)+"   "+str((ed-elim)/elim))
                    print("Trying to solve envals for "+str(l)+sp+str(envals[l])+sp+str(rhoB_input)+sp+str(rhoS_input)+sp+str(rhoQ_input))
                    print("Guesses are "+str(T_guess)+sp+str(muB_guess)+sp+str(muS_guess)+sp+str(muQ_guess))
                TMUFCE = optimize.root(f_system_fce, [T_guess,muB_guess,muS_guess,muQ_guess], args=[envals[l],rhoB_input,rhoS_input,rhoQ_input,plimit])
                if(TMUFCE.success):
                    T_guess,muB_guess,muS_guess,muQ_guess=TMUFCE.x[0:4]
                    if (verbose > 0):
                        print("Success: "+str(T_guess)+sp+str(muB_guess)+sp+str(muS_guess)+sp+str(muQ_guess))
                    skip_because_failed = False
                else:
                    if (verbose > 0):
                        print("Failed, increasing refinement level from "+str(ref_level)+" to "+str(ref_level+1))
                    skip_because_failed = True
                    ref_level=ref_level+1
                    break
        if skip_because_failed:
             continue
        if (verbose > 0):
            print("Trying to solve energy_density_array_low_flipped for "+str(i)+sp+str(ed)+sp+str(rhoB_input)+sp+str(rhoS_input)+sp+str(rhoQ_input))
            print("Guesses are "+str(T_guess)+sp+str(muB_guess)+sp+str(muS_guess)+sp+str(muQ_guess))
        TMUFCE = optimize.root(f_system_fce, [T_guess,muB_guess,muS_guess,muQ_guess], args=[ed,rhoB_input,rhoS_input,rhoQ_input,plimit])
        if(TMUFCE.success):
            T_new,muB_new,muS_new,muQ_new=TMUFCE.x[0:4]
            temp_ratio = T_guess/T_new
            if (temp_ratio > max_temp_ratio):
                if (verbose > 0):
                    print("Ratio between last and current computed temperature: "+str(temp_ratio)+", while maximum allowed is: "+str(max_temp_ratio))
                    print("Increasing refinement level from "+str(ref_level)+" to "+str(ref_level+1))
                ref_level=ref_level+1
            else:     
                T_guess,muB_guess,muS_guess,muQ_guess=T_new,muB_new,muS_new,muQ_new
                if (verbose > 0):
                    print("Success: "+str(T_guess)+sp+str(muB_guess)+sp+str(muS_guess)+sp+str(muQ_guess))
                T_arr[ed_index_start-i-1,0]=T_guess
                muB_arr[ed_index_start-i-1,0]=muB_guess
                muS_arr[ed_index_start-i-1,0]=muS_guess
                muQ_arr[ed_index_start-i-1,0]=muQ_guess
                p_arr[ed_index_start-i-1,0]=get_pressure([T_guess,muB_guess,muS_guess,muQ_guess],plimit)
                go_on = False 
        else:
            if (verbose > 0):
                print("Error for energy density "+str(ed))
            ref_level=ref_level+1
if (verbose > 0):
    print("Low energy density array done, working now with the high energy density part")
T_guess=T_guess_start
muB_guess=muB_guess_start
muS_guess=muS_guess_start
muQ_guess=muQ_guess_start
for i in range(len(energy_density_array_high)):
    ed=energy_density_array_high[i]
    if (verbose > 0):
        print("Trying to solve energy_density_array_high for "+str(i)+sp+str(ed)+sp+str(rhoB_input)+sp+str(rhoS_input)+sp+str(rhoQ_input))
        print("Guesses are "+str(T_guess)+sp+str(muB_guess)+sp+str(muS_guess)+sp+str(muQ_guess))
    TMUFCE = optimize.root(f_system_fce, [T_guess,muB_guess,muS_guess,muQ_guess], args=[ed,rhoB_input,rhoS_input,rhoQ_input,plimit])
    if(TMUFCE.success):
        T_guess,muB_guess,muS_guess,muQ_guess=TMUFCE.x[0:4]
        T_arr[ed_index_start+i,0]=T_guess
        muB_arr[ed_index_start+i,0]=muB_guess
        muS_arr[ed_index_start+i,0]=muS_guess
        muQ_arr[ed_index_start+i,0]=muQ_guess
        p_arr[ed_index_start+i,0]=get_pressure([T_guess,muB_guess,muS_guess,muQ_guess],plimit)
    else:
        if (verbose > 0):
            print("Error for energy density "+str(ed))

if (verbose > 0):
    print("Preliminary part at 0 baryon density done, continuing with other "+str(N_rhoB-1)+" different baryon density computions")
energy_array_reversed=np.flip(energy_density_array)
n_en=len(energy_array_reversed)
for k in range(1,N_rhoB):
    rhoB_input=rhoB_points[k]
    if (verbose > 0):
        print("DBG k is : "+str(k) + " and rhoB_input is: "+str(rhoB_points[k]))
    elim=max(abs(rhoB_input)*limit_B_mass,abs(rhoQ_input)*limit_Q_mass,abs(rhoS_input)*limit_S_mass)
    if (verbose > 0):
        print("DBG: elim is "+str(elim))
    failure_in_refinement = False
    for i in range(n_en):
        ed=energy_array_reversed[i]
        if (verbose > 1):
            print("DBG: Iteration "+str(i)+" energy density: "+str(ed))
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
        if failure_in_refinement:
            break
        plimit=ed*p_factor
        if (plimit < plim_min):
            plimit=plim_min
        if elim != 0:
            if ed/elim < skip_calc:
                if (verbose > 0):
                    print("Energy density ratio with limit: "+str(ed/elim)+" < "+str(skip_calc)+" , skipping energy density "+str(ed))
                break
        go_on = True
        ref_level = 0
        while (go_on):
            skip_because_failed = False
            if (ref_level == n_ref_max) :
                failure_in_refinement = True
                if (verbose > 0):
                    print("Reached max level of refinement recursion, I stop")
                break
            if (ref_level == 0):
                T_guess_start=T_guess
                muB_guess_start=muB_guess
                muQ_guess_start=muQ_guess
                muS_guess_start=muS_guess
            else: 
                T_guess=T_guess_start
                muB_guess=muB_guess_start
                muQ_guess=muQ_guess_start
                muS_guess=muS_guess_start
                if (verbose > 1):
                    print("Trying ref level :"+str(ref_level))
                num_cycles=int(points_per_ref_level**ref_level)
                envals=np.flip(np.linspace(energy_array_reversed[i],energy_array_reversed[i-1],num=num_cycles))
                if (verbose > 0):
                    print("Fine computations: "+str(ed)+"    "+str(rhoB_input)+"   "+str((ed-elim)/elim))
                if (verbose > 1):
                    print(str(envals))
                for l in range(num_cycles):
                    plimit=envals[l]*p_factor
                    if (plimit < plim_min):
                        plimit=plim_min
                    if (verbose > 1):
                        print("Trying to solve envals for "+str(l)+sp+str(envals[l])+sp+str(rhoB_input)+sp+str(rhoS_input)+sp+str(rhoQ_input))
                        print("Guesses are "+str(T_guess)+sp+str(muB_guess)+sp+str(muS_guess)+sp+str(muQ_guess))
                    TMUFCE = optimize.root(f_system_fce, [T_guess,muB_guess,muS_guess,muQ_guess], args=[envals[l],rhoB_input,rhoS_input,rhoQ_input,plimit])
                    if(TMUFCE.success):
                        T_guess,muB_guess,muS_guess,muQ_guess=TMUFCE.x[0:4]
                        if (verbose > 1):
                            print("Success: "+str(T_guess)+sp+str(muB_guess)+sp+str(muS_guess)+sp+str(muQ_guess))
                        skip_because_failed = False
                    else:
                        if (verbose > 1):
                            print("Failed, increasing refinement level from "+str(ref_level)+" to "+str(ref_level+1))
                        skip_because_failed = True
                        ref_level=ref_level+1
                        break
            if skip_because_failed:
                 continue
            if (verbose > 0):
                print("Trying to solve energy_density_reversed for "+str(i)+sp+str(ed)+sp+str(rhoB_input)+sp+str(rhoS_input)+sp+str(rhoQ_input))
                print("Guesses are "+str(T_guess)+sp+str(muB_guess)+sp+str(muS_guess)+sp+str(muQ_guess))
            TMUFCE = optimize.root(f_system_fce, [T_guess,muB_guess,muS_guess,muQ_guess], args=[ed,rhoB_input,rhoS_input,rhoQ_input,plimit])
            if(TMUFCE.success):
                T_new,muB_new,muS_new,muQ_new=TMUFCE.x[0:4]
                temp_ratio = T_guess/T_new
                if (temp_ratio > max_temp_ratio):
                    if (verbose > 0):
                        print("Ratio between last and current computed temperature: "+str(temp_ratio)+", while maximum allowed is: "+str(max_temp_ratio))
                        print("Increasing refinement level from "+str(ref_level)+" to "+str(ref_level+1))
                    ref_level=ref_level+1
                else:     
                    T_guess,muB_guess,muS_guess,muQ_guess=T_new,muB_new,muS_new,muQ_new
                    T_arr[j,k]=T_guess
                    muB_arr[j,k]=muB_guess
                    muS_arr[j,k]=muS_guess
                    muQ_arr[j,k]=muQ_guess
                    p_arr[j,k]=get_pressure([T_guess,muB_guess,muS_guess,muQ_guess],plimit)
                    if (verbose > 0):
                        print("Success: "+str(T_guess)+sp+str(muB_guess)+sp+str(muS_guess)+sp+str(muQ_guess)+sp+str(p_arr[j,k]))
                    go_on = False 
            else:
                if (verbose > 0):
                    print("Error for energy density "+str(ed)+" and baryon density "+str(rhoB_input))
                ref_level=ref_level+1
         
with open(outputfile,"w") as outf:
    outf.write("#strangeness density\n")
    outf.write(sys.argv[2]+"\n")
    outf.write("#charge density\n")
    outf.write(sys.argv[3]+"\n")
    outf.write("#baryon density    energy density     T     p    muB     muS    muQ\n")
    for k in range(N_rhoB):
        for i in range(N_edens):
            vals=(rhoB_points[k], energy_density_array[i], T_arr[i,k], p_arr[i,k], muB_arr[i,k], muS_arr[i,k], muQ_arr[i,k])
            write_results(outf,vals)
