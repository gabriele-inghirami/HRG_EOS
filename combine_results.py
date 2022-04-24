# version 0.3.1 - 28/03/2022

# it combines the results obtained with computes_EOS_HG_SMASH_v0.3.py
# if the header if the EoS file changes, the function get_q_s must be modified accordingly
# warning: the input files must refer to distinct rhoQ and rhoS values!

import fileinput
import math
import numpy as np
import sys
import os
import pickle
import gzip

# we use a single value for rhoS, in this case one must also choose that value
use_only_single_rhoS=True
single_rhoS_value=0

# auxiliary functions
hlines=5 # number of lines of the header
def get_q_s(file_name,file_not_externally_open,fln):
    if os.path.exists(file_name):
        if os.stat(file_name).st_size != 0:
           if file_not_externally_open:
               fln=open(file_name,"r")
           fln.readline()
           s=float(fln.readline().split()[0] )
           fln.readline()
           q=float(fln.readline().split()[0] )
           fln.readline() #just to read all the header
           if file_not_externally_open:
              fln.close()
        else:
            print("Error, "+infile+" seems to be empty.")
            sys.exit(1)
    else:
        print("Error, unable to find "+infile+".")
        sys.exit(1)
    return q, s

def get_edens_rhoB(file_name):
    edens=[]
    rhoB=[]
    with open(file_name,"r") as fln:
         # we skip the lines of the header
         for bbr in range(hlines):
             fln.readline()
         # we read the first line of the data
         rhoB_val,edens_val=np.float64(fln.readline().split()[0:2])
         edens.append(edens_val)
         rhoB.append(rhoB_val)
         for kl in fln:
             rhoB_val,edens_val=np.float64(kl.split()[0:2])
             if edens_val==edens[0]:
                 rhoB.append(rhoB_val)
                 break
             else:
                 edens.append(edens_val)
         for kl in fln:
             rhoB_val,edens_val=np.float64(kl.split()[0:2])
             if edens_val==edens[0]:
                 rhoB.append(rhoB_val)
    return edens,rhoB
                 
             
# we get the name of input and output files
N_input_files=len(sys.argv)-3

# the 00 charachter are introduced to help to remind that the first argument is the outputfile
if((N_input_files<3) or (sys.argv[2]!="00")):
   print ('Syntax: ./computes_EOS_HG_SMASH_v0.3.py <outputfile> 00 <inputfile 1> [inputfile 2] ...')
   sys.exit(1)

outputfile=sys.argv[1]
infiles=sys.argv[3:]

q_list=[]
s_list=[]
# to avoid to waste time, we check if input files all exist and we extract the charge and strangeness density
for infile in infiles:
    q_val, s_val = get_q_s(infile,True,None)
    if q_val not in q_list:
        q_list.append(q_val)
    if s_val not in s_list:
        s_list.append(s_val)

s_list.sort()
q_list.sort()

edens_list,rhoB_list=get_edens_rhoB(infiles[0])
ne=len(edens_list)
nb=len(rhoB_list)
nq=len(q_list)
ns=len(s_list)

edens_array=np.array(edens_list,dtype=np.float64)
rhoB_array=np.array(rhoB_list,dtype=np.float64)
rhoS_array=np.array(s_list,dtype=np.float64)
rhoQ_array=np.array(q_list,dtype=np.float64)

Temp=np.zeros((ne,nb,nq,ns),dtype=np.float64)
press=np.zeros((ne,nb,nq,ns),dtype=np.float64)
muB=np.zeros((ne,nb,nq,ns),dtype=np.float64)
muS=np.zeros((ne,nb,nq,ns),dtype=np.float64)
muQ=np.zeros((ne,nb,nq,ns),dtype=np.float64)


for infile in infiles:
    with open(infile,"r") as fln:
         q_val,s_val=get_q_s(infile,False,fln)
         iQ=np.argmin(abs(q_val-rhoQ_array))
         iS=np.argmin(abs(s_val-rhoS_array))
         # we read the data
         for iB in range(nb):
             for iE in range(ne):
                 T_val,p_val,muB_val,muS_val,muQ_val=np.float64(fln.readline().split()[2:])
                 Temp[iE,iB,iQ,iS]=T_val               
                 press[iE,iB,iQ,iS]=p_val               
                 muB[iE,iB,iQ,iS]=muB_val               
                 muS[iE,iB,iQ,iS]=muS_val               
                 muQ[iE,iB,iQ,iS]=muQ_val               

# now we write the output files
ff='{:9.6e}'
sp="     "

if use_only_single_rhoS:
     iS=np.argmin(abs(single_rhoS_value-rhoS_array))

     with open(outputfile,"w") as outf:
         outf.write("# e [GeV/fm^3] nB[1/fm^3] nQ[1/fm^3] T[GeV] p[GeV/fm^3] muB[GeV] muS[GeV] muQ[GeV]\n")
         for iE in range(ne):
             for iB in range(nb):
                 for iQ in range(nq):
                     outf.write(\
                     ff.format(edens_array[iE])+sp+\
                     ff.format(rhoB_array[iB])+sp+\
                     ff.format(rhoQ_array[iQ])+sp+\
                     ff.format(Temp[iE,iB,iQ,iS])+sp+\
                     ff.format(press[iE,iB,iQ,iS])+sp+\
                     ff.format(muB[iE,iB,iQ,iS])+sp+\
                     ff.format(muS[iE,iB,iQ,iS])+sp+\
                     ff.format(muQ[iE,iB,iQ,iS])+"\n")

