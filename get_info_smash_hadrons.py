#!/usr/bin/python3

# get_info_smash_hadrons.py 23/02/2022
# 
# it reads the hadron data in smash_sources/input/particles.txt
# and returns them as lists

import fileinput
import math
import sys
import os

class data_bundle:
     name=[]
     pdgid=[]
     mass=[]
     parity=[]
     width=[]
     baryon_number=[]
     strangeness=[]
     electric_charge=[] #positron charge units
     spin=[]
     spin_deg=[]
     antip=[]
     kind=[] #1=SMASH evolved hadrons, 2=leptons, 3=Pythia hadrons not evolved by SMASH


def get_strangeness(q):
    if(q!=3):
        return 0
    else:
        return -1

def get_charge(q):
    if((q==2) or (q==4)):
        return 2
    else:
        return -1

def analize_pdg(pdg):
    len_pdg=len(pdg)
    int_pdg=int(pdg)
    q=[0,0,0]
    if(len_pdg==2): #it is a lepton
        Jtot=2 
        B=0
        s=0
        if(int_pdg<20):
            echarge=-1
            has_antiparticle=True
        else:
            echarge=0
            has_antiparticle=False
        kind=3
    else:
        Jtot=int(pdg[-1])
        q[0]=int(pdg[-2])
        q[1]=int(pdg[-3])
        if(len_pdg>3):
            q[2]=int(pdg[-4])
        if(q[2]>0): #it is a baryon
            B=1
            if(Jtot==9):
                Jtot=10; #this is an exception for N(2200), N(2250) and Lambda(2450) with J=9/2, so that 2J+1=10, but in their ID the last number is 9
        else:
            B=0 #2 quarks: it is a meson

        if(max(q)<4): #it is a hadron evolved by SMASH, it does not contain c or b quarks
            kind=1
        else:
            kind=2

        if((q[2]==0) and (q[0]==q[1])):
            s=0
            echarge=0
            has_antiparticle=False
        else:
            s=sum(map(get_strangeness,q))
            if(B==0):
                s=-s #for mesons the strangeness is the opposite
                if(abs(get_charge(q[0]))==abs(get_charge(q[1]))):
                    echarge=0
                else:
                    echarge=1
            else:
                echarge=int(sum(map(get_charge,q)))/3
            has_antiparticle=True
 
    return Jtot, B, s, echarge, kind, has_antiparticle

def extract_data(inputfile,limit_kind=3):
    # only the particles up to kind = limit_kind are included (1 = SMASH, 2 = SMASH + Pythia, 3 = SMASH + Pythia + Leptons)
    try:
         infile=open(inputfile,"r")
    except OSError:
         print("Could not open/read file: ", inputfile)
         sys.exit(1)

    d=data_bundle()

    for line in infile:
        stuff=line.split()
        if(len(stuff)==0):
            continue
        if (stuff[0][0]=="#"):
            continue
        else:
            n_items_tmp=len(stuff)
            n_items=n_items_tmp
            for k in range(5,n_items_tmp): #we search comments
                if(stuff[k][0]=="#"):
                    n_items=k
                    break
            for k in range(4,n_items):
                 Jtot_tmp, B_tmp, s_tmp, echarge_tmp, kind_tmp, has_antiparticle = analize_pdg(stuff[k])
                 if(kind_tmp>limit_kind):
                     continue
                 d.name.append(stuff[0])
                 d.mass.append(float(stuff[1]))
                 d.width.append(float(stuff[2]))
                 if(stuff[3]=="+"):
                     d.parity.append(1)
                 else:
                     d.parity.append(-1)
                 d.pdgid.append(stuff[k])
                 d.spin.append((Jtot_tmp-1)/2.)
                 d.spin_deg.append(Jtot_tmp)
                 d.baryon_number.append(B_tmp)
                 d.strangeness.append(s_tmp)
                 d.electric_charge.append(echarge_tmp)
                 d.kind.append(kind_tmp)
                 if(has_antiparticle):
                     d.antip.append(1)
                     d.antip.append(-1)
                     d.name.append("anti_"+d.name[-1])
                     d.mass.append(d.mass[-1])
                     d.pdgid.append("-"+d.pdgid[-1])
                     d.width.append(d.width[-1])
                     d.parity.append(-d.parity[-1])
                     d.spin.append(d.spin[-1])
                     d.spin_deg.append(d.spin_deg[-1])
                     d.baryon_number.append(-d.baryon_number[-1])
                     d.strangeness.append(-d.strangeness[-1])
                     d.electric_charge.append(-d.electric_charge[-1])
                     d.kind.append(d.kind[-1])
                 else:
                     d.antip.append(0)
     
    infile.close()
    return d

if (__name__ == "__main__" ):
    if (len(sys.argv)!=2):
        print ('Syntax: ./get_info_smash_hadrons.py <path_to_smash/input/particles.txt>')
        sys.exit(1)
    else:
        r=extract_data(sys.argv[1])
        sp="    "
        for i in range(len(r.name)):
            print("Name: "+r.name[i])
            print("Mass: "+str(r.mass[i])+sp+"Width: "+str(r.width[i])+sp+"Parity: "+str(r.parity[i]))
            print("Spin: "+str(r.spin[i])+sp+"B: "+str(r.baryon_number[i])+sp+"S: "+str(r.strangeness[i])+sp+"Q: "+str(r.electric_charge[i]))

