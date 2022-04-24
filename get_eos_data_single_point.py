#!/usr/bin/python3

# it returns the data from SMASH HRG EoS
# the program must be launched in the same directory as the tabulated eos
# in the future one might consider to use class

import fileinput
import math
import numpy as np
import sys
import os
from scipy import interpolate


def count_points(infile):
    if (not os.path.isfile(infile)):
        return None
    try: 
        if (count_points.counter[infile] == 1):
            pass
    except:
        try: 
            len(count_points.counter)
        except:
            count_points.counter={}
            count_points.en_list={}
            count_points.B_list={}
            count_points.Q_list={}
            count_points.nE={}
            count_points.nB={}
            count_points.nQ={}
            count_points.counter[infile] = 1
            count_points.en_list[infile]=[]
            count_points.B_list[infile]=[]
            count_points.Q_list[infile]=[]
            with open(infile,"r") as fin:
                # we skip the first row
                fin.readline()
                # we read the first row
                eref,bref,qref=np.float64(fin.readline().split()[0:3])
                count_points.en_list[infile].append(eref)
                count_points.B_list[infile].append(bref)
                count_points.Q_list[infile].append(qref)
                tnE=1;tnB=1;tnQ=1
                for line in fin:
                    enval,bval,qval=np.float64(line.split()[0:3])
                    if ((tnE==1) and (tnB==1)):
                        if (bval==bref):
                            tnQ=tnQ+1
                            count_points.Q_list[infile].append(qval)
                        else:
                            tnB=tnB+1
                            count_points.B_list[infile].append(bval)
                            bref=bval
                    if ((tnE==1) and (enval==eref)):
                        if (bval!=bref):
                            tnB=tnB+1
                            count_points.B_list[infile].append(bval)
                            bref=bval
                    if (enval!=eref):
                        tnE=tnE+1
                        count_points.en_list[infile].append(enval)
                        eref=enval
                
                count_points.nE[infile]=tnE
                count_points.nB[infile]=tnB
                count_points.nQ[infile]=tnQ
    return count_points.nE[infile],\
           count_points.nB[infile],\
           count_points.nQ[infile],\
           count_points.en_list[infile],\
           count_points.B_list[infile],\
           count_points.Q_list[infile]


def extract_eos_data(input_eos_file,input_edens,input_rhoB,input_rhoQ):
    try: 
        extract_eos_data.counter += 1
    except AttributeError:
        extract_eos_data.counter = 1
    except:
        return None
    if (extract_eos_data.counter == 1):
        nE,nB,nQ,en_list,B_list,Q_list=count_points(input_eos_file)

        en_arr=np.array(en_list,dtype=np.float64)
        B_arr=np.array(B_list,dtype=np.float64)
        Q_arr=np.array(Q_list,dtype=np.float64)

        raw_data=np.loadtxt(input_eos_file,skiprows=1,usecols=(3,4,5,6,7))

        extract_eos_data.data=raw_data.reshape((nE,nB,nQ,5))

    temp = interpolate.interpn((en_arr,B_arr,Q_arr),extract_eos_data.data[:,:,:,0],(input_edens,input_rhoB,input_rhoQ))
    press = interpolate.interpn((en_arr,B_arr,Q_arr),extract_eos_data.data[:,:,:,1],(input_edens,input_rhoB,input_rhoQ))
    muB = interpolate.interpn((en_arr,B_arr,Q_arr),extract_eos_data.data[:,:,:,2],(input_edens,input_rhoB,input_rhoQ))
    muQ = interpolate.interpn((en_arr,B_arr,Q_arr),extract_eos_data.data[:,:,:,3],(input_edens,input_rhoB,input_rhoQ))
    muS = interpolate.interpn((en_arr,B_arr,Q_arr),extract_eos_data.data[:,:,:,4],(input_edens,input_rhoB,input_rhoQ))
 
    return temp[0],press[0],muB[0],muS[0],muQ[0]

if (__name__ == "__main__" ):
    if (len(sys.argv)!=5):
        print ('Syntax: ./get_eos_data_single_point.py <tabulated_eos_file> <en_dens [GeV/fm^3]> <rhoB [GeV/fm^3]> <rhoQ [GeV/fm^3]>')
        sys.exit(1)
    else:
        input_eos_file=sys.argv[1]
        input_edens=float(sys.argv[2])
        input_rhoB=float(sys.argv[3])
        input_rhoQ=float(sys.argv[4])
        temp,press,muB,muS,muQ=extract_eos_data(input_eos_file,input_edens,input_rhoB,input_rhoQ)
        print("Temperature: "+str(temp)+" muB: "+str(muB)+" muS: "+str(muS)+" muQ: "+str(muQ)+" press: "+str(press))
