import argparse
import re
import os
import subprocess
import gpt as g
import numpy as np
import sys
import copy
import sys, cgpt
from gluon_objects import get_objects

parameters = {
    "placeholder" : [0]
#    "q" : [0,1,0,0],
}

def calculate_EE(Obj, L, Mask_all, Rsq_list):
    A = g.eval(g.fft([0,1,2]) * Obj)
    B = g.eval(g.adj(g.fft([0,1,2])) * Obj)
    TwoPt=[]
    for dt in range(int(L[3]/2+1)):
        twopt_fft = g.eval(g.adj(g.fft([0,1,2])) * (B * A))
        for ii in range(len(Rsq_list)):
            TwoPt.append(g.slice(twopt_fft * Mask_all[ii], 3))
        B = g.cshift(B, 3, 1)
    TwoPt = np.array(TwoPt).reshape(int(L[3]/2+1), len(Rsq_list), L[3])
    TwoPt = np.mean(TwoPt, axis=2)
    return TwoPt

def calculate_UU(Obj, L, Mask_all, Rsq_list):
    A, B = [],[]
    for ii in range(len(Obj)):
        A.append(g.eval(g.fft([0,1,2]) * Obj[ii]))
        B.append(g.eval(g.adj(g.fft([0,1,2])) * Obj[ii]))
    TwoPt=[]
    for dt in range(int(L[3]/2+1)):
        X = (0.25*( A[0] - A[4])*( B[0] - B[4])
           + 0.25*( A[0] - A[7])*( B[0] - B[7])
           + 0.25*( A[4] - A[7])*( B[4] - B[7]))/3.0
        Y = A[1]*B[1] + A[2]*B[2] + A[5]*B[5]
        UU = (X*4. + Y*2.)/10.
        twopt_fft = g.eval(g.adj(g.fft([0,1,2])) * UU)
        for ii in range(len(Rsq_list)):
            TwoPt.append(g.slice(twopt_fft * Mask_all[ii], 3))
        for ii in range(len(Obj)):
            B[ii] = g.cshift(B[ii], 3, 1)
    TwoPt = np.array(TwoPt).reshape(int(L[3]/2+1), len(Rsq_list), L[3])
    TwoPt = np.mean(TwoPt, axis=2)
    return TwoPt

def main():
    # get path of configuration
    PathConf = g.default.get("--PathConf", "PathConf")
    # get path of output folder
    PathTwoPtOutFolder = g.default.get("--PathTwoPtOutFolder", "PathTwoPtOutFolder")
    # get configuration number
    confnum = g.default.get("--confnum", "confnum")
    # get the list of r^2 to be measured
    Rsq_list = g.default.get_ivec("--Rsq_list", None, 47)
    #Rsq_list = g.default.get_ivec("--Rsq_list", None, 2)
    
    # get gauge field from configuration
    U = g.convert(g.load(PathConf), g.double)
    g.message("finished loading gauge config")
    
    # get additional parameters
    grid = U[0].grid
    L = U[0].grid.gdimensions
    Ns = U[0].grid.gdimensions[0]
    g.message("rsq obtained") # ???
    coor=g.coordinates(U[0])
    objects = get_objects(parameters) # ???

    # no clue what a mask is in this case
    Mask_all = []
    Mask = g.complex(grid)
    rsq = g.complex(grid)
    
    # loading r^2 values to 3D/4D lattice?
    g.coordinate_mask(rsq, np.array([(Ns-i[0] if i[0]>int(Ns/2) else i[0])**2+(Ns-i[1] if i[1]>int(Ns/2) else i[1])**2+(Ns-i[2] if i[2]>int(Ns/2) else i[2])**2 for i in coor]))
    
    for ii in range(len(Rsq_list)):
        g.message("Rsq: ",Rsq_list[ii])
        Mask[:] = 0
        g.coordinate_mask(Mask, rsq[:] <= float(Rsq_list[ii]))
        Mask_all.append(g.copy(Mask))

    # flow parameters
    n_flow = 45 #1/8, 4/8, 9/8, 16/8 for smearing radius = a, 2a, 3a, 4a
    #n_flow = 2 #1/8, 4/8, 9/8, 16/8 for smearing radius = a, 2a, 3a, 4a
    epsilon=0.05
    ft=0
    
    # empty lists for storing data for each flow time step
    flowtimes=[]
    TwoPtBulk, TwoPtShear = [],[]
    OnePtE, OnePtU=[],[]
    
    # gradient flow
    for i_ft in range(n_flow):
        # append flow times
        flowtimes.append(ft)
        g.message("ft: ", ft)
        # get EMT fields
        E = objects.get_gluon_anomaly(U)
        Umunu = objects.get_Umunu(U)
        EE = calculate_EE(E, L, Mask_all, Rsq_list) # Nt/2 X R
        UU = calculate_UU(Umunu, L, Mask_all, Rsq_list) # Nt/2 X R
        TwoPtBulk.append(EE)
        TwoPtShear.append(UU)
        OnePtE.append(sum(g.slice(E, 3))/L[3])
        Utmp=[]
        for ii in range(len(Umunu)):
            Utmp.append(sum(g.slice(Umunu[ii], 3))/L[3])
        OnePtU.append(Utmp)
        if i_ft == 0:
            #U = g.zeuthen_flow(U, epsilon=1e-4, Nstep=1, meas_interval=1)
            U = g.zeuthen_flow(U, epsilon=1e-4, Nstep=10, meas_interval=10)
            ft += 1e-4*10
            U = g.zeuthen_flow(U, epsilon=1e-3, Nstep=9, meas_interval=9)
            ft += 1e-3*9
            U = g.zeuthen_flow(U, epsilon=1e-2, Nstep=9, meas_interval=9)
            ft += 1e-2*9
        else:
            U = g.zeuthen_flow(U, epsilon=epsilon, Nstep=1, meas_interval=1)
            ft += epsilon
    OnePtE=np.array(OnePtE)
    OnePtU=np.array(OnePtU)
    TwoPtBulk = np.array(TwoPtBulk).reshape(n_flow, int(L[3]/2+1), len(Rsq_list))
    TwoPtBulk = TwoPtBulk.real/Ns**3.
    TwoPtShear = np.array(TwoPtShear).reshape(n_flow, int(L[3]/2+1), len(Rsq_list))
    TwoPtShear = TwoPtShear.real/Ns**3.
    TwoPtBulkSave, TwoPtShearSave, OnePtSave = [],[],[]
    for i_ft in range(n_flow):
        for dt in range(int(L[3]/2+1)):
            for count, Rsq in enumerate(Rsq_list):
                TwoPtBulkSave.append([flowtimes[i_ft], dt, Rsq, TwoPtBulk[i_ft, dt, count]])
                TwoPtShearSave.append([flowtimes[i_ft], dt, Rsq, TwoPtShear[i_ft, dt, count]])
        OnePt_tmp=[flowtimes[i_ft], OnePtE[i_ft].real]
        for ii in range(len(Umunu)):
            OnePt_tmp.append(OnePtU[i_ft, ii].real)
        OnePtSave.append(OnePt_tmp)
    np.savetxt(PathTwoPtOutFolder+"/BulkCorr_conf"+str(confnum)+".dat", TwoPtBulkSave, header="ft, dt, Rsq, EE", fmt='%16.15e')
    np.savetxt(PathTwoPtOutFolder+"/ShearCorr_conf"+str(confnum)+".dat", TwoPtShearSave, header="ft, dt, Rsq, UU", fmt='%16.15e')
    np.savetxt(PathTwoPtOutFolder+"/OnePt_conf"+str(confnum)+".dat", OnePtSave, header="ft, E, U00, U01,...U11, U12,...", fmt='%16.15e')

if __name__ == '__main__':
    main()
