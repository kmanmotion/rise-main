#!/usr/bin/env python3
"""Replay the explicit 51-variable compatibility example. Numerical evidence only.
Readout uses accessible carrier L=A+U as specified in the current manuscript.
"""
import json, platform
from pathlib import Path
import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import root
from numpy.linalg import eigvals

alpha,b,d=1.0,16.0,1.5
delta,beta,rho,ell,mu=0.1,1.0,2.0,0.5,0.7
g=0.1; eta=5e-4; kappa=8.0; nu=1e-5
CRstar=CEstar=1.0

# Reduced reference
def reduced_g(v):
    u,y=v; h=ell+rho*y+g; c=delta/h; s=1-(1+c)*u
    return np.array([b*u*s*s-d-g-delta*(ell+g)/h,
                     2*beta*u-(mu+g)*y-2*rho*y*delta*u/h])
rs=root(reduced_g,[0.556,1.277],tol=1e-13); assert rs.success
u_host,yred=rs.x; h=ell+rho*yred+g; c=delta/h
p=(u_host+np.sqrt(u_host*u_host-4*alpha/b))/2
q=(u_host-np.sqrt(u_host*u_host-4*alpha/b))/2
Ared=np.array([[p,q],[q,p]],float); Dred=c*Ared

# Layout: host 7 tables (28), M (1), O (2), parasite 4 tables (16), R (2), E (2) = 51.
def unpack(z):
    ofs=0; host=[]
    for _ in range(7): host.append(z[ofs:ofs+4].reshape(2,2)); ofs+=4
    A,D,B,H,X,Y,U=host
    M=z[ofs]; ofs+=1
    O=z[ofs:ofs+2]; ofs+=2
    par=[]
    for _ in range(4): par.append(z[ofs:ofs+4].reshape(2,2)); ofs+=4
    P,Z,XP,YP=par
    R=z[ofs:ofs+2]; ofs+=2
    E=z[ofs:ofs+2]; ofs+=2
    return A,D,B,H,X,Y,U,M,O,P,Z,XP,YP,R,E

def totals(z):
    A,D,B,H,X,Y,U,M,O,P,Z,XP,YP,R,E=unpack(z)
    Rt=R.copy(); Et=E.copy()
    for i in range(2):
      for j in range(2):
        Rt[i]+=A[i,j]+D[i,j]+B[i,j]+2*H[i,j]+3*X[i,j]+3*Y[i,j]+U[i,j]
        Et[j]+=A[i,j]+D[i,j]+B[i,j]+2*H[i,j]+2*X[i,j]+3*Y[i,j]+U[i,j]
        Rt[i]+=P[i,j]+Z[i,j]+2*XP[i,j]+2*YP[i,j]
        Et[j]+=P[i,j]+Z[i,j]+XP[i,j]+2*YP[i,j]
    return Rt,Et

def rhs(t,z,support=True):
    A,D,B,H,X,Y,U,M,O,P,Z,XP,YP,R,E=unpack(z)
    dA=np.zeros((2,2)); dD=np.zeros((2,2)); dB=np.zeros((2,2)); dH=np.zeros((2,2)); dX=np.zeros((2,2)); dY=np.zeros((2,2)); dU=np.zeros((2,2))
    dP=np.zeros((2,2)); dZ=np.zeros((2,2)); dXP=np.zeros((2,2)); dYP=np.zeros((2,2)); dO=np.zeros(2); dM=0.0
    # Explicit free material species: dilution + symmetric growth-balanced import.
    dR=np.full(2,g*CRstar)-g*R
    dE=np.full(2,g*CEstar)-g*E
    for i in range(2):
      for j in range(2):
        # basal R+E <-> B, activation B->A
        vf=R[i]*E[j]/eta; vr=B[i,j]/eta**2; va=B[i,j]/eta
        dR[i]+=-vf+vr; dE[j]+=-vf+vr; dB[i,j]+=vf-vr-va; dA[i,j]+=va
        # 2A <-> H
        vh=A[i,j]**2/eta; vhr=H[i,j]/eta**2
        dA[i,j]+=-2*vh+2*vhr; dH[i,j]+=vh-vhr
        # H+R <-> X
        vx=H[i,j]*R[i]/eta**2; vxr=X[i,j]/eta**2
        dH[i,j]+=-vx+vxr; dR[i]+=-vx+vxr; dX[i,j]+=vx-vxr
        # X+E <-> Y
        vy=X[i,j]*E[j]/eta**2; vyr=Y[i,j]/eta**2
        dX[i,j]+=-vy+vyr; dE[j]+=-vy+vyr; dY[i,j]+=vy-vyr
        # fuel-driven Y -> H+A
        vcat=16.0*Y[i,j]/eta
        dY[i,j]-=vcat; dH[i,j]+=vcat; dA[i,j]+=vcat
        # A+S <-> U; U -> A+O, S chemostatted
        vu=A[i,j]/eta; vur=U[i,j]/eta**2; vo=U[i,j]/eta
        dA[i,j]+=-vu+vur+vo; dU[i,j]+=vu-vur-vo; dO[j]+=vo
        # host turnover, damage/repair/decay
        vt=d*A[i,j]; vd=delta*A[i,j]; vrep=rho*D[i,j]*M; vl=ell*D[i,j]
        dA[i,j]+=-vt-vd+vrep; dD[i,j]+=vd-vrep-vl
        dR[i]+=vt+vl; dE[j]+=vt+vl
        # label-blind error A->P
        verr=nu*A[i,j]; dA[i,j]-=verr; dP[i,j]+=verr
        # parasite P+R <-> XP
        vpr=P[i,j]*R[i]/eta; vprr=XP[i,j]/eta**2
        dP[i,j]+=-vpr+vprr; dR[i]+=-vpr+vprr; dXP[i,j]+=vpr-vprr
        # XP+E <-> YP
        vpe=XP[i,j]*E[j]/eta**2; vper=YP[i,j]/eta**2
        dXP[i,j]+=-vpe+vper; dE[j]+=-vpe+vper; dYP[i,j]+=vpe-vper
        # parasite catalysis YP -> 2P
        vpc=kappa*YP[i,j]/eta
        dYP[i,j]-=vpc; dP[i,j]+=2*vpc
        # parasite turnover/damage/repair
        vpt=d*P[i,j]; vpd=delta*P[i,j]; vprep=rho*Z[i,j]*M; vpl=ell*Z[i,j]
        dP[i,j]+=-vpt-vpd+vprep; dZ[i,j]+=vpd-vprep-vpl
        dR[i]+=vpt+vpl; dE[j]+=vpt+vpl
    # outputs -> maintenance
    for j in range(2):
      vmt=O[j]/eta; dO[j]-=vmt
      if support: dM+=vmt
    dM-=rho*M*(D.sum()+Z.sum())+mu*M
    # dilution of all nonfree internal concentrations
    for arr in [dA,dD,dB,dH,dX,dY,dU,dP,dZ,dXP,dYP]: pass
    dA-=g*A; dD-=g*D; dB-=g*B; dH-=g*H; dX-=g*X; dY-=g*Y; dU-=g*U
    dP-=g*P; dZ-=g*Z; dXP-=g*XP; dYP-=g*YP; dO-=g*O; dM-=g*M
    return np.r_[dA.ravel(),dD.ravel(),dB.ravel(),dH.ravel(),dX.ravel(),dY.ravel(),dU.ravel(),dM,dO,
                 dP.ravel(),dZ.ravel(),dXP.ravel(),dYP.ravel(),dR,dE]

def initial():
    A=Ared.copy(); D=Dred.copy()
    # approximate fast host intermediates using reduced free resources
    boundR=(A+D).sum(axis=1); boundE=(A+D).sum(axis=0)
    R=1-boundR; E=1-boundE
    B=np.zeros((2,2)); H=np.zeros((2,2)); X=np.zeros((2,2)); Y=np.zeros((2,2)); U=np.zeros((2,2))
    for i in range(2):
      for j in range(2):
        B[i,j]=eta*R[i]*E[j]; H[i,j]=eta*A[i,j]**2; X[i,j]=H[i,j]*R[i]; Y[i,j]=X[i,j]*E[j]; U[i,j]=eta*A[i,j]
    # Recompute free R/E to make total moiety exactly 1 including fast complexes.
    tmp=np.r_[A.ravel(),D.ravel(),B.ravel(),H.ravel(),X.ravel(),Y.ravel(),U.ravel(),yred,eta*A.sum(axis=0),
              np.full(4,1e-8),np.zeros(12),R,E]
    Rt,Et=totals(tmp); R+=1-Rt; E+=1-Et
    O=eta*A.sum(axis=0); P=np.full((2,2),1e-8); Z=np.zeros((2,2)); XP=np.zeros((2,2)); YP=np.zeros((2,2))
    return np.r_[A.ravel(),D.ravel(),B.ravel(),H.ravel(),X.ravel(),Y.ravel(),U.ravel(),yred,O,P.ravel(),Z.ravel(),XP.ravel(),YP.ravel(),R,E]

def jac_fd(z,rel=1e-7):
    n=len(z); J=np.zeros((n,n))
    for k in range(n):
      hh=rel*max(1.0,abs(z[k])); zp=z.copy(); zm=z.copy(); zp[k]+=hh; zm[k]-=hh
      J[:,k]=(rhs(0,zp)-rhs(0,zm))/(2*hh)
    return J

def cap(z):
    A,D,B,H,X,Y,U,*_=unpack(z); L=A+U
    return 0.5*(abs(L[0,0]-L[1,0])+abs(L[0,1]-L[1,1]))

z0=initial()
sol=solve_ivp(rhs,(0,200),z0,method='BDF',rtol=3e-9,atol=1e-11)
z=sol.y[:,-1]
A,D,B,H,X,Y,U,M,O,P,Z,XP,YP,R,E=unpack(z)
Rt,Et=totals(z)
res=float(np.max(np.abs(rhs(0,z))))
vals=eigvals(jac_fd(z)); maxreal=float(vals.real.max())
L=A+U
K=L/L.sum(axis=1,keepdims=True)
C=float(0.5*np.abs(K[0]-K[1]).sum())
parasite=float(P.sum()+Z.sum()+XP.sum()+YP.sum())
# matched support block
times=np.array([0.05,0.1,0.2,0.5,1.0])
si=solve_ivp(lambda t,x:rhs(t,x,True),(0,1),z,t_eval=times,method='BDF',rtol=1e-9,atol=1e-11)
sb=solve_ivp(lambda t,x:rhs(t,x,False),(0,1),z,t_eval=times,method='BDF',rtol=1e-9,atol=1e-11)
contr=[float(cap(si.y[:,k])-cap(sb.y[:,k])) for k in range(len(times))]
result={
 'proof_role':'numerical compatibility witness only; not analytic proof',
 'status':'PASS' if (sol.success and res<1e-8 and C>0.35 and min(R)>0.1 and min(E)>0.1 and parasite<1e-3 and maxreal<-0.09 and max(abs(Rt-1))<1e-8 and max(abs(Et-1))<1e-8 and all(x>0 for x in contr)) else 'FAIL',
 'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
 'parameters':{'g':g,'eta':eta,'kappa':kappa,'nu':nu,'C_R_star':CRstar,'C_E_star':CEstar},
 'dimension':51,'solver_success':bool(sol.success),'support_solvers_success':bool(si.success and sb.success),'rhs_residual':res,'operational_C':C,
 'free_R':R.tolist(),'free_E':E.tolist(),'total_R_moiety':Rt.tolist(),'total_E_moiety':Et.tolist(),
 'parasite_total':parasite,'max_real_full_Jacobian':maxreal,'support_block_contrasts':contr,'support_block_times':times.tolist(),'operational_carrier':'L=A+U',
 'note':'Explicit R/E free species and symmetric g*C*V material influx replace the previous algebraic fixed-total closure. Numerical certificate only.'
}
print(json.dumps(result,indent=2))

if not (si.success and sb.success):
    raise RuntimeError("Support assay integration failed")

if result['status']!='PASS':
    raise SystemExit(1)
