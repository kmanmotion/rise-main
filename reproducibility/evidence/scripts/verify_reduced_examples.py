#!/usr/bin/env python3
"""Recompute the quoted reduced examples and Figure 5 points, not a proof."""
from pathlib import Path
import csv, json, platform
import numpy as np
import scipy
from scipy.optimize import root

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/results'
OUT.mkdir(parents=True,exist_ok=True)
alpha,b,d,delta,beta,rho,ell,mu=1.,16.,1.5,.1,1.,2.,.5,.7
expected=[(0,.5053639619955188,9.270995112117777),
          (.05,.47329942936671915,9.081604449707957),
          (.1,.4361280031495309,8.890028395383084),
          (.2,.33725275439455105,8.497854106024457),
          (.3,.14685837011363528,8.087690542187975),
          (.32,.02553720950188858,8.00260987286559)]
def eq(v,g):
    u,y=v; h=ell+rho*y+g; c=delta/h; s=1-(1+c)*u
    return [b*u*s*s-d-g-delta*(ell+g)/h,
            2*beta*u-(mu+g)*y-2*rho*y*delta*u/h]
def rhs(z,g=0):
    A=z[:4].reshape(2,2);D=z[4:8].reshape(2,2);y=z[8]
    r=1-(A+D).sum(axis=1);e=1-(A+D).sum(axis=0)
    return np.r_[((alpha+b*A*A)*r[:,None]*e[None,:]-(d+delta+g)*A+rho*y*D).ravel(),
                 (delta*A-(ell+rho*y+g)*D).ravel(),
                 beta*A.sum()-(mu+g)*y-rho*y*D.sum()]
rows=[];guess=[.579,1.45]
for g,ce,kce in expected:
    fit=root(lambda v:eq(v,g),guess,tol=1e-11)
    assert fit.success and np.max(np.abs(eq(fit.x,g)))<1e-10
    u,y=fit.x;guess=fit.x
    c=delta/(ell+rho*y+g);m=np.sqrt(u*u-4*alpha/b)
    p,q=(u+m)/2,(u-m)/2;C=m/u;kc=b*u
    assert abs(C-ce)<2e-9 and abs(kc-kce)<2e-9
    assert abs(kc*kc*(1-C*C)-4*alpha*b)<1e-10
    rows.append({'g':g,'u':float(u),'y':float(y),'p':float(p),'q':float(q),
                 'C':float(C),'kappa_c':float(kc),'free_resource':float(1-(1+c)*u)})
v=rows[0];A=np.array([[v['p'],v['q']],[v['q'],v['p']]])
c=delta/(ell+rho*v['y']);z=np.r_[A.ravel(),(c*A).ravel(),v['y']]
h=1e-6;J=np.column_stack([(rhs(z+h*np.eye(9)[i])-rhs(z-h*np.eye(9)[i]))/(2*h) for i in range(9)])
spectral=float(np.linalg.eigvals(J).real.max())
coefficient=float(rho*beta*v['u']*c*(v['p']-v['q']))
assert np.max(np.abs(rhs(z)))<1e-10
assert abs(spectral-(-.2598875203))<1e-7
assert abs(coefficient-.0096217374)<1e-9
with (OUT/'FIGURE5_REDUCED_POINTS.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
result={'status':'PASS','role':'representative numerical and algebraic checks, not universal proof',
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
        'growth_points':rows,'g0_full_9D_spectral_abscissa':spectral,
        'g0_support_capacity_t2_coefficient':coefficient,
        'figure5_max_allowed_absolute_discrepancy':2e-9}
(OUT/'REDUCED_EXAMPLES.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
