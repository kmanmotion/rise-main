#!/usr/bin/env python3
"""Recompute the quoted reduced examples and Figure 5 points, not a proof."""
from pathlib import Path
import csv, hashlib, io, json, platform, tempfile
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
def write_output(name, data):
    temporary = tempfile.NamedTemporaryFile(dir=OUT, delete=False)
    path = Path(temporary.name)
    try:
        with temporary:
            temporary.write(data)
        path.replace(OUT / name)
    finally:
        path.unlink(missing_ok=True)


def report(result):
    payload = json.dumps(result, indent=2) + "\n"
    write_output("REDUCED_EXAMPLES.json", payload.encode("utf-8"))
    print(payload, end="")


def require(condition, check):
    if not condition:
        report({"status": "FAIL", "check": check})
        raise SystemExit(1)


rows=[];guess=[.579,1.45]
for g,ce,kce in expected:
    fit=root(lambda v:eq(v,g),guess,tol=1e-11)
    require(fit.success and np.max(np.abs(eq(fit.x,g)))<1e-10, f"equilibrium solve at g={g}")
    u,y=fit.x;guess=fit.x
    c=delta/(ell+rho*y+g);m=np.sqrt(u*u-4*alpha/b)
    p,q=(u+m)/2,(u-m)/2;C=m/u;kc=b*u
    require(abs(C-ce)<2e-9 and abs(kc-kce)<2e-9, f"Figure 5 reference at g={g}")
    require(abs(kc*kc*(1-C*C)-4*alpha*b)<1e-10, f"branch identity at g={g}")
    rows.append({'g':g,'u':float(u),'y':float(y),'p':float(p),'q':float(q),
                 'C':float(C),'kappa_c':float(kc),'free_resource':float(1-(1+c)*u)})
v=rows[0];A=np.array([[v['p'],v['q']],[v['q'],v['p']]])
c=delta/(ell+rho*v['y']);z=np.r_[A.ravel(),(c*A).ravel(),v['y']]
h=1e-6;J=np.column_stack([(rhs(z+h*np.eye(9)[i])-rhs(z-h*np.eye(9)[i]))/(2*h) for i in range(9)])
spectral=float(np.linalg.eigvals(J).real.max())
coefficient=float(rho*beta*v['u']*c*(v['p']-v['q']))
require(np.max(np.abs(rhs(z)))<1e-10, "g=0 equilibrium residual")
require(abs(spectral-(-.2598875203))<1e-7, "g=0 spectral abscissa")
require(abs(coefficient-.0096217374)<1e-9, "g=0 support coefficient")
points = io.StringIO(newline="")
writer = csv.DictWriter(points, fieldnames=rows[0])
writer.writeheader()
writer.writerows(rows)
point_bytes = points.getvalue().encode("utf-8")
write_output("FIGURE5_REDUCED_POINTS.csv", point_bytes)
result={'status':'PASS','role':'representative numerical and algebraic checks, not universal proof',
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
        'growth_points':rows,'g0_full_9D_spectral_abscissa':spectral,
        'g0_support_capacity_t2_coefficient':coefficient,
        'figure5_max_allowed_absolute_discrepancy':2e-9,
        'figure5_csv_sha256':hashlib.sha256(point_bytes).hexdigest()}
report(result)
