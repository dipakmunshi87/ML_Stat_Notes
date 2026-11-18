import numpy as np, json, time
rng=np.random.default_rng(20261117)
n=20; x=np.linspace(-1,1,n); sig=0.3; tau=2.0
truth=np.array([1.0,0.8,-0.6])
y=truth[0]+truth[1]*x+truth[2]*x**2+sig*rng.standard_normal(n)
out={"x":x.tolist(),"y":np.round(y,4).tolist(),"sig":sig,"tau":tau,"truth":truth.tolist()}
y=np.round(y,4)
def setup(d):
    X=np.vander(x,d,increasing=True); G=X.T@X
    P=np.linalg.solve(G,X.T); s=P@y; C=sig**2*np.linalg.inv(G)
    Sig=np.linalg.inv(G/sig**2+np.eye(d)/tau**2); mu=Sig@(X.T@y)/sig**2
    K=sig**2*np.eye(n)+tau**2*X@X.T
    lnZ=-0.5*(y@np.linalg.solve(K,y))-0.5*np.linalg.slogdet(K)[1]-0.5*n*np.log(2*np.pi)
    return X,P,s,C,mu,Sig,lnZ
res={}
for d in (1,2,3,4,5):
    X,P,s,C,mu,Sig,lnZ=setup(d)
    chi2=float(((y-X@s)**2).sum()/sig**2)
    res[d]=dict(ols=s.tolist(),ols_sd=np.sqrt(np.diag(C)).tolist(),mu=mu.tolist(),sd=np.sqrt(np.diag(Sig)).tolist(),lnZ=float(lnZ),chi2=chi2,
                corr=(Sig/np.sqrt(np.outer(np.diag(Sig),np.diag(Sig)))).tolist())
    print(d,"mu",np.round(mu,3),"sd",np.round(np.sqrt(np.diag(Sig)),3),"lnZ %.2f chi2 %.1f"%(lnZ,chi2))
out["exact"]=res
# ABC
abc={}
NS=4_000_000; B=500_000
for d in (2,3,4):
    X,P,s,C,mu,Sig,lnZ=setup(d); Li=np.linalg.inv(np.linalg.cholesky(C))
    t0=time.time(); keep={e:[] for e in (2.0,1.0,0.5)}; rawd=[]; rawth=[]
    for b in range(NS//B):
        th=tau*rng.standard_normal((B,d)); ys=th@X.T+sig*rng.standard_normal((B,n))
        ss=ys@P.T; dist=np.linalg.norm((ss-s)@Li.T,axis=1)
        for e in keep: keep[e].append(th[dist<e])
        dr=np.linalg.norm(ys-y,axis=1)/sig; idx=np.argsort(dr)[:2000]; rawd.append(dr[idx]); rawth.append(th[idx])
        print(f"[progress] d={d} {100*(b+1)*B//NS}% elapsed {time.time()-t0:.0f}s",flush=True)
    abc[d]={}
    for e in keep:
        k=np.concatenate(keep[e]); m=len(k)
        abc[d][str(e)]=dict(kept=m,frac=m/NS,sd_ratio=(k.std(0)/np.sqrt(np.diag(Sig))).tolist() if m>30 else None,
                             shift=((k.mean(0)-mu)/np.sqrt(np.diag(Sig))).tolist() if m>30 else None)
        print(d,e,m,m/NS,None if m<=30 else np.round(k.std(0)/np.sqrt(np.diag(Sig)),2))
    # raw-data ABC with the same kept number as eps=1 summaries
    rd=np.concatenate(rawd); rt=np.concatenate(rawth); m=max(len(np.concatenate(keep[1.0])),200)
    idx=np.argsort(rd)[:m]; k=rt[idx]
    abc[d]["raw"]=dict(kept=int(m),eps_raw=float(rd[idx].max()),sd_ratio=(k.std(0)/np.sqrt(np.diag(Sig))).tolist())
    print(d,"raw",m,rd[idx].max(),np.round(k.std(0)/np.sqrt(np.diag(Sig)),2))
out["abc"]=abc; out["NS"]=NS
json.dump(out,open("exact_abc.json","w"),indent=1)
