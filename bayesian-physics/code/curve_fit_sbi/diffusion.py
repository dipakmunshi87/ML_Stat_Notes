import numpy as np, json, time, torch, sys, warnings
warnings.filterwarnings("ignore")
from sbi.inference import NPSE, FMPE
E=json.load(open("exact_abc.json")); x=np.array(E["x"]); y=np.array(E["y"]); sig=E["sig"]; tau=E["tau"]; n=len(x)
NSIM=5000; t00=time.time(); jobs=[(d,m,s) for d in (2,3,4) for m in ("NPSE","FMPE") for s in (0,1)]
res=json.load(open("diffusion.json")) if len(sys.argv)>1 else {}
for j,(d,m,seed) in enumerate(jobs):
    key=f"{d}_{m}_{seed}"
    if key in res: continue
    torch.manual_seed(seed); np.random.seed(seed); t0=time.time()
    X=np.vander(x,d,increasing=True); P=np.linalg.solve(X.T@X,X.T); sobs=torch.tensor(P@y,dtype=torch.float32)
    prior=torch.distributions.MultivariateNormal(torch.zeros(d),tau**2*torch.eye(d))
    th=prior.sample((NSIM,)); ys=th.numpy()@X.T+sig*np.random.standard_normal((NSIM,n)); ss=torch.tensor(ys@P.T,dtype=torch.float32)
    inf={"NPSE":NPSE,"FMPE":FMPE}[m](prior=prior,show_progress_bars=False)
    est=inf.append_simulations(th,ss).train()
    if True: post=inf.build_posterior(est)
    else: post=inf.build_posterior(est,sample_with="mcmc",mcmc_method="slice_np_vectorized",mcmc_parameters=dict(num_chains=20,thin=5,warmup_steps=200))
    sam=post.sample((4000,),x=sobs,show_progress_bars=False).numpy()
    ex=E["exact"][str(d)]; mu=np.array(ex["mu"]); sd=np.array(ex["sd"])
    res[key]=dict(sd_ratio=(sam.std(0)/sd).tolist(),shift=((sam.mean(0)-mu)/sd).tolist(),secs=time.time()-t0,
                  corr01=float(np.corrcoef(sam[:,0],sam[:,-1])[0,1]))
    json.dump(res,open("diffusion.json","w"),indent=1)
    el=time.time()-t00; print(f"[progress] {j+1}/{len(jobs)} {key} sd/exact {np.round(sam.std(0)/sd,2)} shift {np.round((sam.mean(0)-mu)/sd,2)} | elapsed {el/60:.1f} min, left ~{el/(j+1)*(len(jobs)-j-1)/60:.1f} min",flush=True)
print("DONE")
