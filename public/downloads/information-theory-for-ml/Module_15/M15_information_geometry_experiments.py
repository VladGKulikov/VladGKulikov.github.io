from __future__ import annotations

from pathlib import Path
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'generated_experiment'
OUT.mkdir(exist_ok=True)


def bernoulli_kl(p: np.ndarray | float, q: np.ndarray | float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))


def local_kl_experiment() -> dict[str, float]:
    p0 = 0.2
    deltas = np.logspace(-5, -1, 160)
    exact = bernoulli_kl(p0, p0+deltas)
    fisher = 1/(p0*(1-p0))
    quadratic = 0.5*fisher*deltas**2
    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    ax.loglog(deltas, exact, label='exact KL')
    ax.loglog(deltas, quadratic, '--', label=r'$\frac{1}{2}F(p)\delta^2$')
    ax.set_xlabel(r'$\delta$')
    ax.set_ylabel('nats')
    ax.set_title('Bernoulli: local KL and Fisher quadratic approximation')
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT/'M15_local_kl.png', dpi=190)
    plt.close(fig)
    print('LOCAL KL')
    for d in (0.1, 0.05, 0.01, 0.001):
        e=float(bernoulli_kl(p0,p0+d)); q=0.5*fisher*d*d
        print(f'delta={d:g} exact={e:.10f} quadratic={q:.10f} ratio={e/q:.6f}')
    return {'fisher': fisher, 'ratio_1e3': float(bernoulli_kl(p0,p0+1e-3)/(0.5*fisher*1e-6))}


def sigmoid(z: np.ndarray | float) -> np.ndarray:
    return 1/(1+np.exp(-np.asarray(z,dtype=float)))


def logit(p: float) -> float:
    return math.log(p/(1-p))


def reparameterization_experiment() -> dict[str,float]:
    target=0.8; p0=0.1; dt=0.01; steps=300
    time=np.arange(steps+1)*dt
    p_nat=np.empty(steps+1); p_nat[0]=p0
    z_nat=np.empty(steps+1); z_nat[0]=logit(p0)
    for t in range(steps):
        p_nat[t+1]=p_nat[t]-dt*(p_nat[t]-target)
        p=float(sigmoid(z_nat[t]))
        z_nat[t+1]=z_nat[t]-dt*(p-target)/(p*(1-p))
    pz_nat=sigmoid(z_nat)
    # Coordinate-dependent ordinary-gradient examples.
    p_gd=np.empty(steps+1); p_gd[0]=p0
    z_gd=np.empty(steps+1); z_gd[0]=logit(p0)
    for t in range(steps):
        p=np.clip(p_gd[t],1e-6,1-1e-6)
        grad=(p-target)/(p*(1-p))
        p_gd[t+1]=np.clip(p-0.002*grad,1e-6,1-1e-6)
        p=float(sigmoid(z_gd[t])); z_gd[t+1]=z_gd[t]-0.08*(p-target)
    pz_gd=sigmoid(z_gd)
    exact=target+(p0-target)*np.exp(-time)
    fig, ax = plt.subplots(figsize=(9.2,5.4))
    ax.plot(time,p_gd,label='ordinary GD in p')
    ax.plot(time,pz_gd,label='ordinary GD in logit')
    ax.plot(time,p_nat,'--',label='natural Euler in p')
    ax.plot(time,pz_nat,':',label='natural Euler in logit')
    ax.plot(time,exact,color='black',linewidth=1.2,label='exact natural flow')
    ax.set_xlabel('time / iteration scale'); ax.set_ylabel('Bernoulli p')
    ax.set_ylim(0,0.9); ax.set_title('Reparameterization: vector field vs finite steps')
    ax.legend(fontsize=9); fig.tight_layout(); fig.savefig(OUT/'M15_reparameterization.png',dpi=190); plt.close(fig)
    nat=float(np.max(np.abs(p_nat-pz_nat))); ordinary=float(np.max(np.abs(p_gd-pz_gd)))
    print('\nREPARAMETERIZATION')
    print(f'max ordinary trajectory difference={ordinary:.6f}')
    print(f'max natural Euler difference={nat:.6f}')
    return {'ordinary_difference':ordinary,'natural_difference':nat}


def empirical_fisher_experiment() -> dict[str,float]:
    q=0.8
    p=np.linspace(0.01,0.99,300)
    model=p*(1-p)
    data=q*(1-p)**2+(1-q)*p**2
    fig, ax=plt.subplots(figsize=(8.8,5.2))
    ax.plot(p,model,label=r'model Fisher $p(1-p)$')
    ax.plot(p,data,label='gradient second moment under data')
    ax.axvline(q,color='black',linewidth=0.8,alpha=0.5)
    ax.set_xlabel('model probability p'); ax.set_ylabel('matrix entry / second moment')
    ax.set_title('Model Fisher and observed-gradient second moment differ')
    ax.legend(); fig.tight_layout(); fig.savefig(OUT/'M15_empirical_fisher.png',dpi=190); plt.close(fig)
    print('\nMODEL VS EMPIRICAL')
    vals={}
    for p0 in (0.2,0.5,0.8):
        f=p0*(1-p0); e=q*(1-p0)**2+(1-q)*p0**2
        vals[p0]=(f,e); print(f'p={p0:.1f} model={f:.6f} data={e:.6f}')
    return {'model_p02':vals[0.2][0],'data_p02':vals[0.2][1]}


def categorical_kl(p: np.ndarray,q: np.ndarray)->float:
    return float(np.sum(p*np.log(p/q)))


def ppo_counterexample() -> dict[str,float]:
    old=np.array([0.50,0.49,0.01])
    q1=np.geomspace(0.49,1e-8,240)
    policies=np.column_stack([np.full_like(q1,0.50),q1,0.50-q1])
    kl=np.array([categorical_kl(old,q) for q in policies])
    ratio=policies[:,0]/old[0]
    surrogate=np.minimum(ratio,np.clip(ratio,0.8,1.2))
    x=-np.log10(q1)
    fig, ax=plt.subplots(figsize=(9.2,5.3))
    ax.plot(x,kl,label=r'$D_{KL}(\pi_{old}\|\pi_{new})$')
    ax.set_xlabel(r'$-\log_{10}\pi_{new}(a_1)$'); ax.set_ylabel('KL, nats')
    ax2=ax.twinx(); ax2.plot(x,surrogate,'--',color='black',label='sampled PPO surrogate')
    ax2.set_ylabel('sampled surrogate')
    lines,labels=ax.get_legend_handles_labels(); lines2,labels2=ax2.get_legend_handles_labels()
    ax.legend(lines+lines2,labels+labels2,loc='upper left',fontsize=9)
    ax.set_title('PPO clipping on sampled actions is not a global KL constraint')
    fig.tight_layout(); fig.savefig(OUT/'M15_ppo_counterexample.png',dpi=190); plt.close(fig)
    m=float(kl.max()); print('\nPPO COUNTEREXAMPLE'); print(f'max KL={m:.6f} nats; sampled ratio={ratio[0]:.1f}')
    return {'max_kl':m}


def flow_coupling_experiment() -> dict[str,float]:
    mu0,s0=-1.0,0.7; mu1,s1=2.0,1.4
    t=np.linspace(0,1,200)
    var_ot=((1-t)*s0+t*s1)**2
    var_ind=(1-t)**2*s0**2+t**2*s1**2
    action_ot=(mu1-mu0)**2+(s1-s0)**2
    action_ind=(mu1-mu0)**2+s0**2+s1**2
    fig, ax=plt.subplots(figsize=(8.8,5.2))
    ax.plot(t,var_ot,label='shared-noise / OT coupling')
    ax.plot(t,var_ind,label='independent coupling')
    ax.set_xlabel('t'); ax.set_ylabel(r'$\mathrm{Var}(X_t)$')
    ax.set_title('The coupling determines the Gaussian interpolation path')
    ax.legend(); fig.tight_layout(); fig.savefig(OUT/'M15_flow_couplings.png',dpi=190); plt.close(fig)
    print('\nFLOW COUPLINGS')
    print(f'OT action={action_ot:.6f}; independent action={action_ind:.6f}; gap={action_ind-action_ot:.6f}')
    return {'ot_action':action_ot,'ind_action':action_ind,'gap':action_ind-action_ot}


def main()->None:
    local_kl_experiment(); reparameterization_experiment(); empirical_fisher_experiment(); ppo_counterexample(); flow_coupling_experiment()
    print(f'\nSaved plots to {OUT}')

if __name__=='__main__':
    main()
