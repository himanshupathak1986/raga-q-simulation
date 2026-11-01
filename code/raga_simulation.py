"""RAGA-Q synthetic simulation. No Java, LLM, PIT or security tool is executed.
Run: python code/raga_simulation.py --output . --repetitions 30
All outcome, cost and checker behavior are assumptions defined below.
"""
from pathlib import Path
import argparse, json, hashlib, platform
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED=20261028
FEATURES=['z_cc','z_cognitive','z_size','z_mi','z_nesting']
SCENARIOS=['aligned','no_signal','shifted']
SETTINGS=dict(seed=SEED,projects=24,tasks_per_project=5,candidates_per_task=6,
 development_projects=10,calibration_projects=5,test_projects=9,
 repetitions=30,bootstrap_replicates=5000,calibration_risk_slack=0.02,
 calibration_coverage_slack=0.10,noninferiority_margin=0.01,
 cost_units='arbitrary simulated verification units',
 provenance='Fully synthetic; no source-code execution or real model outputs',
 u_detection=[.25,.45,.78,.94],q_detection=[.05,.35,.65,.90],
 false_rejection=[.01,.02,.035,.05],tier_cost=[1.,2.5,7.,18.],
 screening_cost=.8,probe_cost=.6,regularization=.01)

def generate(seed,scenario):
 rng=np.random.default_rng(seed); rows=[]
 # Split assignment precedes features and labels. No shared projects between partitions.
 order=rng.permutation(24); split={int(p):('development' if i<10 else 'calibration' if i<15 else 'test') for i,p in enumerate(order)}
 for project in range(24):
  project_effect=rng.normal(0,.5)
  for task in range(5):
   task_effect=rng.normal(0,.6); structure=expit(rng.normal(-.5,.9))
   exposure=expit(rng.normal(-.6,.9)); privacy=expit(rng.normal(-1.,.9))
   category=['functional','quality','security','privacy'][(project*5+task)%4]
   for generator in range(3):
    for repeat in range(2):
     a=np.clip(structure+rng.normal(0,.16)+.03*generator,0,1)
     hidden=rng.normal(0,.8)+project_effect+task_effect
     vals=np.clip(a+rng.normal(0,.17,5),0,1)
     s=np.clip(exposure+rng.normal(0,.1),0,1);p=np.clip(privacy+rng.normal(0,.1),0,1)
     b=np.clip(.35*a+.35*s+rng.normal(.15,.1),0,1)
     c=float(rng.choice([0,.5,1],p=[.55,.30,.15]))
     probe=np.clip(.25+.30*s+.20*expit(hidden)+rng.normal(0,.17),0,1)
     # Shift changes only the quality-outcome relationship on held-out projects.
     beta=5. if scenario=='aligned' else 0. if scenario=='no_signal' else (-3. if split[project]=='test' else 5.)
     qprob=expit(-3.4+beta*a+.65*hidden)
     uprob=expit(-3.+1.7*s+1.3*p+.65*hidden)
     q=int(rng.random()<qprob);u=int(rng.random()<uprob)
     valid=int(rng.random()<expit(2.8-.4*a-.2*generator))
     override=int(s>.82 or p>.82 or c==1)
     row=dict(synthetic=True,seed=seed,scenario=scenario,project_id=f'SYN_P{project:02d}',
       task_id=f'SYN_P{project:02d}_T{task:02d}',candidate_id=f'SYN_P{project:02d}_T{task:02d}_G{generator}_R{repeat}',
       split=split[project],category=category,generator_proxy=f'Proxy_{chr(65+generator)}',repeat=repeat,
       valid=valid,S=s,P=p,B=b,C=c,T=probe,override=override,U=u,Q=q,D=int(u or q),
       detect_u_draw=rng.random(),detect_q_draw=rng.random(),false_reject_draw=rng.random(),
       unavailable_draw=rng.random(),cost_multiplier=float(rng.lognormal(.0,.25)),
       latent_structure=a,latent_hidden=hidden,prob_U=uprob,prob_Q=qprob)
     row.update(dict(zip(FEATURES,vals)));rows.append(row)
 return pd.DataFrame(rows)

def fit_nonnegative(X,y):
 X=np.asarray(X);y=np.asarray(y)
 def loss(t):
  z=t[0]+X@t[1:];p=expit(z)
  value=np.mean(np.logaddexp(0,z)-y*z)+.01*np.sum(t[1:]**2)
  grad=np.r_[np.mean(p-y),X.T@(p-y)/len(y)+.02*t[1:]]
  return value,grad
 result=minimize(loss,np.zeros(X.shape[1]+1),jac=True,method='L-BFGS-B',bounds=[(None,None)]+[(0,None)]*X.shape[1],options={'ftol':1e-12,'maxiter':1000})
 if not result.success:raise RuntimeError(result.message)
 return result.x

def predict(X,coef):return expit(coef[0]+np.asarray(X)@coef[1:])

def outcome(df,tier,probe=True):
 tier=np.asarray(tier,dtype=int);j=tier-1
 adverse=df.D.to_numpy().astype(bool)
 detect=((df.U.to_numpy()==1)&(df.detect_u_draw.to_numpy()<np.array(SETTINGS['u_detection'])[j]))|((df.Q.to_numpy()==1)&(df.detect_q_draw.to_numpy()<np.array(SETTINGS['q_detection'])[j]))
 false=(~adverse)&(df.false_reject_draw.to_numpy()<np.array(SETTINGS['false_rejection'])[j])
 reject=detect|false
 defer=(~reject)&(tier==4)&(df.unavailable_draw.to_numpy()<.02)
 accept=(~reject)&(~defer)
 screening=.8+(.6 if probe else 0.)
 cost=screening+np.array(SETTINGS['tier_cost'])[j]*df.cost_multiplier.to_numpy()
 return pd.DataFrame(dict(candidate_id=df.candidate_id.to_numpy(),project_id=df.project_id.to_numpy(),tier=tier,
 accept=accept.astype(int),reject=reject.astype(int),defer=defer.astype(int),
 adverse=adverse.astype(int),accepted_adverse=(accept&adverse).astype(int),
 false_reject=false.astype(int),cost_units=cost))

def summary(r):
 n=len(r);a=int(r.accept.sum());d=int(r.adverse.sum());m=int(r.accepted_adverse.sum())
 return dict(n=n,accepted=a,rejected=int(r.reject.sum()),deferred=int(r.defer.sum()),adverse=d,
 accepted_adverse=m,risk=m/n,coverage=a/n,conditional_risk=m/a if a else np.nan,
 false_reject_rate=r.false_reject.sum()/(n-d) if n>d else np.nan,
 mean_cost=r.cost_units.mean(),mean_tier=r.tier.mean())

def routed(score,cut,override,levels=4):
 if cut is None:return np.full(len(score),4)
 thresholds=cut
 if levels==2:t=np.where(score>=thresholds[0],4,1)
 else:t=1+(np.asarray(score)[:,None]>=np.array(thresholds)[None,:]).sum(axis=1)
 return np.where(np.asarray(override)==1,4,t).astype(int)

def select_policy(cal,score,probe=True,levels=4):
 full=summary(outcome(cal,np.full(len(cal),4),probe=probe));options=[]
 quantiles=[np.clip(np.array([.2,.5,.8])+v,.01,.99) for v in [-.2,-.1,0,.1,.2]] if levels==4 else [np.array([v]) for v in [.2,.4,.6,.8]]
 for qs in quantiles:
  cut=np.quantile(score,qs).tolist();t=routed(score,cut,cal.override,levels)
  st=summary(outcome(cal,t,probe))
  if st['risk']<=full['risk']+.02 and st['coverage']>=full['coverage']-.10:options.append((st['mean_cost'],-st['coverage'],cut))
 if not options:return None
 return min(options,key=lambda v:(v[0],v[1]))[2]

def experiment(seed,scenario):
 df=generate(seed,scenario);v=df[df.valid==1].copy();dev=v[v.split=='development'];cal=v[v.split=='calibration'];test=v[v.split=='test']
 aq=fit_nonnegative(dev[FEATURES],dev.Q);weights=aq[1:]/aq[1:].sum() if aq[1:].sum()>1e-10 else np.ones(5)/5
 v['AQRI']=v[FEATURES].to_numpy()@weights
 dev=v[v.split=='development'];cal=v[v.split=='calibration'];test=v[v.split=='test']
 policies={};parameters={'aqri_weights':weights.tolist(),'aqri_fit':aq.tolist(),'models':{},'thresholds':{}}
 for name,cols,probe,levels in [('RAGA-Q',['AQRI','T','S','P','B','C'],True,4),('No AQRI',['T','S','P','B','C'],True,4),('No probe',['AQRI','S','P','B','C'],False,4),('Two tiers',['AQRI','T','S','P','B','C'],True,2),('Size heuristic',['z_size'],True,4)]:
  coef=fit_nonnegative(dev[cols],dev.D);cs=predict(cal[cols],coef);ts=predict(test[cols],coef)
  cuts=select_policy(cal,cs,probe,levels);tier=routed(ts,cuts,test.override,levels)
  policies[name]=outcome(test,tier,probe);parameters['models'][name]=dict(features=cols,coefficients=coef.tolist());parameters['thresholds'][name]=cuts
 policies['CI only']=outcome(test,np.ones(len(test),int));policies['Full']=outcome(test,np.full(len(test),4),probe=False)
 # CI has no metric/probe screening: cost only the base package.
 policies['CI only']['cost_units']=test.cost_multiplier.to_numpy()
 # Full needs no routing probe. Adaptive fallback retains its extra probe cost.
 rng=np.random.default_rng(seed+99)
 rt=policies['RAGA-Q'].tier.to_numpy().copy();noncritical=test.override.to_numpy()==0;rt[noncritical]=rng.permutation(rt[noncritical])
 policies['Random tiers']=outcome(test,rt) # same realized tier counts, not exact cost budget
 metrics=[]
 for name,r in policies.items():metrics.append(dict(seed=seed,scenario=scenario,policy=name,fallback_to_full=(name in parameters['thresholds'] and parameters['thresholds'][name] is None),**summary(r)))
 fullcost=summary(policies['Full'])['mean_cost']
 for r in metrics:r['cost_saving']=1-r['mean_cost']/fullcost
 # Attach scores to all candidates by split-independent prediction, preserving invalid rows as missing.
 df=df.merge(v[['candidate_id','AQRI']],on='candidate_id',how='left')
 cols=parameters['models']['RAGA-Q']['features'];coef=parameters['models']['RAGA-Q']['coefficients']
 scored=v.copy();scored['score']=predict(scored[cols],np.array(coef))
 df=df.merge(scored[['candidate_id','score']],on='candidate_id',how='left')
 discrimination=dict(auroc=float(roc_auc_score(test.D,predict(test[cols],np.array(coef)))),
 ap=float(average_precision_score(test.D,predict(test[cols],np.array(coef)))))
 # Descriptive test-set sensitivity curve; it is never used to select the frozen policy.
 curves=[]
 for shift in np.linspace(-.25,.25,21):
  cut=np.quantile(predict(cal[cols],np.array(coef)),np.clip(np.array([.2,.5,.8])+shift,.01,.99))
  st=summary(outcome(test,routed(predict(test[cols],np.array(coef)),cut,test.override)))
  curves.append(dict(shift=shift,**st))
 return df,pd.DataFrame(metrics),policies,parameters,discrimination,pd.DataFrame(curves)

def bootstrap(policies,reps=5000):
 rng=np.random.default_rng(SEED+500);r=policies['RAGA-Q'];f=policies['Full'];groups=r.project_id.unique();rows=[]
 for g in groups:
  a=r[r.project_id==g];b=f[f.project_id==g];rows.append([len(a),a.accepted_adverse.sum()-b.accepted_adverse.sum(),a.accept.sum()-b.accept.sum(),a.cost_units.sum(),b.cost_units.sum()])
 ar=np.array(rows);res=[]
 for _ in range(reps):
  v=ar[rng.integers(0,len(ar),len(ar))].sum(axis=0)
  res.append([v[1]/v[0],v[2]/v[0],1-v[3]/v[4]])
 q=np.quantile(res,[.025,.05,.95,.975],axis=0)
 return dict(groups=len(groups),reps=reps,risk_difference_ci95=q[[0,3],0].tolist(),risk_difference_upper95=float(q[2,0]),coverage_difference_ci95=q[[0,3],1].tolist(),cost_saving_ci95=q[[0,3],2].tolist(),numerical_margin_rule_passes=bool(q[2,0]<.01),identical_acceptance_decisions=bool((r.accept.to_numpy()==f.accept.to_numpy()).all()),noninferiority_established=False,warning='Percentile project-cluster bootstrap is descriptive with only nine projects; no real-world noninferiority claim.')

def figures(root,primary,rep,curves,df,policies):
 plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220})
 order=['CI only','Size heuristic','Random tiers','No AQRI','No probe','Two tiers','RAGA-Q','Full']
 z=primary.set_index('policy').loc[order];colors=['#9d9d9d']*6+['#155b8d','#333333']
 fig,ax=plt.subplots(1,2,figsize=(5,2.0),layout='constrained')
 ax[0].barh(order,z.risk*100,color=colors);ax[0].invert_yaxis();ax[0].set_xlabel('Accepted adverse / eligible (%)')
 ax[1].barh(order,z.mean_cost,color=colors);ax[1].invert_yaxis();ax[1].set_yticklabels([]);ax[1].set_xlabel('Mean simulated cost units')
 fig.suptitle('Synthetic aligned scenario · fixed primary seed');fig.savefig(root/'figures/policy_comparison.png');fig.savefig(root/'figures/policy_comparison.pdf');plt.close(fig)
 fig,ax=plt.subplots(1,2,figsize=(5,2.2),layout='constrained')
 ax[0].plot(curves.coverage*100,curves.risk*100,'o-',color='#155b8d',ms=3);ax[0].set_xlabel('Acceptance coverage (%)');ax[0].set_ylabel('Accepted adverse / eligible (%)')
 for name in ['CI only','RAGA-Q','Full']:
  r=primary[primary.policy==name].iloc[0];ax[0].scatter(r.coverage*100,r.risk*100,s=30,label=name)
 ax[0].legend(fontsize=8)
 counts=policies['RAGA-Q'].tier.value_counts().reindex([1,2,3,4],fill_value=0)
 ax[1].bar(['T1','T2','T3','T4'],counts,color='#3c799c');ax[1].set_ylabel('Held-out candidates')
 fig.suptitle('Synthetic policy sensitivity and chosen tier distribution');fig.savefig(root/'figures/coverage_tiers.png');fig.savefig(root/'figures/coverage_tiers.pdf');plt.close(fig)
 fig,ax=plt.subplots(1,2,figsize=(5,2.2),layout='constrained')
 for i,sc in enumerate(SCENARIOS):
  sub=rep[(rep.scenario==sc)&(rep.policy=='RAGA-Q')]
  vals=sub.cost_saving*100;rr=(sub.risk.to_numpy()-rep[(rep.scenario==sc)&(rep.policy=='Full')].risk.to_numpy())*100
  for a,data in zip(ax,[vals,rr]):
   lo,hi=np.quantile(data,[.025,.975]);a.errorbar(i,np.mean(data),yerr=[[np.mean(data)-lo],[hi-np.mean(data)]],fmt='o',capsize=5,color='#155b8d')
 for a in ax:a.set_xticks(range(3),SCENARIOS);a.axhline(0,color='grey',lw=.7)
 ax[0].set_ylabel('Cost saving vs full (%)');ax[1].set_ylabel('Adverse risk difference (pp)')
 fig.suptitle('30 synthetic repetitions\nMean and 2.5–97.5% simulation range');fig.savefig(root/'figures/scenario_sensitivity.png');fig.savefig(root/'figures/scenario_sensitivity.pdf');plt.close(fig)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path('.'));ap.add_argument('--repetitions',type=int,default=30);args=ap.parse_args();root=args.output
 for sub in ['data','results','figures']:(root/sub).mkdir(parents=True,exist_ok=True)
 dictionary = {'synthetic': ('boolean', 'Always true; all observations are simulated.'),
  'seed': ('integer', 'NumPy generator seed identifying a complete simulation run.'),
  'scenario': ('category',
               'aligned, no_signal, or shifted; see README and generation equations.'),
  'project_id': ('identifier',
                 'Fictional project ID; independent grouping unit for bootstrap.'),
  'task_id': ('identifier', 'Fictional task nested within project; six candidates per task.'),
  'candidate_id': ('identifier', 'Unique within seed and scenario, not across repeated runs.'),
  'split': ('category', 'development, calibration, or test; assigned at project level.'),
  'category': ('category',
               'Balanced synthetic tag: functional, quality, security or privacy.'),
  'generator_proxy': ('category', 'Fictional Proxy_A/B/C; no actual LLM execution.'),
  'repeat': ('integer', '0 or 1; repeat per fictional generator/task.'),
  'valid': ('binary',
            'Simulated applicability/build validity, 1 eligible for policy analysis.'),
  'S': ('[0,1]', 'Simulated security-exposure feature.'),
  'P': ('[0,1]', 'Simulated privacy-exposure feature.'),
  'B': ('[0,1]', 'Simulated dependency/blast-radius feature.'),
  'C': ('{0,0.5,1}', 'Simulated component criticality.'),
  'T': ('[0,1]', 'Simulated bounded-probe inadequacy feature; no real mutants executed.'),
  'override': ('binary', 'Force T4 when S>0.82 or P>0.82 or C=1.'),
  'U': ('binary', 'Simulated unsafe-outcome label, sampled from prob_U.'),
  'Q': ('binary', 'Simulated severe-quality label, sampled from prob_Q.'),
  'D': ('binary', 'Union of U and Q.'),
  'detect_u_draw': ('[0,1]',
                    'Shared uniform draw used against assumed unsafe-detection '
                    'sensitivities.'),
  'detect_q_draw': ('[0,1]',
                    'Shared uniform draw used against assumed quality-detection '
                    'sensitivities.'),
  'false_reject_draw': ('[0,1]', 'Shared uniform draw for false rejection among D=0.'),
  'unavailable_draw': ('[0,1]',
                       'T4 review-unavailability draw; values below 0.02 defer unless already '
                       'rejected.'),
  'cost_multiplier': ('unitless',
                      'Lognormal factor multiplying assumed package costs; not observed '
                      'runtime.'),
  'latent_structure': ('[0,1]',
                       'Simulator-only structural latent variable; never fed directly to '
                       'fitted policy.'),
  'latent_hidden': ('real',
                    'Unobserved simulator risk including project/task effects; not a model '
                    'input.'),
  'prob_U': ('[0,1]', 'Simulator-only probability of unsafe outcome; not a model input.'),
  'prob_Q': ('[0,1]',
             'Simulator-only probability of severe-quality outcome; not a model input.'),
  'z_cc': ('[0,1]',
           'Noisy structural proxy named after cyclomatic degradation; not a Java metric.'),
  'z_cognitive': ('[0,1]',
                  'Noisy structural proxy named after cognitive complexity degradation.'),
  'z_size': ('[0,1]', 'Noisy structural size proxy; not measured changed LOC.'),
  'z_mi': ('[0,1]', 'Noisy proxy for worsening maintainability; MI formula is not executed.'),
  'z_nesting': ('[0,1]', 'Noisy proxy for increased nesting.'),
  'AQRI': ('[0,1]',
           'Weighted structural proxy score fitted on valid development data; blank for '
           'invalid rows.'),
  'score': ('[0,1]',
            'Constrained logistic predicted synthetic adverse risk; blank for invalid rows.')}
 pd.DataFrame([dict(column=k,units=u,description=t) for k,(u,t) in dictionary.items()]).to_csv(root/'DATA_DICTIONARY.csv',index=False)
 config={**SETTINGS,'repetitions':args.repetitions};(root/'config.json').write_text(json.dumps(config,indent=2))
 allmetrics=[];alldata=[];primary=None
 for sc in SCENARIOS:
  for i in range(args.repetitions):
   result=experiment(SEED+i,sc);df,m,pol,params,disc,curves=result;allmetrics.append(m);alldata.append(df)
   if sc=='aligned' and i==0:primary=result
  print('Completed',sc,flush=True)
 df,m,pol,params,disc,curves=primary
 df.to_csv(root/'data/synthetic_primary.csv',index=False,float_format='%.10g')
 pd.concat(alldata,ignore_index=True).to_csv(root/'data/synthetic_all_scenarios.csv',index=False,float_format='%.10g')
 rep=pd.concat(allmetrics,ignore_index=True);rep.to_csv(root/'results/repetition_results.csv',index=False)
 m.to_csv(root/'results/primary_policy_results.csv',index=False)
 pd.concat([r.assign(policy=name) for name,r in pol.items()],ignore_index=True).to_csv(root/'results/primary_decisions.csv',index=False)
 curves.to_csv(root/'results/primary_sensitivity_curve.csv',index=False)
 (root/'results/fitted_parameters.json').write_text(json.dumps(params,indent=2))
 ci=bootstrap(pol);(root/'results/primary_inference.json').write_text(json.dumps({**ci,**disc},indent=2))
 agg=rep.groupby(['scenario','policy']).agg(risk_mean=('risk','mean'),risk_sd=('risk','std'),cost_saving_mean=('cost_saving','mean'),coverage_mean=('coverage','mean')).reset_index();agg.to_csv(root/'results/scenario_summary.csv',index=False)
 figures(root,m,rep,curves,df,pol)
 # Core verification: pairing, split isolation, monotonic checkers, count identity and declared size.
 assert len(df)==720 and df.candidate_id.is_unique
 assert df.groupby('project_id').split.nunique().max()==1
 assert all((r.accept+r.reject+r.defer==1).all() for r in pol.values())
 assert all(r.candidate_id.tolist()==pol['Full'].candidate_id.tolist() for r in pol.values())
 assert (pol['Full'].accepted_adverse<=pol['RAGA-Q'].accepted_adverse).all()
 expected=720*3*args.repetitions;assert sum(len(x) for x in alldata)==expected
 env=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__)
 (root/'results/validation.json').write_text(json.dumps(dict(passed=True,primary_attempts=720,total_synthetic_rows=expected,paired_candidates=len(pol['Full']),environment=env),indent=2))
 print(m.to_string(index=False));print(json.dumps(ci,indent=2))
 if args.repetitions==30:
  (root/'results/checksums.json').write_text(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'data').glob('*.csv'))},indent=2))
if __name__=='__main__':main()
