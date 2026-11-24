"""Reproducible ensemble homework experiments. Only student inputs are read.

python run_experiments.py --dataset wdbc --input-root data --output-root results
python run_experiments.py --dataset ctg  --input-root data --output-root results
"""
from pathlib import Path
import argparse, json, platform, time, warnings
import numpy as np
import pandas as pd
import scipy, sklearn
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, log_loss, brier_score_loss, confusion_matrix
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text

SEED=1010
MODELS=['logistic','rbf_svm','single_tree','deep_tree','bagging','random_forest','adaboost','gradient_boosting']
DATASETS={'wdbc':('wdbc','malignant',None),'ctg':('ctg','pathologic','recording_id')}

def models():
    return {
      'logistic':make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=5000,random_state=SEED)),
      'rbf_svm':make_pipeline(StandardScaler(),SVC(C=1,kernel='rbf',gamma='scale',probability=False,random_state=SEED)),
      'single_tree':DecisionTreeClassifier(max_depth=4,random_state=SEED),
      'deep_tree':DecisionTreeClassifier(max_depth=None,random_state=SEED),
      'bagging':BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=None,random_state=SEED),n_estimators=100,max_features=1.,bootstrap=True,random_state=SEED,n_jobs=1),
      'random_forest':RandomForestClassifier(n_estimators=100,max_features='sqrt',bootstrap=True,random_state=SEED,n_jobs=1),
      'adaboost':AdaBoostClassifier(n_estimators=50,learning_rate=1.,random_state=SEED),
      'gradient_boosting':GradientBoostingClassifier(n_estimators=100,learning_rate=.1,max_depth=3,random_state=SEED)
    }

def output_prediction(model,X):
    pred=model.predict(X)
    if hasattr(model,'predict_proba'):
        prob=model.predict_proba(X)[:,1]
        return prob,prob,pred
    return model.decision_function(X),np.full(len(X),np.nan),pred

def metrics(y,score,prob,pred):
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    r={'roc_auc':roc_auc_score(y,score),'average_precision':average_precision_score(y,score),
       'balanced_accuracy':balanced_accuracy_score(y,pred),'sensitivity':tp/(tp+fn),'specificity':tn/(tn+fp),
       'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}
    r['log_loss']=log_loss(y,prob,labels=[0,1]) if np.isfinite(prob).all() else np.nan
    r['brier']=brier_score_loss(y,prob) if np.isfinite(prob).all() else np.nan
    return r

def write_json(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=lambda x:x.item() if hasattr(x,'item') else str(x)))

def boot_indices(y,groups,rng):
    if groups is None:
        ix=rng.choice(len(y),size=len(y),replace=True)
    else:
        unique=np.unique(groups)
        picked=rng.choice(unique,size=len(unique),replace=True)
        ix=np.concatenate([np.flatnonzero(groups==g) for g in picked])
    return ix

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',choices=DATASETS,required=True)
    ap.add_argument('--input-root',type=Path,default=Path(__file__).resolve().parents[1] / 'data')
    ap.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parents[1] / 'results')
    args=ap.parse_args(); start=time.time()
    out=args.output_root/args.dataset; out.mkdir(parents=True,exist_ok=True)
    folder,label,group=DATASETS[args.dataset]
    base=args.input_root/folder/'student'
    train=pd.read_csv(base/'train.csv'); valid=pd.read_csv(base/'validation.csv')
    # Test labels are never opened; test features are inspected only to audit split IDs.
    test=pd.read_csv(base/'locked-test-features.csv')
    feats=[x for x in train if x not in {'sample_id','recording_id',label}]
    assert label not in test and not ({'sample_id','recording_id',label}&set(feats))
    assert set(train.sample_id).isdisjoint(valid.sample_id)
    assert set(train.sample_id).isdisjoint(test.sample_id)
    assert set(valid.sample_id).isdisjoint(test.sample_id)
    if group:
        assert set(train[group]).isdisjoint(valid[group])
        assert set(train[group]).isdisjoint(test[group])
        assert set(valid[group]).isdisjoint(test[group])
    X=train[feats]; y=train[label].to_numpy(); Xv=valid[feats]; yv=valid[label].to_numpy()
    assert np.isfinite(X.to_numpy()).all() and np.isfinite(Xv.to_numpy()).all()
    dev=pd.concat([train,valid],ignore_index=True); Xd=dev[feats]; yd=dev[label].to_numpy()
    groups=train[group].to_numpy() if group else None
    vgroups=valid[group].to_numpy() if group else None
    dgroups=dev[group].to_numpy() if group else None
    proto={
       'dataset':args.dataset,'seed':SEED,'label':label,'positive_class':1,'features':feats,'group':group,
       'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},
       'fixed_models':{name:m.get_params(deep=True) for name,m in models().items()},
       'comparison':'Fixed original train/validation; shared five development CV folds on train+validation, not an independent replication.',
       'cv':'StratifiedGroupKFold(5,shuffle=True,random_state=1010)' if group else 'StratifiedKFold(5,shuffle=True,random_state=1010)',
       'test_policy':'Locked test labels never read, no locked test metrics or model-selection claims.',
       'svm':'Uncalibrated decision_function for ROC-AUC/AP; predict at decision 0 for BA; probability metrics omitted.',
       'thresholds':'Estimator predict; generally p>0.5, ties resolved by estimator. SVM margin 0.',
       'depth_grid':[1,2,3,4,None],'tree_counts':[1,10,50,100,300],
       'boosting_grid':{'learning_rate':[.03,.1,.3],'stages':300,'max_depth':3},
       'stability':{'refits':30,'shared_samples':True,'model_seed_fixed':True,'unit':'recording cluster' if group else 'row',
           'probability_sd':'Per validation sample SD across 30 refits, ddof=1; mean/median across samples.',
           'flip_rate':'Mean of 30 x N indicators that resampled-fit class differs from full-train fitted class.',
           'any_change':'Fraction of validation samples with both classes among 30 refits.'},
       'metric_bootstrap':{'draws':2000,'unit':'recording cluster' if group else 'row','paired':True,'interval':'percentile 2.5/97.5; fixed fitted models and validation only; invalid one-class draws skipped'},
       'importance':'RF100 MDI and validation permutation ROC-AUC decrease, 30 repeats; CTG permutes rows and is descriptive, not an independent-group inferential test.',
       'selection':'No data-driven refitting of selected hyperparameters; sensitivity grids descriptive and use validation repeatedly.'
    }
    write_json(out/'protocol.json',proto)
    audit=[]
    for split,d in [('train',train),('validation',valid),('locked_test_features',test)]:
        audit.append({'split':split,'n':len(d),'n_features':len(feats),'n_positive':int(d[label].sum()) if label in d else None,
             'prevalence':float(d[label].mean()) if label in d else None,'n_groups':d[group].nunique() if group else None,
             'missing_cells':int(d[feats].isna().sum().sum()),'duplicate_feature_rows':int(d[feats].duplicated().sum())})
    pd.DataFrame(audit).to_csv(out/'data-audit.csv',index=False)
    # Baseline fits.
    fitted={}; predictions={}; comparison=[]; predframe=valid[[x for x in ['sample_id',group,label] if x is not None]].copy()
    for name,m in models().items():
        t=time.time(); m.fit(X,y); fitted[name]=m
        s,p,c=output_prediction(m,Xv); predictions[name]=(s,p,c)
        comparison.append({'model':name,**metrics(yv,s,p,c),'fit_seconds':time.time()-t})
        predframe[name+'_score']=s; predframe[name+'_probability']=p; predframe[name+'_class']=c
    pd.DataFrame(comparison).to_csv(out/'validation-metrics.csv',index=False)
    predframe.to_csv(out/'validation-predictions.csv',index=False)
    print(args.dataset,'baseline complete',flush=True)
    # Shared CV folds and all OOF scores.
    splitter=(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED) if group else StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED))
    folds=list(splitter.split(Xd,yd,dgroups)); cvrows=[]
    oof=dev[[x for x in ['sample_id',group,label] if x is not None]].copy(); oof['fold']=-1
    for name in MODELS:
        oof[name+'_score']=np.nan; oof[name+'_probability']=np.nan; oof[name+'_class']=np.nan
    for fold,(it,iv) in enumerate(folds,1):
        if group: assert set(dgroups[it]).isdisjoint(dgroups[iv])
        oof.loc[iv,'fold']=fold
        for name,m in models().items():
            m.fit(Xd.iloc[it],yd[it]); s,p,c=output_prediction(m,Xd.iloc[iv])
            cvrows.append({'model':name,'fold':fold,'n_train':len(it),'n_validation':len(iv),'n_positive':int(yd[iv].sum()),**metrics(yd[iv],s,p,c)})
            oof.loc[iv,name+'_score']=s; oof.loc[iv,name+'_probability']=p; oof.loc[iv,name+'_class']=c
    cvdf=pd.DataFrame(cvrows); cvdf.to_csv(out/'cv-fold-metrics.csv',index=False)
    cvsum=cvdf.groupby('model')[['roc_auc','average_precision','balanced_accuracy']].agg(['mean','std']).reset_index()
    cvsum.columns=['_'.join(c).strip('_').replace('_std','_sd') for c in cvsum.columns]; cvsum.to_csv(out/'cv-summary.csv',index=False)
    oof.to_csv(out/'oof-predictions.csv',index=False)
    print(args.dataset,'CV complete',flush=True)
    # Depth and root perturbation.
    rows=[]
    for depth in [1,2,3,4,None]:
        m=DecisionTreeClassifier(max_depth=depth,random_state=SEED).fit(X,y)
        for split,xx,yy in [('train',X,y),('validation',Xv,yv)]:
            s,p,c=output_prediction(m,xx)
            rows.append({'max_depth':'unlimited' if depth is None else str(depth),'actual_depth':m.get_depth(),'n_leaves':m.get_n_leaves(),'split':split,**metrics(yy,s,p,c)})
    pd.DataFrame(rows).to_csv(out/'tree-depth.csv',index=False)
    shallow=DecisionTreeClassifier(max_depth=2,random_state=SEED).fit(X,y)
    (out/'depth2-tree.txt').write_text(export_text(shallow,feature_names=feats,decimals=5))
    rootrows=[]; rng=np.random.default_rng(SEED)
    shared=[boot_indices(y,groups,rng) for _ in range(30)]
    np.savez_compressed(out/'training-bootstrap-indices.npz',**{f'refit_{b+1}':ix for b,ix in enumerate(shared)})
    for b,ix in enumerate(shared[:10],1):
        m=DecisionTreeClassifier(max_depth=2,random_state=SEED).fit(X.iloc[ix],y[ix])
        rootrows.append({'refit':b,'feature':feats[m.tree_.feature[0]],'threshold':m.tree_.threshold[0], 'n_resampled_rows':len(ix),'n_unique_rows':len(np.unique(ix))})
    pd.DataFrame(rootrows).to_csv(out/'bootstrap-roots.csv',index=False)
    # Actual finite-n bootstrap sampling experiment.
    sample_rows=[]; sample_summary=[]
    for n in [5,len(y)]:
        rng=np.random.default_rng(SEED+n)
        for b in range(1000):
            ix=rng.choice(n,size=n,replace=True); unique=len(np.unique(ix))
            sample_rows.append({'n':n,'draw':b+1,'unique_fraction':unique/n,'oob_fraction':1-unique/n})
        sel=[r['unique_fraction'] for r in sample_rows if r['n']==n]
        sample_summary.append({'n':n,'mean_unique_fraction':np.mean(sel),'sd_unique_fraction':np.std(sel,ddof=1),'mean_oob_fraction':1-np.mean(sel),'theory_unique_fraction':1-(1-1/n)**n,'theory_oob_fraction':(1-1/n)**n})
    pd.DataFrame(sample_rows).to_csv(out/'bootstrap-simulation.csv',index=False)
    pd.DataFrame(sample_summary).to_csv(out/'bootstrap-simulation-summary.csv',index=False)
    # Outer-bootstrap training perturbation; fixed inner model seeds.
    stability_rows=[]; per_sample=valid[[x for x in ['sample_id',group,label] if x is not None]].copy(); cubes={}
    for name in ['single_tree','deep_tree','bagging','random_forest','gradient_boosting','logistic']:
        probs=[]; classes=[]
        for ix in shared:
            m=clone(models()[name]).fit(X.iloc[ix],y[ix]); _,p,c=output_prediction(m,Xv)
            probs.append(p); classes.append(c)
        probs=np.array(probs); classes=np.array(classes); sd=probs.std(axis=0,ddof=1)
        flip=(classes!=predictions[name][2][None,:]).mean(axis=0)
        any_change=classes.min(axis=0)!=classes.max(axis=0)
        stability_rows.append({'model':name,'mean_prediction_sd':sd.mean(),'median_prediction_sd':np.median(sd),'max_prediction_sd':sd.max(),
              'mean_class_flip_vs_full_train':flip.mean(),'fraction_samples_any_class_change':any_change.mean(),'n_refits':len(shared)})
        per_sample[name+'_prob_sd']=sd; per_sample[name+'_flip_vs_full_train']=flip; per_sample[name+'_any_change']=any_change
        cubes[name+'_probability']=probs; cubes[name+'_class']=classes
    pd.DataFrame(stability_rows).to_csv(out/'stability-summary.csv',index=False)
    per_sample.to_csv(out/'stability-per-sample.csv',index=False); np.savez_compressed(out/'stability-predictions.npz',**cubes)
    print(args.dataset,'stability complete',flush=True)
    # Number of trees, OOB and per-tree error correlation.
    countrows=[]; tree_cubes={}
    for count in [1,10,50,100,300]:
        for name in ['bagging','random_forest']:
            m=clone(models()[name]).set_params(n_estimators=count,oob_score=(count>=50)).fit(X,y)
            s,p,c=output_prediction(m,Xv)
            row={'model':name,'n_trees':count,**metrics(yv,s,p,c),'oob_accuracy':m.oob_score_ if count>=50 else np.nan,
                 'oob_valid_for_group_generalization':False if group else True}
            if count==100:
                xx=Xv.to_numpy()
                if name=='bagging':
                    tree_prob=np.array([t.predict_proba(xx[:,fi])[:,1] for t,fi in zip(m.estimators_,m.estimators_features_)])
                    tree_pred=np.array([t.predict(xx[:,fi]) for t,fi in zip(m.estimators_,m.estimators_features_)])
                else:
                    tree_prob=np.array([t.predict_proba(xx)[:,1] for t in m.estimators_])
                    tree_pred=np.array([t.predict(xx) for t in m.estimators_])
                err=(tree_pred!=yv).astype(float)
                ok=err.std(axis=1)>0; cor=np.corrcoef(err[ok]); vals=cor[np.triu_indices(len(cor),1)]
                row['mean_pairwise_error_correlation']=vals.mean(); row['n_nonconstant_error_trees']=int(ok.sum()); row['mean_individual_tree_accuracy']=float((tree_pred==yv).mean())
                res=tree_prob-yv[None,:]; rc=np.corrcoef(res); row['mean_pairwise_probability_residual_correlation']=rc[np.triu_indices(len(rc),1)].mean()
                tree_cubes[name+'_probability']=tree_prob; tree_cubes[name+'_error']=err
            countrows.append(row)
    pd.DataFrame(countrows).to_csv(out/'tree-count.csv',index=False); np.savez_compressed(out/'individual-tree-predictions.npz',**tree_cubes)
    # Boosting staged train/validation losses.
    rows=[]
    for rate in [.03,.1,.3]:
        m=GradientBoostingClassifier(n_estimators=300,learning_rate=rate,max_depth=3,random_state=SEED).fit(X,y)
        for stage,(pt,pv) in enumerate(zip(m.staged_predict_proba(X),m.staged_predict_proba(Xv)),1):
            for split,yy,pp in [('train',y,pt[:,1]),('validation',yv,pv[:,1])]:
                rows.append({'learning_rate':rate,'stage':stage,'split':split,'log_loss':log_loss(yy,pp),'roc_auc':roc_auc_score(yy,pp),'average_precision':average_precision_score(yy,pp)})
    pd.DataFrame(rows).to_csv(out/'boosting-stages.csv',index=False)
    # MDI + permutation and feature correlation.
    rf=fitted['random_forest']
    pim=permutation_importance(rf,Xv,yv,n_repeats=30,random_state=SEED,scoring='roc_auc',n_jobs=1)
    imp=pd.DataFrame({'feature':feats,'mdi':rf.feature_importances_,'permutation_auc_drop_mean':pim.importances_mean,'permutation_auc_drop_sd':pim.importances.std(axis=1,ddof=1)})
    imp.to_csv(out/'feature-importance.csv',index=False)
    pd.DataFrame(pim.importances,index=feats,columns=[f'repeat_{i+1}' for i in range(30)]).to_csv(out/'permutation-repeats.csv',index_label='feature')
    X.corr(method='spearman').to_csv(out/'train-spearman.csv',index_label='feature')
    X.corr(method='pearson').to_csv(out/'train-pearson.csv',index_label='feature')
    print(args.dataset,'curves and importance complete',flush=True)
    # Paired bootstrap CIs of fixed-model validation metrics.
    rng=np.random.default_rng(SEED+2000); metric_draws=[]; valid_draws=0
    names=['random_forest','single_tree','bagging','logistic','gradient_boosting']
    for b in range(2000):
        ix=boot_indices(yv,vgroups,rng)
        if len(np.unique(yv[ix]))<2: continue
        valid_draws+=1
        for name in names:
            ss=predictions[name][0][ix]
            metric_draws.append({'draw':b+1,'model':name,'n_resampled_rows':len(ix),'roc_auc':roc_auc_score(yv[ix],ss),'average_precision':average_precision_score(yv[ix],ss)})
    bd=pd.DataFrame(metric_draws); bd.to_csv(out/'validation-bootstrap-draws.csv',index=False)
    interval=[]; diff=[]
    valtab=pd.DataFrame(comparison).set_index('model')
    for name in names:
        d=bd.loc[bd.model==name]
        for metric in ['roc_auc','average_precision']:
            lo,hi=np.quantile(d[metric],[.025,.975])
            interval.append({'model':name,'metric':metric,'estimate':valtab.loc[name,metric],'lower':lo,'upper':hi,'valid_draws':valid_draws,'requested_draws':2000})
    for other in ['single_tree','bagging','logistic','gradient_boosting']:
        for metric in ['roc_auc','average_precision']:
            pivot=bd.pivot(index='draw',columns='model',values=metric)
            delta=pivot.random_forest-pivot[other]; lo,hi=np.quantile(delta,[.025,.975])
            diff.append({'contrast':'random_forest - '+other,'metric':metric,'estimate':valtab.loc['random_forest',metric]-valtab.loc[other,metric],
                         'lower':lo,'upper':hi,'valid_draws':valid_draws,'requested_draws':2000})
    pd.DataFrame(interval).to_csv(out/'validation-bootstrap-intervals.csv',index=False)
    pd.DataFrame(diff).to_csv(out/'validation-paired-differences.csv',index=False)
    # Pre-specified sample summaries useful for plot captions.
    summary={'elapsed_seconds':time.time()-start,'completed':True,'test_labels_read':False,
             'n_correlated_pairs_pearson_abs_gt_0_9':int(np.triu(np.abs(X.corr().to_numpy())>.9,1).sum()),
             'n_correlated_pairs_spearman_abs_gt_0_9':int(np.triu(np.abs(X.corr(method='spearman').to_numpy())>.9,1).sum()),
             'train_groups':int(train[group].nunique()) if group else None,'validation_groups':int(valid[group].nunique()) if group else None,
             'valid_metric_bootstrap_draws':valid_draws,'invalid_metric_bootstrap_draws':2000-valid_draws}
    print(args.dataset,'ALL COMPLETE',summary,flush=True)

if __name__=='__main__': main()
