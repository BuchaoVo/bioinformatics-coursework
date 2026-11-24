#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UCI thyroid0387 (9172 records): fully reproducible medical tabular ML comparison.
Do not claim clinical diagnostic validity. Requirements in environment/requirements.txt.
Usage from the week directory: python code/run_study.py --cv 3 --shap
"""
from __future__ import annotations
import argparse, json, re, warnings, sys, os, urllib.request
from pathlib import Path
from collections import Counter
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, StratifiedGroupKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import clone, BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, accuracy_score,
    balanced_accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, log_loss, brier_score_loss, roc_curve,
    precision_recall_curve, ConfusionMatrixDisplay)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from scipy.stats import bootstrap

ROOT=Path(__file__).resolve().parents[1]
SEED=2026
COLS=['age','sex','on_thyroxine','query_on_thyroxine','on_antithyroid_medication',
'sick','pregnant','thyroid_surgery','I131_treatment','query_hypothyroid',
'query_hyperthyroid','lithium','goitre','tumor','hypopituitary','psych',
'TSH_measured','TSH','T3_measured','T3','TT4_measured','TT4',
'T4U_measured','T4U','FTI_measured','FTI','TBG_measured','TBG','referral_source']
MEASURED=[c for c in COLS if c.endswith('_measured')]
LABS=['TSH','T3','TT4','T4U','FTI','TBG']
NUMS=['age']+LABS
BOOLS=[c for c in COLS if c not in NUMS+['sex','referral_source']]
CATS=['sex','referral_source']
CLASS_GROUPS={'hyperthyroid':'ABCD','hypothyroid':'EFGH','binding_protein':'IJ',
              'non_thyroidal':'K','replacement':'LMN','antithyroid_treatment':'OPQ',
              'miscellaneous':'RST'}
URLS=[
 'https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/thyroid0387.data',
 'https://rioultf.users.greyc.fr/uci/files/thyroid-disease/thyroid0387.data',
 'https://raw.githubusercontent.com/veronica242001/Thyroid_Disease/main/thyroid0387.data']

def download_data(dest:Path):
 dest.parent.mkdir(parents=True,exist_ok=True)
 for u in URLS:
  try:
   with urllib.request.urlopen(u,timeout=35) as r: data=r.read()
   if len(data)>400000 and b'[' in data and data.count(b'\n')>9000:
    dest.write_bytes(data); print('Data downloaded from',u,'bytes',len(data)); return dest
  except Exception as e: print('Could not download',u,'reason:',str(e)[:140])
 raise RuntimeError('Unable to retrieve UCI raw dataset. Manually download thyroid0387.data from the UCI dataset page and pass --data FILE; no results were fabricated.')

def load_raw(path:Path):
 lines=path.read_text(encoding='latin-1').splitlines()
 rows=[]; bad=[]
 for i,line in enumerate(lines,1):
  if not line.strip(): continue
  parts=line.split(',')
  if len(parts)!=30: bad.append((i,len(parts))); continue
  m=re.fullmatch(r'([A-T|\-]+)\[(\d+)\]',parts[-1].strip())
  if not m: bad.append((i,'last-field '+parts[-1][:25])); continue
  rows.append(parts[:29]+[m.group(1),m.group(2)])
 if bad: raise ValueError(f'Malformed input: {len(bad)} rows; first errors: {bad[:8]}')
 df=pd.DataFrame(rows, columns=COLS+['diagnosis','record_id'])
 df[COLS]=df[COLS].replace('?',np.nan)
 for c in NUMS:df[c]=pd.to_numeric(df[c],errors='coerce')
 # QC: age values outside plausible 1-110 years are coding anomalies, not real ages.
 # Convert to missing; impute per training fold downstream (never inspect labels).
 df.loc[~df['age'].between(1,110),'age']=np.nan
 for c in BOOLS:df[c]=df[c].map({'t':1.0,'f':0.0})
 for c in CATS:df[c]=df[c].astype('object').where(df[c].notna(),np.nan)
 df['diagnosis']=df['diagnosis'].astype(str)
 df['target']=(df['diagnosis']!='-').astype(int)
 def groups(d):
  ss=set(d.replace('|','').replace('-',''))
  return [k for k,v in CLASS_GROUPS.items() if ss.intersection(v)]
 df['diagnosis_groups']=df['diagnosis'].map(lambda x:';'.join(groups(x)) or 'none')
 return df

def preprocessor(features):
 nums=[c for c in NUMS if c in features]
 flags=[c for c in BOOLS if c in features]
 cats=[c for c in CATS if c in features]
 parts=[]
 if nums: parts.append(('num',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),nums))
 if flags:parts.append(('flags',SimpleImputer(strategy='most_frequent'),flags))
 if cats:parts.append(('cat',Pipeline([('fill',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cats))
 return ColumnTransformer(parts,sparse_threshold=0)

class CatBoostAdapter(ClassifierMixin, BaseEstimator):
 """Make CatBoost compatible with sklearn 1.8 model selection and Pipeline tags."""
 def __init__(self,iterations=320,depth=5,learning_rate=0.06,l2_leaf_reg=4,thread_count=2,random_seed=SEED):
  self.iterations=iterations;self.depth=depth;self.learning_rate=learning_rate
  self.l2_leaf_reg=l2_leaf_reg;self.thread_count=thread_count;self.random_seed=random_seed
 def fit(self,X,y):
  from catboost import CatBoostClassifier
  self.model_=CatBoostClassifier(iterations=self.iterations,depth=self.depth,learning_rate=self.learning_rate,
      l2_leaf_reg=self.l2_leaf_reg,auto_class_weights='Balanced',verbose=False,
      thread_count=self.thread_count,random_seed=self.random_seed,allow_writing_files=False)
  self.model_.fit(X,y);self.classes_=np.array([0,1]);self.n_features_in_=X.shape[1]
  return self
 def predict_proba(self,X):return self.model_.predict_proba(X)
 def predict(self,X):return self.model_.predict(X).astype(int).ravel()


def grouped_three_way_split(y, sig):
 """Prevent identical full feature vectors spanning train, validation and test."""
 groups=pd.factorize(sig,sort=False)[0]
 idx=np.arange(len(y))
 splitter=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
 candidates=list(splitter.split(idx,y,groups))
 trval,test=min(candidates,key=lambda v: abs(len(v[1])/len(y)-.2)+2*abs(float(y[v[1]].mean())-float(y.mean())))
 split2=StratifiedGroupKFold(n_splits=4,shuffle=True,random_state=SEED+1)
 candidates2=list(split2.split(trval,y[trval],groups[trval]))
 trsub,valsub=min(candidates2,key=lambda v: abs(len(v[1])/len(trval)-.25)+2*abs(float(y[trval[v[1]]].mean())-float(y[trval].mean())))
 train=trval[trsub];val=trval[valsub]
 assert set(groups[train]).isdisjoint(set(groups[val])) and set(groups[train]).isdisjoint(set(groups[test])) and set(groups[val]).isdisjoint(set(groups[test]))
 return train,val,test,groups

def estimator_fns(y_train):
 neg=(y_train==0).sum(); pos=(y_train==1).sum()
 weight=max(1.,neg/max(pos,1))
 d={
  'GaussianNB': lambda:GaussianNB(),
  'LogisticRegression':lambda:LogisticRegression(max_iter=1600,class_weight='balanced',random_state=SEED),
  'RandomForest':lambda:RandomForestClassifier(n_estimators=220,max_depth=12,min_samples_leaf=3,class_weight='balanced_subsample',n_jobs=2,random_state=SEED),
  'RBF-SVM':lambda:SVC(kernel='rbf',C=2,probability=True,class_weight='balanced',cache_size=512,random_state=SEED),
  'MLP':lambda:MLPClassifier(hidden_layer_sizes=(64,32),alpha=0.02,early_stopping=True,validation_fraction=0.15,max_iter=320,random_state=SEED),
 }
 try:
  from xgboost import XGBClassifier
  d['XGBoost']=lambda:XGBClassifier(n_estimators=230,max_depth=4,learning_rate=0.06,subsample=0.85,colsample_bytree=0.85,reg_lambda=3,scale_pos_weight=weight,eval_metric='logloss',tree_method='hist',n_jobs=2,random_state=SEED)
 except Exception as e:print('XGBoost not installed:',e)
 try:
  from lightgbm import LGBMClassifier
  d['LightGBM']=lambda:LGBMClassifier(n_estimators=240,max_depth=-1,num_leaves=15,learning_rate=0.04,min_child_samples=30,class_weight='balanced',verbosity=-1,n_jobs=2,random_state=SEED)
 except Exception as e:print('LightGBM not installed:',e)
 try:
  from catboost import CatBoostClassifier
  d['CatBoost']=lambda:CatBoostAdapter()
 except Exception as e:print('CatBoost not installed:',e)
 return d

def scores(y,p,threshold=0.5):
 pred=(p>=threshold).astype(int)
 cm=confusion_matrix(y,pred,labels=[0,1]);tn,fp,fn,tp=cm.ravel()
 return {'n':len(y),'positives':int(np.sum(y)),'threshold':float(threshold),
  'accuracy':accuracy_score(y,pred),'balanced_accuracy':balanced_accuracy_score(y,pred),
  'precision':precision_score(y,pred,zero_division=0),'recall_sensitivity':recall_score(y,pred,zero_division=0),
  'specificity':tn/(tn+fp) if tn+fp else np.nan,
  'f1':f1_score(y,pred,zero_division=0),
  'roc_auc':roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan,
  'pr_auc_average_precision':average_precision_score(y,p) if len(np.unique(y))==2 else np.nan,
  'brier':brier_score_loss(y,p),'log_loss':log_loss(y,p,labels=[0,1]),
  'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp)}

def select_threshold(y,p):
 ts=np.linspace(.05,.95,91)
 a=[(f1_score(y,p>=t,zero_division=0),t) for t in ts]
 v=max(x[0] for x in a)
 return float(sorted([t for f,t in a if f==v],key=lambda t:abs(t-0.5))[0])

def _plot_data(df,plots):
 plt.rcParams.update({'font.size':10,'figure.dpi':140})
 f,axes=plt.subplots(1,2,figsize=(11,4.3))
 counts=df.target.value_counts().reindex([0,1],fill_value=0)
 axes[0].bar(['No diagnosis','Diagnosis present'],counts.values,color=['#356c99','#d47b52'])
 for i,v in enumerate(counts.values):axes[0].text(i,v+50,f'{v:,} ({v/len(df):.1%})',ha='center')
 axes[0].set_ylim(0,max(counts.values)*1.15);axes[0].set_title('Target class prevalence')
 miss=df[COLS].isna().mean().sort_values(ascending=False).head(12)
 axes[1].barh(miss.index[::-1],miss.values[::-1],color='#547e9a');axes[1].set_xlabel('Missing rate')
 axes[1].set_xlim(0,1);axes[1].set_title('Top 12 missing features')
 f.tight_layout();f.savefig(plots/'01_dataset_distribution.png',bbox_inches='tight');plt.close(f)
 f,axes=plt.subplots(2,3,figsize=(11,6.2))
 for ax,c in zip(axes.flat,LABS):
  arrays=[df.loc[df.target==i,c].dropna() for i in [0,1]]
  if all(len(a)>0 for a in arrays):
   # winsorize for visualization only; raw values used for models
   lim=np.nanpercentile(df[c].to_numpy(float),99)
   arrays=[a.clip(upper=lim) for a in arrays]
   ax.boxplot(arrays,tick_labels=['No','Yes'],showfliers=False)
  ax.set_title(c); ax.grid(alpha=.1)
 f.suptitle('Laboratory feature distribution by target (top 1% capped in plot only)')
 f.tight_layout();f.savefig(plots/'02_lab_boxplots.png',bbox_inches='tight');plt.close(f)

def _ci_test(y,p,rng):
 ids=np.arange(len(y)); aps=[]; rocs=[]
 for k in range(650):
  subset=rng.choice(ids,len(ids),replace=True)
  if len(np.unique(y[subset]))<2:continue
  aps.append(average_precision_score(y[subset],p[subset]))
  rocs.append(roc_auc_score(y[subset],p[subset]))
 return {'ap_ci95':[float(x) for x in np.percentile(aps,[2.5,97.5])],
         'roc_auc_ci95':[float(x) for x in np.percentile(rocs,[2.5,97.5])],
         'method':'nonparametric test-set bootstrap, 650 replicates, fixed model'}

def _plot_models(validation,plots):
 d=validation.sort_values('pr_auc_average_precision')
 f,ax=plt.subplots(figsize=(9,5))
 pos=np.arange(len(d));ax.barh(pos-.18,d.pr_auc_average_precision,.35,label='PR-AUC / AP',color='#3c7096')
 ax.barh(pos+.18,d.roc_auc,.35,label='ROC-AUC',color='#de905d')
 ax.set_yticks(pos,d.model);ax.set_xlim(0,1.02);ax.legend();ax.set_title('Validation: model comparison (test excluded)')
 f.tight_layout();f.savefig(plots/'03_model_comparison.png',bbox_inches='tight');plt.close(f)

def _plot_best(y,p,t,plots):
 f,axes=plt.subplots(1,2,figsize=(10,4.2))
 fpr,tpr,_=roc_curve(y,p); prec,rec,_=precision_recall_curve(y,p)
 axes[0].plot(fpr,tpr,lw=2);axes[0].plot([0,1],[0,1],'--',lw=.7,color='#888')
 axes[0].set(xlabel='False positive rate',ylabel='True positive rate',title=f'Locked test ROC (AUC={roc_auc_score(y,p):.3f})')
 axes[1].plot(rec,prec,lw=2);axes[1].axhline(np.mean(y),ls='--',color='#888',lw=.7)
 axes[1].set(xlabel='Recall',ylabel='Precision',title=f'Locked test PR (AP={average_precision_score(y,p):.3f})')
 for ax in axes:ax.grid(alpha=.15)
 f.tight_layout();f.savefig(plots/'04_test_roc_pr.png',bbox_inches='tight');plt.close(f)
 f,axes=plt.subplots(1,2,figsize=(10,4))
 cm=confusion_matrix(y,p>=t,labels=[0,1]); ConfusionMatrixDisplay(cm,display_labels=['No','Yes']).plot(ax=axes[0],cmap='Blues',colorbar=False)
 frac,avg=calibration_curve(y,p,n_bins=8,strategy='quantile')
 axes[1].plot(avg,frac,'o-',label='Observed');axes[1].plot([0,1],[0,1],'--',color='#aaa',label='Ideal')
 axes[1].set(xlabel='Mean predicted risk',ylabel='Observed positive fraction',title='Test reliability curve')
 axes[1].legend();f.tight_layout();f.savefig(plots/'05_test_confusion_calibration.png',bbox_inches='tight');plt.close(f)

def main(args):
 out=Path(args.output);out.mkdir(parents=True,exist_ok=True);plots=out.parent/'figures';plots.mkdir(exist_ok=True)
 path=Path(args.data) if args.data else ROOT/'data'/'thyroid0387.data'
 if not path.exists():
  if args.download:download_data(path)
  else:raise FileNotFoundError(f'Data not present: {path}. Use --download or provide --data.')
 df=load_raw(path)
 if len(df)!=9172: warnings.warn(f'Expected 9,172 rows, got {len(df)}. Check source before comparing against UCI.')
 y=df.target.to_numpy();X=df[COLS].copy()
 audit={'rows':len(df),'features':29,'age_quality_rule':'Ages outside [1,110] recoded to missing; no row removed','positive_n':int(y.sum()),'positive_fraction':float(y.mean()),
        'exact_raw_duplicate_feature_rows':int(X.duplicated(keep=False).sum()),
        'record_id_unique':int(df.record_id.nunique()),
        'diagnosis_raw_counts':df.diagnosis.value_counts().to_dict(),
        'diagnosis_group_counts':dict(Counter(g for value in df.diagnosis_groups for g in value.split(';'))),
        'missing_rate':df[COLS].isna().mean().round(6).to_dict()}
 _plot_data(df,plots)
 # Approximately 60/20/20, group-disjoint exact feature patterns, final test frozen.
 sig=X.astype(str).agg('||'.join,axis=1)
 train,val,test,group_ids=grouped_three_way_split(y,sig)
 df_indices=pd.DataFrame({'row_index':np.arange(len(df)),'split':'test'})
 df_indices.loc[train,'split']='train';df_indices.loc[val,'split']='validation'
 df_indices.sort_values('row_index').to_csv(out/'row_split.csv',index=False)
 audit['split_n']={'train':len(train),'validation':len(val),'test':len(test)}
 train_signatures=set(sig.iloc[train]);audit['validation_shared_feature_patterns_with_train']=int(sig.iloc[val].isin(train_signatures).sum()); audit['test_shared_feature_patterns_with_train']=int(sig.iloc[test].isin(train_signatures).sum())
 audit['limitations']='No reliable patient identifier; record_id appears administrative and is excluded from predictors; cannot claim patient-level independence.'
 (out/'dataset_summary.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
 defs=estimator_fns(y[train]); records=[]; trained={}; vprobs={}; cv_records=[]
 for name,fn in defs.items():
  print('Training:',name,flush=True)
  model=Pipeline([('prep',preprocessor(COLS)),('clf',fn())])
  try:
   if args.cv and args.cv>=2:
    skf=StratifiedGroupKFold(n_splits=args.cv,shuffle=True,random_state=SEED)
    cv=cross_val_score(model,X.iloc[train],y[train],groups=group_ids[train],cv=skf,scoring='average_precision',n_jobs=1,error_score='raise')
    cv_records.append({'model':name,'cv_ap_mean':float(cv.mean()),'cv_ap_sd':float(cv.std(ddof=1)) if len(cv)>1 else 0.})
   model.fit(X.iloc[train],y[train]);pv=model.predict_proba(X.iloc[val])[:,1]
   t=select_threshold(y[val],pv)
   d=scores(y[val],pv,t);d['model']=name;records.append(d)
   trained[name]=model;vprobs[name]=pv
  except Exception as e:
   print('Model failed:',name,str(e)); cv_records.append({'model':name,'error':str(e)})
 if not records:raise RuntimeError('All models failed, no test set scored.')
 v=pd.DataFrame(records).sort_values('pr_auc_average_precision',ascending=False)
 v.to_csv(out/'validation_model_metrics.csv',index=False)
 pd.DataFrame(cv_records).to_csv(out/'training_cross_validation.csv',index=False)
 _plot_models(v,plots)
 best=str(v.iloc[0].model);threshold=float(v.iloc[0].threshold)
 print('Selected model on validation AP:',best,'threshold by validation F1:',threshold)
 model=trained[best];pt=model.predict_proba(X.iloc[test])[:,1]
 tsc=scores(y[test],pt,threshold);tsc['model']=best
 tsc['confidence_intervals']=_ci_test(y[test],pt,np.random.default_rng(SEED))
 (out/'locked_test_result.json').write_text(json.dumps(tsc,ensure_ascii=False,indent=2),encoding='utf-8')
 pd.DataFrame({'row_index':test,'true_label':y[test],'probability':pt,'predicted':(pt>=threshold).astype(int)}).to_csv(out/'locked_test_predictions.csv',index=False)
 _plot_best(y[test],pt,threshold,plots)
 # One-time, pre-specified secondary comparison of every model on the locked test set.
 test_comparison=[]; pred_cache={}
 for name,mdl in trained.items():
  q=mdl.predict_proba(X.iloc[test])[:,1]
  th=float(v.loc[v.model==name,'threshold'].iloc[0])
  record=scores(y[test],q,th);record['model']=name
  test_comparison.append(record);pred_cache[name]=(q>=th).astype(int)
 pd.DataFrame(test_comparison).sort_values('pr_auc_average_precision',ascending=False).to_csv(out/'locked_test_all_models.csv',index=False)
 # McNemar exact test, multiplicity adjusted for seven pre-specified baseline contrasts.
 from scipy.stats import binomtest
 base=pred_cache.get('GaussianNB')
 contrasts=[]
 if base is not None:
  good_base=(base==y[test])
  for name,pred in pred_cache.items():
   if name=='GaussianNB':continue
   good=(pred==y[test]);n10=int(np.sum(good & ~good_base));n01=int(np.sum(~good & good_base))
   pv=float(binomtest(min(n10,n01),n10+n01,0.5).pvalue) if n10+n01 else 1.0
   contrasts.append({'model':name,'baseline':'GaussianNB','b_model_only_correct':n10,'c_NB_only_correct':n01,'mcnemar_p_exact':pv})
 if contrasts:
  cc=pd.DataFrame(contrasts).sort_values('mcnemar_p_exact').reset_index(drop=True)
  m=len(cc); adjustment=[]
  for j,pval in enumerate(cc.mcnemar_p_exact): adjustment.append(min(1,float(pval)*(m-j)))
  cc['holm_adjusted_p']=np.maximum.accumulate(adjustment)
  cc.to_csv(out/'locked_test_mcnemar_holm.csv',index=False)

 # Validation-only importance and ablations prevent additional test-set selection.
 try:
  from sklearn.inspection import permutation_importance
  importance=permutation_importance(model,X.iloc[val],y[val],scoring='average_precision',n_repeats=5,random_state=SEED,n_jobs=1)
  order=np.argsort(importance.importances_mean)[::-1]
  dd=pd.DataFrame({'feature':np.asarray(COLS)[order],'ap_decrease':importance.importances_mean[order],'sd':importance.importances_std[order]})
  dd.to_csv(out/'validation_permutation_importance.csv',index=False)
  top=dd.head(15).iloc[::-1]
  f,ax=plt.subplots(figsize=(9,6));ax.barh(top.feature,top.ap_decrease,xerr=top.sd,color='#467b9c');ax.set_xlabel('Decrease in validation AP when permuted');ax.set_title(f'Permutation importance: {best}')
  f.tight_layout();f.savefig(plots/'06_validation_permutation_importance.png',bbox_inches='tight');plt.close(f)
 except Exception as e:print('Permutation importance not generated:',e)
 ablations=[('full',COLS),('minus_referral',[c for c in COLS if c!='referral_source']),('minus_measurement_flags',[c for c in COLS if c not in MEASURED]),('minus_labs',[c for c in COLS if c not in LABS])]
 ab=[]
 for label,features in ablations:
  try:
   z=Pipeline([('prep',preprocessor(features)),('clf',defs[best]())]);z.fit(X.iloc[train][features],y[train]);p=z.predict_proba(X.iloc[val][features])[:,1]
   ab.append({'ablation':label,'n_features':len(features),'validation_ap':average_precision_score(y[val],p),'validation_roc_auc':roc_auc_score(y[val],p)})
  except Exception as e:print('Ablation failed:',label,e)
 pd.DataFrame(ab).to_csv(out/'validation_ablation.csv',index=False)
 if len(ab):
  f,ax=plt.subplots(figsize=(8,3.6));a=pd.DataFrame(ab);ax.bar(a.ablation,a.validation_ap,color='#5682a1');ax.set_ylim(0,1);ax.set_ylabel('Validation AP');ax.set_title('Feature-group ablations (validation only)');ax.tick_params(axis='x',rotation=15)
  f.tight_layout();f.savefig(plots/'07_validation_ablations.png',bbox_inches='tight');plt.close(f)
 # Optional tree SHAP (validation-only). It may be unavailable due to version constraints.
 if args.shap and best in {'RandomForest','XGBoost','LightGBM','CatBoost'}:
  try:
   import shap
   tr=model.named_steps['prep']
   features=np.asarray(tr.get_feature_names_out())
   zz=tr.transform(X.iloc[val].sample(n=min(220,len(val)),random_state=SEED))
   clf=model.named_steps['clf'];est=clf.model_ if hasattr(clf,'model_') else clf
   expl=shap.TreeExplainer(est)
   raw=expl.shap_values(zz)
   if isinstance(raw,list):raw=raw[1] if len(raw)>1 else raw[0]
   raw=np.asarray(raw)
   if raw.ndim==3:raw=raw[:,:,1]
   importances=np.abs(raw).mean(axis=0); ids=np.argsort(importances)[-18:]
   plt.figure(figsize=(10,6));plt.barh(features[ids],importances[ids],color='#467b9c')
   plt.title(f'SHAP mean |value| on validation: {best}');plt.xlabel('Mean absolute SHAP value')
   plt.tight_layout();plt.savefig(plots/'09_optional_validation_shap.png',bbox_inches='tight');plt.close()
   pd.DataFrame({'feature':features,'mean_abs_shap':importances}).sort_values('mean_abs_shap',ascending=False).to_csv(out/'optional_validation_shap.csv',index=False)
  except Exception as e:print('Optional SHAP not available:',str(e))
 # Optional exploratory seven-group multilabel evaluation (each group is a binary task).
 # Not the primary, pre-registered binary endpoint; sparsely supported groups are skipped.
 if args.multilabel:
  mrows=[]
  for group in CLASS_GROUPS:
   yy=np.array([int(group in item.split(';')) for item in df.diagnosis_groups])
   ntr=int(yy[train].sum());nv=int(yy[val].sum());nt=int(yy[test].sum())
   if ntr<20 or nv<3 or nt<3:
    mrows.append({'group':group,'model':'not_evaluated','train_positives':ntr,'val_positives':nv,'test_positives':nt,
                  'reason':'Too few group-positive records for a stable held-out comparison'})
    continue
   for mdl_name in ['LogisticRegression','CatBoost','RandomForest']:
    try:
     z=Pipeline([('prep',preprocessor(COLS)),('clf',estimator_fns(yy[train])[mdl_name]())])
     z.fit(X.iloc[train],yy[train]); pval=z.predict_proba(X.iloc[val])[:,1]
     tt=select_threshold(yy[val],pval)
     ptest=z.predict_proba(X.iloc[test])[:,1]
     mrows.append({'group':group,'model':mdl_name,'train_positives':ntr,'val_positives':nv,
                   'test_positives':nt,'val_ap':average_precision_score(yy[val],pval),
                   'test_ap':average_precision_score(yy[test],ptest),'test_f1':f1_score(yy[test],ptest>=tt,zero_division=0),
                   'val_selected_threshold':tt,'reason':'exploratory secondary target'})
    except Exception as e:mrows.append({'group':group,'model':mdl_name,'reason':str(e)})
  pd.DataFrame(mrows).to_csv(out/'secondary_multilabel_groups.csv',index=False)
 # diagnosis composition, descriptive only
 freq=df.diagnosis.value_counts().head(15)
 f,ax=plt.subplots(figsize=(10,5));ax.barh(freq.index[::-1],freq.values[::-1],color='#5d809b');ax.set_xlabel('Count');ax.set_title('Top 15 original diagnostic codes (not equal to diagnostic groups)')
 f.tight_layout();f.savefig(plots/'08_diagnosis_codes.png',bbox_inches='tight');plt.close(f)
 print('COMPLETE. Saved',out.resolve(),'test AP',tsc['pr_auc_average_precision'],'test ROC AUC',tsc['roc_auc'])

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',default='');p.add_argument('--download',action='store_true');p.add_argument('--output',default=str(ROOT/'results'),help='Numerical output directory; figures go to its sibling figures directory');p.add_argument('--cv',default=3,type=int);p.add_argument('--multilabel',action='store_true',help='Also run exploratory seven-disease-group tasks');p.add_argument('--shap',action='store_true',help='Tree-SHAP on validation if selected model permits');args=p.parse_args();main(args)
