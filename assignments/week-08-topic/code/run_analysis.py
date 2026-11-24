#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lecture 8 heart-data study. Run from the week directory: python code/run_analysis.py
All preprocessing is nested within the fitting fold. Frozen test used only after dev selection.
"""
import os, json, argparse, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, confusion_matrix, precision_score, recall_score, f1_score, balanced_accuracy_score,
    roc_curve, precision_recall_curve, accuracy_score)
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=UserWarning)
rcParams['font.family']='DejaVu Sans'
rcParams['axes.unicode_minus']=False
rcParams['font.size']=9
rcParams['figure.dpi']=130
rcParams['savefig.dpi']=240
rcParams['axes.spines.top']=False
rcParams['axes.spines.right']=False
BLUE='#1e6083'; ORANGE='#da8057'; GREEN='#26998b'; PURPLE='#8064a2'; GREY='#7b8792'; PALETTE=[BLUE,ORANGE,GREEN,PURPLE,'#c49c32','#756f69','#ac4671','#9ea4b0']
NUM=['age','trestbps','chol','thalach','oldpeak']
CAT=['sex','cp','fbs','restecg','exang','slope','ca','thal']
FEATURES=NUM+CAT

def savefig(out,name):
    plt.tight_layout(pad=1.4); plt.savefig(out/'figures'/(name+'.png'),bbox_inches='tight');plt.close()

def processor():
    return ColumnTransformer([
      ('num',Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True)),('scaler',StandardScaler())]),NUM),
      ('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False))]),CAT)
    ],sparse_threshold=0)

def algos():
    return {
       'Prevalence baseline':DummyClassifier(strategy='prior'),
       'Gaussian NB':GaussianNB(var_smoothing=1e-8),
       'LDA':LinearDiscriminantAnalysis(solver='svd',tol=1e-4),
       'Shrinkage LDA':LinearDiscriminantAnalysis(solver='lsqr',shrinkage=0.25),
       'Logistic':LogisticRegression(C=1.0,max_iter=1800,solver='lbfgs',random_state=2026),
       'Random Forest':RandomForestClassifier(n_estimators=180,max_depth=5,min_samples_leaf=4,max_features='sqrt',random_state=2026,n_jobs=1),
       'XGBoost':XGBClassifier(n_estimators=160,max_depth=2,learning_rate=0.045,subsample=0.9,colsample_bytree=0.85,min_child_weight=4,reg_lambda=4,tree_method='hist',eval_metric='logloss',random_state=2026,n_jobs=1),
       'RBF-SVM':SVC(C=1.0,gamma='scale',probability=True,random_state=2026)
    }

def make_pipeline(name,model):
    if name=='Prevalence baseline':return Pipeline([('pre',processor()),('model',model)])
    return Pipeline([('pre',processor()),('model',model)])

def predict_metrics(y,p,t=0.5):
    z=(np.asarray(p)>=t).astype(int);a=confusion_matrix(y,z,labels=[0,1]);tn,fp,fn,tp=a.ravel()
    return dict(ROC_AUC=float(roc_auc_score(y,p)),AP=float(average_precision_score(y,p)),
      Brier=float(brier_score_loss(y,p)),LogLoss=float(log_loss(y,p,labels=[0,1])),
      Sensitivity=float(recall_score(y,z,zero_division=0)),Specificity=float(tn/(tn+fp) if tn+fp else np.nan),
      Precision=float(precision_score(y,z,zero_division=0)),F1=float(f1_score(y,z,zero_division=0)),
      BalancedAccuracy=float(balanced_accuracy_score(y,z)),Accuracy=float(accuracy_score(y,z)),
      TP=int(tp),TN=int(tn),FP=int(fp),FN=int(fn),Cost=int(5*fn+fp),Threshold=float(t))

def assess_cost(y,p):
    rows=[]
    for t in np.round(np.arange(0.05,0.951,0.01),2):
      z=(p>=t).astype(int);tn,fp,fn,tp=confusion_matrix(y,z,labels=[0,1]).ravel(); rows.append((float(t),int(5*fn+fp),int(fn),int(fp),int(tp),int(tn)))
    a=pd.DataFrame(rows,columns=['threshold','cost','FN','FP','TP','TN']);m=a.cost.min()
    # deterministic tie: closest to p* =1/(5+1)
    th=float(a[a.cost==m].iloc[(a[a.cost==m].threshold-1/6).abs().argmin()].threshold)
    return th,a

def bootstrap_ci(y,p,t=0.5,B=500,seed=2026):
    rng=np.random.default_rng(seed); ys=np.asarray(y);ps=np.asarray(p);r=[]
    for i in range(B):
       idx=rng.choice(len(ys),len(ys),replace=True); yy=ys[idx];pp=ps[idx]
       if len(np.unique(yy))<2:continue
       m=predict_metrics(yy,pp,t)
       r.append({k:m[k] for k in ['ROC_AUC','AP','Sensitivity','Specificity','F1','Brier','Cost']})
    z=pd.DataFrame(r)
    return {k:[float(z[k].quantile(.025)),float(z[k].quantile(.975))] for k in z.columns}

def prep_eda(out,raw,dev, ydev):
    # 1 label + site
    f,ax=plt.subplots(1,2,figsize=(10.6,3.3)); counts=raw.heart_disease_present.value_counts().sort_index()
    ax[0].bar(['No disease recorded','Disease recorded'],counts.values,color=[BLUE,ORANGE]);
    for i,n in enumerate(counts):ax[0].text(i,n+3,str(n),ha='center');ax[0].set_ylabel('Sample count')
    ct=pd.crosstab(raw.site,raw.heart_disease_present,normalize='index').sort_index();ct.plot(kind='barh',stacked=True,ax=ax[1],color=[BLUE,ORANGE]);ax[1].legend(['Negative','Positive'],loc='lower right',frameon=False);ax[1].set_xlabel('Outcome proportion by site');ax[1].set_ylabel('')
    savefig(out,'01_结局及站点标签分布')
    # 2 missing
    miss=raw[FEATURES].isna().sum().sort_values(ascending=False)
    fig,ax=plt.subplots(figsize=(8.0,4.5));ax.barh(miss.index[::-1],(miss/len(raw)*100).values[::-1],color=BLUE)
    for i,n in enumerate(miss.values[::-1]):
        if n:ax.text((n/len(raw)*100)+1,i,f'{n} / {len(raw)}',va='center',fontsize=8)
    ax.set_xlabel('Missing (%)');ax.set_xlim(0,max(miss/len(raw)*100)*1.25);savefig(out,'02_字段缺失比例')
    # 3 missing heatmap site ordered, sorted, cropped samples only perhaps all
    sub=raw.sort_values(['site','heart_disease_present']).reset_index(drop=True)
    fig,ax=plt.subplots(figsize=(9,5));ax.imshow(sub[FEATURES].isna().to_numpy().T,interpolation='nearest',aspect='auto',cmap='Greys',vmin=0,vmax=1)
    ax.set_yticks(range(len(FEATURES)));ax.set_yticklabels(FEATURES,fontsize=8);ax.set_xlabel('Records sorted by site and outcome');ax.set_title('White: observed; dark: missing')
    cuts=np.cumsum(sub.site.value_counts(sort=False).reindex(sub.site.drop_duplicates()).values)[:-1]
    for x in cuts:ax.axvline(x-.5,color=ORANGE,linewidth=1)
    savefig(out,'03_站点排序的缺失模式热图')
    # 4 numeric lab box
    sub=raw.copy();sub['group']=np.where(sub.heart_disease_present==1,'Positive','Negative')
    fig,axes=plt.subplots(2,3,figsize=(10,6));cn=NUM+['ca'];
    for i,(ax,c) in enumerate(zip(axes.flat,cn)):
        d=[sub.loc[sub.group==g,c].dropna() for g in ('Negative','Positive')]
        if c=='chol':d=[v[v>0] for v in d]
        ax.boxplot(d,positions=[1,2],showfliers=False,patch_artist=True,boxprops={'facecolor':'#e8f3f6'},medianprops={'color':ORANGE});ax.set_xticks([1,2]);ax.set_xticklabels(['Negative','Positive']);ax.set_title(c)
    savefig(out,'04_连续指标组间箱线图')
    # 5 corr
    corr=raw[NUM+['ca']].corr(method='spearman');L=linkage((1-corr.fillna(0).to_numpy())[np.triu_indices(len(corr),1)],method='average');order=leaves_list(L)
    c=corr.iloc[order,order];fig,ax=plt.subplots(figsize=(6.5,5.4));im=ax.imshow(c,cmap='RdBu_r',vmin=-1,vmax=1)
    ax.set_xticks(range(len(c)));ax.set_xticklabels(c.columns,rotation=40,ha='right');ax.set_yticks(range(len(c)));ax.set_yticklabels(c.index)
    for i in range(len(c)):
        for j in range(len(c)):ax.text(j,i,f'{c.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if abs(c.iloc[i,j])>.65 else '#233')
    fig.colorbar(im,ax=ax,shrink=.8,label='Spearman ρ');savefig(out,'05_层次聚类相关性热力图')
    # 6 cate prevalence heat map
    cols=['cp','restecg','slope','ca','thal','sex','fbs','exang'];entries=[]
    for c in cols:
       for k,g in raw.groupby(c,dropna=False):
          if len(g)>=7:entries.append((f'{c}={k}',len(g),g.heart_disease_present.mean()))
    e=pd.DataFrame(entries,columns=['category','n','rate']).sort_values('rate'); fig,ax=plt.subplots(figsize=(8,7));vals=e['rate'].to_numpy().reshape(-1,1)
    im=ax.imshow(vals,aspect='auto',cmap='YlOrRd',vmin=0,vmax=1)
    ax.set_yticks(range(len(e)));ax.set_yticklabels([f'{a}  (n={b})' for a,b in zip(e.category,e.n)],fontsize=7.5);ax.set_xticks([0]);ax.set_xticklabels(['Positive rate']);
    for i,v in enumerate(e.rate):ax.text(0,i,f'{v*100:.0f}%',ha='center',va='center',fontsize=7,color='white' if v>.64 else '#21354a')
    fig.colorbar(im,ax=ax,shrink=.7);savefig(out,'06_分类变量分层阳性率热图')
    # 7 distribution positive by site missing
    a=raw.groupby('site').agg(n=('heart_disease_present','size'),positive=('heart_disease_present','mean'))
    mm=raw.groupby('site')[['chol','slope','ca','thal']].apply(lambda z:z.isna().mean()).T
    fig,ax=plt.subplots(figsize=(7.4,3.8));im=ax.imshow(mm,cmap='YlGnBu',vmin=0,vmax=1);ax.set_xticks(range(len(mm.columns)));ax.set_xticklabels(mm.columns,rotation=20,ha='right');ax.set_yticks(range(len(mm.index)));ax.set_yticklabels(mm.index)
    for i in range(len(mm.index)):
        for j in range(len(mm.columns)):ax.text(j,i,f'{mm.iloc[i,j]*100:.0f}%',ha='center',va='center',color='white' if mm.iloc[i,j]>.65 else '#21354a')
    fig.colorbar(im,ax=ax,label='Missing fraction');savefig(out,'07_站点实验室资料缺失热图')
    # 8 PCA illustrative
    pp=processor().fit(dev);XX=pp.transform(dev);pcs=PCA(n_components=2,random_state=2026).fit_transform(XX);pca=PCA(n_components=2).fit(XX)
    fig,ax=plt.subplots(figsize=(7,5));
    for y,color,s in [(0,BLUE,'Negative'),(1,ORANGE,'Positive')]:
       q=np.asarray(ydev)==y;ax.scatter(pcs[q,0],pcs[q,1],s=14,alpha=.6,color=color,label=s)
    ax.set_xlabel('PC1');ax.set_ylabel('PC2');ax.set_title('Development PCA (descriptive only)');ax.legend(frameon=False);savefig(out,'08_PCA降维投影')
    return miss, a

def main(args):
  out=Path(args.out);(out/'figures').mkdir(parents=True,exist_ok=True);(out/'results').mkdir(parents=True,exist_ok=True)
  raw=pd.read_csv(args.data); assert len(raw)==270
  feature_raw=raw[FEATURES].copy(); y=raw.heart_disease_present.astype(int)
  # physiological sentinel 0 for total cholesterol only; preserve 0 as missing information not as physiology
  feature_raw.loc[feature_raw.chol==0,'chol']=np.nan
  analysis_raw=raw.copy();analysis_raw.loc[analysis_raw.chol==0,'chol']=np.nan
  idx_dev,idx_test=train_test_split(np.arange(len(raw)),test_size=.20,stratify=y,random_state=2026)
  Xdev=feature_raw.iloc[idx_dev];Xtest=feature_raw.iloc[idx_test];ydev=y.iloc[idx_dev]
  # save names of locked test only after protocol computed; data split indices do not imply peeking labels
  missing,site_summary=prep_eda(out,analysis_raw,Xdev,ydev)
  site_summary.to_csv(out/'results'/'站点组成.csv');missing.to_csv(out/'results'/'缺失数.csv',header=['missing_n'])
  raw[FEATURES+['heart_disease_present']].describe(include='all').to_csv(out/'results'/'基本描述.csv')
  # setup parameter dimensions exact on development
  proc=processor().fit(Xdev); q=proc.transform(Xdev).shape[1]
  cols=proc.get_feature_names_out().tolist(); xnum=proc.transform(Xdev)
  param=dict(n_rows=int(len(raw)),dev_n=int(len(idx_dev)),test_n=int(len(idx_test)),dev_positive=int(ydev.sum()),test_positive="locked_until_protocol_frozen",processed_dimension=int(q),logistic_parameters=int(q+1),lda_means=int(2*q),lda_covariance=int(q*(q+1)//2),lda_prior=1,
    original_missing={k:int(v) for k,v in raw[FEATURES].isna().sum().items()},chol_zero_as_missing=int((raw.chol==0).sum()),zero_count_original=int(raw[FEATURES].isna().sum().sum()),missing_after_sentinel=int(analysis_raw[FEATURES].isna().sum().sum()),feature_names=cols)
  print('PARAM',json.dumps({k:v for k,v in param.items() if k!='feature_names'},ensure_ascii=False))
  # Missingness is frequent and driven by site; use site only in stratified exploratory analysis, never as predictor.
  models=algos(); cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=2026)
  oof_cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=2026)
  stats_rows=[];model_oofs={};model_thresholds={};model_costcurves={};cv_rows=[]
  for name,alg in models.items():
    print('DEV',name,flush=True)
    # 5 fold single pass out-of-fold probabilities for model selection and threshold tuning
    pdev=np.zeros(len(ydev),dtype=float)
    for tr,va in oof_cv.split(Xdev,ydev):
       pipe=make_pipeline(name,alg); pipe.fit(Xdev.iloc[tr],ydev.iloc[tr]);pdev[va]=pipe.predict_proba(Xdev.iloc[va])[:,1]
    model_oofs[name]=pdev
    t,c=assess_cost(ydev,pdev)
    model_thresholds[name]=t;model_costcurves[name]=c
    metrics=predict_metrics(ydev,pdev,t);stats_rows.append(dict(model=name,**metrics))
    for fold,(tr,va) in enumerate(cv.split(Xdev,ydev)):
       pipe=make_pipeline(name,alg);pipe.fit(Xdev.iloc[tr],ydev.iloc[tr]);p=pipe.predict_proba(Xdev.iloc[va])[:,1]
       cv_rows.append(dict(model=name,fold=fold,AP=average_precision_score(ydev.iloc[va],p),ROC_AUC=roc_auc_score(ydev.iloc[va],p),Brier=brier_score_loss(ydev.iloc[va],p)))
  devtable=pd.DataFrame(stats_rows);cvtable=pd.DataFrame(cv_rows)
  devtable.to_csv(out/'results'/'开发集OOF_模型指标.csv',index=False,encoding='utf-8-sig');cvtable.to_csv(out/'results'/'开发集15折重复CV.csv',index=False,encoding='utf-8-sig')
  oofp=pd.DataFrame({'sample_id':raw.iloc[idx_dev].sample_id.to_numpy(),'y':ydev.to_numpy(),**{n:p for n,p in model_oofs.items()}})
  oofp.to_csv(out/'results'/'开发集折外预测概率.csv',index=False,encoding='utf-8-sig')
  # choose between LDA-shrinkage and logistic by OOF cost, tie break Brier; preselected primary comparison, not among all complex models
  candidate=['Logistic','Shrinkage LDA']; cand=devtable.set_index('model').loc[candidate].sort_values(['Cost','Brier'])
  chosen=str(cand.index[0]); chosen_t=model_thresholds[chosen]
  frozen={'model':chosen,'threshold':chosen_t,'selection':'Only Logistic vs shrinkage-LDA; minimize 5*FN+FP on single-pass 5-fold development OOF; tie-break Brier','test_locked_before_model_and_threshold_selection':True,'random_state':2026,'dev_test_split':'stratified 80/20','preprocess':'cholesterol=0 -> NA; train-fold median numeric+missing indicators+standardize; train-fold mode categorical+one-hot; no site/identifiers/num','primary_test_metrics':['AP','ROC_AUC','Brier','Sensitivity','Specificity','F1','Cost']}
  (out/'results'/'冻结方案_先于测试.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2),encoding='utf-8')
  # illustration figures with development only before opening test
  plt.figure(figsize=(8.5,4));groups=[cvtable[cvtable.model==n].AP.values for n in models];plt.boxplot(groups,vert=False,tick_labels=list(models),showfliers=False,patch_artist=True,boxprops={'facecolor':'#d9e9ef'});plt.xlabel('AP across 15 repeated stratified CV folds');savefig(out,'09_重复交叉验证AP分布')
  # 10 radar normalized to [0,1], Brier inversion; full y range
  radar_names=['Logistic','LDA','Shrinkage LDA','Random Forest','XGBoost'];radar_cols=['ROC_AUC','AP','Sensitivity','Specificity','F1','BalancedAccuracy']
  theta=np.linspace(0,2*np.pi,len(radar_cols),endpoint=False);theta=np.r_[theta,theta[0]]
  fig,ax=plt.subplots(figsize=(7,6),subplot_kw={'polar':True});
  for i,n in enumerate(radar_names):
       r=devtable.set_index('model').loc[n,radar_cols].to_numpy().astype(float);r=np.r_[r,r[0]];ax.plot(theta,r,color=PALETTE[i],linewidth=1.6,label=n);ax.fill(theta,r,color=PALETTE[i],alpha=.06)
  ax.set_xticks(theta[:-1]);ax.set_xticklabels(['ROC-AUC','AP','Sensitivity','Specificity','F1','Balanced accuracy']);ax.set_ylim(0,1);ax.legend(loc='center left',bbox_to_anchor=(1.08,.5),frameon=False);savefig(out,'10_开发集多维性能雷达图')
  # 11 metric heatmap
  mat=devtable.set_index('model')[['ROC_AUC','AP','F1','Sensitivity','Specificity','Brier']]
  fig,ax=plt.subplots(figsize=(9.2,4.5));im=ax.imshow(mat.to_numpy(),cmap='YlGnBu',vmin=0,vmax=1,aspect='auto');ax.set_xticks(range(len(mat.columns)));ax.set_xticklabels(mat.columns,rotation=30,ha='right');ax.set_yticks(range(len(mat.index)));ax.set_yticklabels(mat.index)
  for i in range(len(mat.index)):
      for j in range(len(mat.columns)):ax.text(j,i,f'{mat.iloc[i,j]:.3f}',ha='center',va='center',fontsize=8,color='white' if mat.iloc[i,j]>.65 else '#202d38')
  fig.colorbar(im,ax=ax,shrink=.7);savefig(out,'11_开发集指标热力图_注意Brier越小越好')
  # 12,13 ROC/PR CV predictions
  fig,ax=plt.subplots(figsize=(7,5));
  for i,n in enumerate(models):
      if n=='Prevalence baseline':continue
      fpr,tpr,_=roc_curve(ydev,model_oofs[n]);ax.plot(fpr,tpr,color=PALETTE[i],lw=1.7,label=f'{n}: {roc_auc_score(ydev,model_oofs[n]):.3f}')
  ax.plot([0,1],[0,1],'--',color='#b8bac0');ax.set(xlabel='1 − Specificity',ylabel='Sensitivity',title='Development ROC: single-pass five-fold OOF');ax.legend(fontsize=7,frameon=False,loc='lower right');savefig(out,'12_开发集ROC曲线')
  fig,ax=plt.subplots(figsize=(7,5));
  for i,n in enumerate(models):
      if n=='Prevalence baseline':continue
      pr,re,_=precision_recall_curve(ydev,model_oofs[n]);ax.plot(re,pr,color=PALETTE[i],lw=1.7,label=f'{n}: AP {average_precision_score(ydev,model_oofs[n]):.3f}')
  ax.axhline(ydev.mean(),linestyle='--',color=GREY);ax.set(xlabel='Recall',ylabel='Precision',title='Development OOF precision-recall');ax.legend(fontsize=7,frameon=False,loc='lower left');savefig(out,'13_开发集PR曲线')
  # 14 calibration
  fig,ax=plt.subplots(figsize=(6.2,5));
  for i,n in enumerate(['Logistic','LDA','Shrinkage LDA','Random Forest','XGBoost','Gaussian NB']):
      p=model_oofs[n];pred,meanp=calibration_curve(ydev,p,n_bins=6,strategy='quantile');ax.plot(meanp,pred,'-o',lw=1.2,markersize=3,color=PALETTE[i],label=n)
  ax.plot([0,1],[0,1],'--',color=GREY);ax.set(xlabel='Mean predicted probability in bin',ylabel='Observed positive rate',title='Development calibration (quantile bins)');ax.legend(frameon=False,fontsize=7);savefig(out,'14_开发集概率可靠性曲线')
  # 15 cost curves
  fig,ax=plt.subplots(figsize=(7,4.7));
  for i,n in enumerate(['Logistic','LDA','Shrinkage LDA','Random Forest','XGBoost']):
     c=model_costcurves[n];ax.plot(c.threshold,c.cost,color=PALETTE[i],label=f'{n} t={model_thresholds[n]:.2f}',lw=1.5)
  ax.set(xlabel='Threshold',ylabel='Development cost: 5×FN+FP',title='OOF threshold vs. cost (development selection)');ax.legend(frameon=False,fontsize=8);savefig(out,'15_代价敏感阈值曲线')
  # model parameters, associations logistic fit dev
  logit=make_pipeline('Logistic',LogisticRegression(C=1,max_iter=1800,random_state=2026));logit.fit(Xdev,ydev)
  feat=logit.named_steps['pre'].get_feature_names_out(); coef=logit.named_steps['model'].coef_.ravel();
  coeff=pd.DataFrame({'encoded_feature':feat,'coefficient':coef,'OR_per_unit':np.exp(coef)}).sort_values('coefficient');coeff.to_csv(out/'results'/'Logistic系数OR.csv',index=False,encoding='utf-8-sig')
  selected=pd.concat([coeff.head(9),coeff.tail(9)]).drop_duplicates();fig,ax=plt.subplots(figsize=(8.2,6.5));ax.barh(selected.encoded_feature,selected.coefficient,color=[BLUE if a<0 else ORANGE for a in selected.coefficient]);ax.axvline(0,color=GREY,lw=.8);ax.set_xlabel('Adjusted log-odds coefficient (per SD for continuous inputs)');savefig(out,'16_Logistic条件关联系数森林条形图')
  # bootstrap: near duplicate age: direct numeric duplicate with gaussian noise deterministic
  rng=np.random.default_rng(2026);xd=Xdev.copy();xd['age_near_dup']=xd['age']+rng.normal(0,.15,len(xd));
  pre2=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True)),('scaler',StandardScaler())]),NUM+['age_near_dup']),('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(drop='first',handle_unknown='ignore',sparse_output=False))]),CAT)],sparse_threshold=0)
  coefs=[];coeff2=[];bootmetrics=[]
  for b in range(180):
      ind=rng.choice(np.arange(len(xd)),size=len(xd),replace=True);
      if len(np.unique(ydev.iloc[ind]))<2:continue
      x_b=Xdev.iloc[ind];y_b=ydev.iloc[ind];m0=make_pipeline('Logistic',LogisticRegression(C=1,max_iter=900,random_state=2026)).fit(x_b,y_b)
      m1=Pipeline([('pre',pre2),('model',LogisticRegression(C=1,max_iter=900,random_state=2026))]).fit(xd.iloc[ind],y_b)
      c0=dict(zip(m0.named_steps['pre'].get_feature_names_out(),m0.named_steps['model'].coef_[0]));c1=dict(zip(m1.named_steps['pre'].get_feature_names_out(),m1.named_steps['model'].coef_[0]))
      coefs.append(c0.get('num__age',np.nan));coeff2.append((c1.get('num__age',np.nan),c1.get('num__age_near_dup',np.nan)))
      bootmetrics.append([roc_auc_score(ydev,m0.predict_proba(Xdev)[:,1]),roc_auc_score(ydev,m1.predict_proba(xd)[:,1])])
  coefs=np.array(coefs);coeff2=np.array(coeff2);bootmetrics=np.array(bootmetrics)
  sta=pd.DataFrame({'baseline_age_coefficient':coefs,'duplicate_age_original_coefficient':coeff2[:,0],'duplicate_age_copy_coefficient':coeff2[:,1], 'baseline_apparent_auc':bootmetrics[:,0],'duplicate_apparent_auc':bootmetrics[:,1]})
  sta.to_csv(out/'results'/'年龄近重复_Bootstrap系数.csv',index=False,encoding='utf-8-sig')
  fig,ax=plt.subplots(figsize=(8,4.5));ax.boxplot([coefs,coeff2[:,0],coeff2[:,1],coeff2.sum(axis=1)],tick_labels=['Original age','Age after addition','Near-duplicate age','Sum of age coefficients'],patch_artist=True,boxprops={'facecolor':'#e1edf1'});ax.axhline(0,color=GREY,lw=.8);ax.set_ylabel('Bootstrap regression coefficient');ax.set_title('Near collinearity: individual and summed coefficients (180 resamples)');savefig(out,'17_年龄近重复变量系数稳定性')
  # 18 LDA gaussian stress class conditional skew and covariance
  tmp=proc.transform(Xdev); yarr=ydev.to_numpy();v0=tmp[yarr==0];v1=tmp[yarr==1]
  s0=np.cov(v0,rowvar=False);s1=np.cov(v1,rowvar=False);covdiff=float(np.linalg.norm(s0-s1,'fro')/(np.linalg.norm((s0+s1)/2,'fro')+1e-9))
  skews={k:[float(stats.skew(Xdev.loc[ydev==v,k].dropna())) if len(Xdev.loc[ydev==v,k].dropna())>10 else None for v in [0,1]] for k in NUM}
  fig,axes=plt.subplots(1,2,figsize=(9,3.7));
  k=min(12,s0.shape[0]);axes[0].imshow(s0[:k,:k],vmin=-2,vmax=2,cmap='RdBu_r');axes[1].imshow(s1[:k,:k],vmin=-2,vmax=2,cmap='RdBu_r');axes[0].set_title('Development negative covariance (first 12 features)');axes[1].set_title('Development positive covariance (first 12 features)')
  savefig(out,'18_LDA类内协方差压力图')
  # non-param rank tests on development only
  univ=[]
  for c in NUM:
     x0=Xdev.loc[ydev==0,c].dropna(); x1=Xdev.loc[ydev==1,c].dropna();
     u,p=stats.mannwhitneyu(x1,x0,alternative='two-sided'); effect=2*u/(len(x0)*len(x1))-1
     univ.append(dict(feature=c,method='Mann–Whitney U',effect=float(effect),p=float(p),n0=len(x0),n1=len(x1)))
  for c in CAT:
     ta=pd.crosstab(Xdev[c].fillna(-99),ydev); stat,p,dof,expected=stats.chi2_contingency(ta);v=np.sqrt(stat/(ta.values.sum()*(min(ta.shape)-1))) if min(ta.shape)>1 else np.nan
     univ.append(dict(feature=c,method='Chi-square exploratory',effect=float(v),p=float(p),n0=int((ydev==0).sum()),n1=int((ydev==1).sum())))
  ut=pd.DataFrame(univ);ut['q_BH']=multipletests(ut.p,method='fdr_bh')[1];ut.to_csv(out/'results'/'单变量差异及BH校正.csv',index=False,encoding='utf-8-sig')
  fig,ax=plt.subplots(figsize=(8,4.2));plot=ut.sort_values('q_BH');ax.barh(plot.feature[::-1],-np.log10(plot.q_BH.clip(lower=1e-300))[::-1],color=[GREEN if q<.05 else GREY for q in plot.q_BH[::-1]]);ax.axvline(-np.log10(.05),color=ORANGE,linestyle='--');ax.set_xlabel('−log10 (BH-adjusted p)');ax.set_title('Development univariate associations (different effect scales)');savefig(out,'19_BH校正单变量关联证据')
  # IMPORTANT: frozen before accessing test label below -- no retraining/tuning after test
  print('FROZEN',json.dumps(frozen,ensure_ascii=False))
  # Unlock test labels only after model and threshold have been frozen.
  ytest=y.iloc[idx_test]
  param['test_positive']=int(ytest.sum())
  rows=[];testprobs={};chosen_predictor=None
  for name,alg in models.items():
      pipe=make_pipeline(name,alg);pipe.fit(Xdev,ydev);pt=pipe.predict_proba(Xtest)[:,1];testprobs[name]=pt
      m=predict_metrics(ytest,pt,model_thresholds[name]);rows.append(dict(model=name,**m))
      if name==chosen:chosen_predictor=pipe
  testtable=pd.DataFrame(rows);testtable.to_csv(out/'results'/'锁定测试_八模型实测指标.csv',index=False,encoding='utf-8-sig')
  pd.DataFrame({'sample_id':raw.iloc[idx_test].sample_id.to_numpy(),'y':ytest.to_numpy(),**testprobs}).to_csv(out/'results'/'锁定测试_全部模型预测概率.csv',index=False,encoding='utf-8-sig')
  ci=bootstrap_ci(ytest,testprobs[chosen],chosen_t,B=650);(out/'results'/'选择模型测试Bootstrap_CI.json').write_text(json.dumps(ci,ensure_ascii=False,indent=2))
  # 20 tests heatmap
  mat=testtable.set_index('model')[['ROC_AUC','AP','F1','Sensitivity','Specificity','Brier']];fig,ax=plt.subplots(figsize=(9.2,4.5));im=ax.imshow(mat,cmap='YlGnBu',vmin=0,vmax=1,aspect='auto');ax.set_xticks(range(len(mat.columns)));ax.set_xticklabels(mat.columns,rotation=30,ha='right');ax.set_yticks(range(len(mat.index)));ax.set_yticklabels(mat.index)
  for i in range(len(mat.index)):
      for j in range(len(mat.columns)):ax.text(j,i,f'{mat.iloc[i,j]:.3f}',ha='center',va='center',fontsize=8,color='white' if mat.iloc[i,j]>.65 else '#172634')
  fig.colorbar(im,ax=ax,shrink=.7);savefig(out,'20_锁定测试模型性能热图')
  # 21/22 ROC PR test
  fig,ax=plt.subplots(figsize=(7,5))
  for i,n in enumerate(models):
      if n=='Prevalence baseline':continue
      fpr,tpr,_=roc_curve(ytest,testprobs[n]);ax.plot(fpr,tpr,lw=1.6,color=PALETTE[i],label=f'{n} {roc_auc_score(ytest,testprobs[n]):.3f}')
  ax.plot([0,1],[0,1],color=GREY,linestyle='--');ax.set(xlabel='1 − Specificity',ylabel='Sensitivity',title='Locked test: ROC discrimination');ax.legend(frameon=False,fontsize=7,loc='lower right');savefig(out,'21_锁定测试ROC')
  fig,ax=plt.subplots(figsize=(7,5))
  for i,n in enumerate(models):
      if n=='Prevalence baseline':continue
      pr,re,_=precision_recall_curve(ytest,testprobs[n]);ax.plot(re,pr,lw=1.6,color=PALETTE[i],label=f'{n} AP {average_precision_score(ytest,testprobs[n]):.3f}')
  ax.axhline(ytest.mean(),color=GREY,linestyle='--');ax.set(xlabel='Recall',ylabel='Precision',title='Locked test: precision-recall');ax.legend(frameon=False,fontsize=7,loc='lower left');savefig(out,'22_锁定测试PR')
  # 23 confusion + 24 calibration
  mt=testtable.set_index('model').loc[chosen];a=confusion_matrix(ytest,(testprobs[chosen]>=chosen_t).astype(int),labels=[0,1]);fig,ax=plt.subplots(figsize=(5,4.3));ax.imshow(a,cmap='Blues')
  ax.set_xticks([0,1]);ax.set_xticklabels(['Predicted negative','Predicted positive']);ax.set_yticks([0,1]);ax.set_yticklabels(['Observed negative','Observed positive']);
  for i in range(2):
     for j in range(2):ax.text(j,i,str(a[i,j]),ha='center',va='center',fontsize=16,color='white' if a[i,j]>.5*a.max() else '#1d3547')
  ax.set_title(f'Frozen model: {chosen}, threshold={chosen_t:.2f}');savefig(out,'23_选定模型测试集混淆矩阵')
  fig,ax=plt.subplots(figsize=(5.7,4.6));
  for i,n in enumerate(['Logistic','LDA','Shrinkage LDA','Random Forest','XGBoost']):
     frac,prob=calibration_curve(ytest,testprobs[n],n_bins=6,strategy='quantile');ax.plot(prob,frac,'-o',markersize=3,color=PALETTE[i],label=n)
  ax.plot([0,1],[0,1],'--',color=GREY);ax.set(xlabel='Mean predicted probability',ylabel='Observed positive rate in bin',title='Locked-test calibration (descriptive)');ax.legend(frameon=False,fontsize=8);savefig(out,'24_锁定测试概率可靠性曲线')
  # 25 importance (permute raw data on locked test descriptive only, no feature tuning)
  perm=permutation_importance(chosen_predictor,Xtest,ytest,n_repeats=15,random_state=2026,scoring='average_precision',n_jobs=1)
  imp=pd.DataFrame({'feature':FEATURES,'mean_AP_decrease':perm.importances_mean,'sd':perm.importances_std}).sort_values('mean_AP_decrease',ascending=False);imp.to_csv(out/'results'/'冻结模型测试集置换重要性_描述性.csv',index=False,encoding='utf-8-sig')
  fig,ax=plt.subplots(figsize=(7.7,4.5));e=imp.sort_values('mean_AP_decrease');ax.barh(e.feature,e.mean_AP_decrease,xerr=e.sd,color=BLUE,error_kw={'elinewidth':.8});ax.axvline(0,color=GREY,linewidth=.8);ax.set_xlabel('AP decrease after permutation, mean ± SD (descriptive only)');savefig(out,'25_模型原始特征置换重要性')
  # 26 site performance and prevalence test
  test_site=raw.iloc[idx_test].site.reset_index(drop=True); ys=ytest.reset_index(drop=True);pred=np.asarray(testprobs[chosen]);summ=[]
  for site in sorted(test_site.unique()):
     inds=np.flatnonzero((test_site==site).to_numpy());yy=ys.iloc[inds].to_numpy();pp=pred[inds];
     zz=pp>=chosen_t;tn,fp,fn,tp=confusion_matrix(yy,zz,labels=[0,1]).ravel();summ.append(dict(site=site,n=len(inds),positive=int(yy.sum()),positive_rate=float(yy.mean()),sensitivity=float(tp/(tp+fn)) if tp+fn else np.nan,specificity=float(tn/(tn+fp)) if tn+fp else np.nan, cost=int(5*fn+fp)))
  subg=pd.DataFrame(summ);subg.to_csv(out/'results'/'测试集站点分层结果_探索性.csv',index=False,encoding='utf-8-sig')
  fig,ax=plt.subplots(figsize=(7.5,4));xx=np.arange(len(subg));ax.bar(xx-.2,subg.sensitivity,width=.4,color=ORANGE,label='Sensitivity');ax.bar(xx+.2,subg.specificity,width=.4,color=BLUE,label='Specificity');ax.set_xticks(xx);ax.set_xticklabels([f'{s}\nn={n}' for s,n in zip(subg.site,subg.n)],fontsize=8);ax.set_ylim(0,1.08);ax.legend(frameon=False);ax.set_title('Locked test by site (small samples; exploratory)');savefig(out,'26_锁定测试站点分层敏感度特异度')
  # 27 compare costs
  fig,ax=plt.subplots(figsize=(7.5,4));m=testtable.sort_values('Cost',ascending=True);ax.barh(m.model[::-1],m.Cost[::-1],color=[GREEN if z==chosen else BLUE for z in m.model[::-1]]);ax.set_xlabel('Locked-test error cost: 5×FN+FP');ax.set_title('Test evaluation at development-frozen thresholds');savefig(out,'27_测试集模型错误代价比较')
  # output summary
  summary={'parameters':param,'frozen':frozen,'development':devtable.to_dict('records'),'test':testtable.to_dict('records'),'CI_selected':ci,'LDA_cov_difference_ratio':covdiff,'skews_by_class':skews,'age_bootstrap':{'base_sd':float(np.std(coefs)),'age_with_copy_sd':float(np.std(coeff2[:,0])),'copy_sd':float(np.std(coeff2[:,1])),'base_auc_apparent_mean':float(bootmetrics[:,0].mean()),'copy_auc_apparent_mean':float(bootmetrics[:,1].mean())},'site_subgroups':summ}
  (out/'results'/'完整结果摘要.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
  print('FINISHED selected=',chosen,'threshold=',chosen_t,'TEST=',testtable.loc[testtable.model==chosen].to_dict('records'),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',default=str(ROOT/'data'/'heart.csv'));p.add_argument('--out',default=str(ROOT));a=p.parse_args();main(a)
