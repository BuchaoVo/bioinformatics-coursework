import os,sys
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve,roc_auc_score,precision_recall_curve,average_precision_score,brier_score_loss
from sklearn.calibration import calibration_curve
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_study import load_raw,COLS,preprocessor,estimator_fns
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results';F=ROOT/'figures';D=load_raw(ROOT/'data'/'thyroid0387.data');split=pd.read_csv(R/'row_split.csv');tr=split.loc[split.split=='train','row_index'].to_numpy();te=split.loc[split.split=='test','row_index'].to_numpy();y=D.target.to_numpy();X=D[COLS];ref=pd.read_csv(R/'locked_test_all_models.csv').set_index('model')
P=pd.DataFrame({'row_index':te,'target':y[te]});algs=estimator_fns(y[tr]);
for m,f in algs.items():
 print('Curve',m,flush=True);z=Pipeline([('prep',preprocessor(COLS)),('clf',f())]);z.fit(X.iloc[tr],y[tr]);p=z.predict_proba(X.iloc[te])[:,1];P[m]=p
 np.testing.assert_allclose(average_precision_score(y[te],p),float(ref.loc[m,'pr_auc_average_precision']),atol=.00002)
P.to_csv(R/'locked_test_probabilities_all_models.csv',index=False)
plt.rcParams.update({'font.family':'DejaVu Sans','axes.unicode_minus':False,'font.size':9.5,'savefig.dpi':230});colors=plt.cm.tab10.colors
for typ in ['ROC','PR']:
 fig,ax=plt.subplots(figsize=(7.5,5.5))
 for k,m in enumerate(ref.index):
  p=P[m].to_numpy();
  if typ=='ROC':x,yv,_=roc_curve(y[te],p);label=f'{m} ({roc_auc_score(y[te],p):.3f})'
  else:yv,x,_=precision_recall_curve(y[te],p);label=f'{m} ({average_precision_score(y[te],p):.3f})'
  ax.plot(x,yv,label=label,linewidth=1.55,color=colors[k%len(colors)])
 if typ=='ROC':ax.plot([0,1],[0,1],'--',color='gray',linewidth=.8);ax.set_xlabel('False positive rate (FPR)');ax.set_ylabel('True positive rate (TPR)')
 else:ax.axhline(np.mean(y[te]),linestyle='--',color='gray',linewidth=.8);ax.set_xlabel('Recall');ax.set_ylabel('Precision')
 ax.grid(alpha=.18);ax.set_title('Locked test: eight-model '+typ+' curves (validation-selected settings)');ax.legend(loc='lower right' if typ=='ROC' else 'lower left',fontsize=8.3);fig.tight_layout();fig.savefig(F/f'3{0 if typ=="ROC" else 1}_all_models_{typ.lower()}_curves.png',bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(7.4,5.3))
for k,m in enumerate(['LightGBM','CatBoost','RandomForest','XGBoost','GaussianNB']):
 p=P[m].to_numpy();frac,mean=calibration_curve(y[te],p,n_bins=9,strategy='quantile');ax.plot(mean,frac,'o-',label=f'{m} | Brier={brier_score_loss(y[te],p):.3f}',color=colors[k],linewidth=1.5)
ax.plot([0,1],[0,1],'--',color='gray');ax.set_xlabel('Mean predicted probability');ax.set_ylabel('Observed positive-code fraction');ax.set_title('Locked-test calibration (without recalibration)');ax.legend(fontsize=8);ax.grid(alpha=.13);fig.tight_layout();fig.savefig(F/'32_model_calibration_curves.png',bbox_inches='tight');plt.close(fig)
# 250 bootstrap resamples per model for descriptive intervals; not for model selection.
rng=np.random.default_rng(2026);n=len(te);idx=np.arange(n);boot=[]
for k,m in enumerate(ref.index):
 ap=[];roc=[];p=P[m].to_numpy()
 for i in range(250):
  s=rng.choice(idx,n,replace=True);yy=y[te][s]
  if len(set(yy))==2:ap.append(average_precision_score(yy,p[s]));roc.append(roc_auc_score(yy,p[s]))
 boot.append({'model':m,'test_ap':average_precision_score(y[te],p),'ap_ci_l':np.percentile(ap,2.5),'ap_ci_u':np.percentile(ap,97.5),'roc_ci_l':np.percentile(roc,2.5),'roc_ci_u':np.percentile(roc,97.5)})
B=pd.DataFrame(boot).sort_values('test_ap');B.to_csv(R/'descriptive_test_model_bootstrap_ci.csv',index=False)
fig,ax=plt.subplots(figsize=(7.3,4.8));ax.errorbar(B.test_ap,B.model,xerr=[B.test_ap-B.ap_ci_l,B.ap_ci_u-B.test_ap],fmt='o',color='#245e83',ecolor='#d4774d',capsize=3);ax.set_xlim(.58,1.01);ax.set_xlabel('Test AP and 95% bootstrap CI (250 resamples per model)');ax.set_title('AP uncertainty (descriptive; no model reselection)');fig.tight_layout();fig.savefig(F/'33_all_models_ap_confidence_intervals.png',bbox_inches='tight');plt.close(fig)
print('All eight model probability curves DONE')
