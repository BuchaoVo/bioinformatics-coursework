#!/usr/bin/env python3
"""Publication supplemental plots: observations derived from real uploaded data only."""
import sys, json, warnings
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, chi2_contingency, fisher_exact
from scipy.cluster.hierarchy import linkage
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, f1_score, precision_score,recall_score, roc_auc_score, roc_curve, precision_recall_curve
from statsmodels.stats.multitest import multipletests
sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_study import load_raw,COLS,BOOLS,LABS,CATS,NUMS,MEASURED,preprocessor,estimator_fns,SEED
ROOT=Path(__file__).resolve().parents[1];RES=ROOT/'results'; FIG=ROOT/'figures';FIG.mkdir(exist_ok=True,parents=True)
D=load_raw(ROOT/'data/thyroid0387.data');V=pd.read_csv(RES/'validation_model_metrics.csv'); T=pd.read_csv(RES/'locked_test_all_models.csv');CV=pd.read_csv(RES/'training_cross_validation.csv'); A=pd.read_csv(RES/'validation_ablation.csv'); I=pd.read_csv(RES/'validation_permutation_importance.csv'); S=pd.read_csv(RES/'row_split.csv');rng=np.random.default_rng(2026)
plt.rcParams.update({'font.family':'DejaVu Sans','axes.unicode_minus':False,'font.size':10,'axes.titlesize':12,'savefig.dpi':220,'axes.spines.top':False,'axes.spines.right':False})
sns.set_context('paper',font_scale=1.08);sns.set_style('whitegrid',{'font.family':'DejaVu Sans'})
COLORS=['#245e83','#d4774d','#59a89c','#8e78a6','#daab52','#669acd','#cd7087','#7a8e73'];blue='#245e83';orange='#d4774d'
def save(f,n):f.savefig(FIG/n,dpi=230,bbox_inches='tight',facecolor='white');plt.close(f);print('FIG',n,flush=True)
# 10 Class/count composition
f,axs=plt.subplots(1,2,figsize=(10,4.0));counts=D.target.value_counts().reindex([0,1]); axs[0].pie(counts,labels=['No additional code (-)','Additional code present'],autopct='%1.1f%%',startangle=90,colors=[blue,orange],wedgeprops=dict(width=.42,edgecolor='white'));axs[0].text(0,0,str(len(D)),ha='center',va='center',weight='bold',fontsize=15);axs[0].set_title('Target proportions in all records')
ct=D.diagnosis.value_counts().head(12);axs[1].barh(ct.index[::-1],ct.values[::-1],color=[orange if s!='-' else blue for s in ct.index[::-1]]);axs[1].set_title('Top 12 diagnosis code combinations');axs[1].set_xlabel('Record count');save(f,'10_target_and_diagnosis_codes.png')
# 11 missingness: row sample and mean
miss=D[COLS].isna();pref=list(miss.mean().sort_values(ascending=False).head(15).index);inds=rng.choice(len(D),min(450,len(D)),replace=False);f,axs=plt.subplots(1,2,figsize=(11,4.1),gridspec_kw={'width_ratios':[1.3,1]});sns.heatmap(miss.iloc[inds][pref].astype(int),ax=axs[0],cmap=sns.color_palette(['#eef2f5',orange]),cbar=False,yticklabels=False,xticklabels=True);axs[0].tick_params(axis='x',rotation=75);axs[0].set_title('Missingness patterns (450 sampled records)');a=miss.mean().sort_values(ascending=False).head(12);axs[1].barh(a.index[::-1],100*a.values[::-1],color=blue);axs[1].set_xlim(0,105);axs[1].set_xlabel('Missing (%)');axs[1].set_title('Missingness by feature');f.tight_layout();save(f,'11_missingness_patterns.png')
# 12 Laboratory correlation spearman
corr=D[['age']+LABS].corr(method='spearman',min_periods=100);f,ax=plt.subplots(figsize=(6.8,5.5));sns.heatmap(corr,cmap='RdBu_r',vmin=-1,vmax=1,center=0,annot=True,fmt='.2f',square=True,ax=ax,linewidths=.4,cbar_kws={'label':'Spearman ρ','shrink':.72});ax.set_title('Spearman correlations: age and laboratory values');save(f,'12_clinical_feature_correlations.png')
# 13 lab differences numeric effect with BH (rank biserial)
rows=[]
for c in ['age']+LABS:
 x=D.loc[D.target==1,c].dropna().to_numpy(float);y=D.loc[D.target==0,c].dropna().to_numpy(float)
 if len(x)<10 or len(y)<10:continue
 u,p=mannwhitneyu(x,y,alternative='two-sided');rbc=2*u/(len(x)*len(y))-1
 rows.append({'feature':c,'n_non_dash':len(x),'n_dash':len(y),'median_non_dash':np.median(x),'median_dash':np.median(y),'rank_biserial_r':rbc,'p_value':p})
rank=pd.DataFrame(rows);rank['q_bh']=multipletests(rank.p_value,method='fdr_bh')[1];rank.to_csv(RES/'numeric_mannwhitney_fdr.csv',index=False)
f,ax=plt.subplots(figsize=(8.1,4.0));z=rank.sort_values('rank_biserial_r');ax.barh(z.feature,z.rank_biserial_r,color=[orange if t>0 else blue for t in z.rank_biserial_r]);ax.axvline(0,color='#555',lw=.8);ax.set_xlabel('Rank-biserial correlation');ax.set_title('Numeric effect sizes (Mann-Whitney U)');
for i,v in enumerate(z.itertuples()):ax.text(v.rank_biserial_r+(.01 if v.rank_biserial_r>=0 else -.01),i,'q='+('%.1e'%v.q_bh),va='center',fontsize=8,ha='left' if v.rank_biserial_r>=0 else 'right')
save(f,'13_numeric_effect_sizes.png')
# 14 categorical associations Fisher + BH
entries=[]
for c in BOOLS:
 tab=pd.crosstab(D[c],D.target).reindex(index=[0.,1.],columns=[0,1],fill_value=0).to_numpy()
 if not np.all(tab>=0):continue
 odds,p=fisher_exact(tab)
 rr=(tab[1,1]/max(tab[1].sum(),1))-(tab[0,1]/max(tab[0].sum(),1))
 entries.append({'feature':c,'group_positive_difference':rr,'odds_ratio':odds,'p_value':p,'positive_n':int(tab[1].sum())})
cat=pd.DataFrame(entries);cat['q_bh']=multipletests(cat.p_value,method='fdr_bh')[1];cat.to_csv(RES/'binary_feature_fisher_fdr.csv',index=False)
top=cat.sort_values('q_bh').head(13).sort_values('group_positive_difference');f,ax=plt.subplots(figsize=(8.2,5.1));ax.barh(top.feature,top.group_positive_difference,color=[orange if v>0 else blue for v in top.group_positive_difference]);ax.axvline(0,color='#666',lw=.8);ax.set_xlabel('Difference in positive-code proportion with vs. without condition');ax.set_title('Binary feature associations (Fisher exact test)');save(f,'14_binary_feature_associations.png')
# 15 row-cluster / group median heatmap (codes top)
code=D.diagnosis.value_counts();codes=[z for z in code.index if code[z]>=60][:13];features=['TSH','T3','TT4','T4U','FTI','age'];trans=D[features].copy();trans['TSH']=np.log1p(trans['TSH'].clip(lower=0));trans['TBG']=0 if False else trans.get('TBG',pd.Series(dtype=float));trans=trans[features]; median=trans.median();scale=(trans.quantile(.75)-trans.quantile(.25)).replace(0,1);Xg=trans.assign(code=D.diagnosis).groupby('code')[features].median().loc[codes];zg=((Xg-median)/scale).clip(-3,3);f,ax=plt.subplots(figsize=(7.2,6));sns.heatmap(zg,cmap='RdBu_r',center=0,vmin=-2,vmax=2,annot=True,fmt='.1f',linewidths=.3,cbar_kws={'label':'Group medians scaled by overall IQR'},ax=ax);ax.set_title('Laboratory profiles by diagnosis code');ax.set_xlabel('Numeric features (log1p applied to TSH)');ax.set_ylabel('Original diagnosis code');save(f,'15_diagnosis_laboratory_profiles.png')
# 16 PCA model preprocessor unsupervised: use random at most 2400
inds=rng.choice(len(D),min(2400,len(D)),replace=False);xs=D.iloc[inds][COLS];prep=preprocessor(COLS);zp=prep.fit_transform(xs);pca=PCA(n_components=2,random_state=SEED).fit(zp);xx=pca.transform(zp);f,ax=plt.subplots(figsize=(7,5.1));for_order=[0,1]
for j,color,lab in [(0,blue,'-'),(1,orange,'Non-dash')]:q=D.iloc[inds].target.to_numpy()==j;ax.scatter(xx[q,0],xx[q,1],s=9,alpha=.35,label=lab,c=color,rasterized=True)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})');ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})');ax.legend(title='Target label',markerscale=2);ax.set_title('Clinical feature PCA (exploratory; n=2,400)');save(f,'16_clinical_feature_pca.png')
# 17 age distribution & sex
f,axs=plt.subplots(1,2,figsize=(10.4,4));
for j,c in [(0,blue),(1,orange)]:sns.kdeplot(D.loc[D.target==j,'age'].clip(0,110),ax=axs[0],label='-' if j==0 else 'Non-dash',color=c,fill=True,alpha=.22,linewidth=1.5)
axs[0].legend();axs[0].set_xlabel('Age (years)');axs[0].set_title('Age density');tb=pd.crosstab(D.sex.fillna('Unknown'),D.target);pd.DataFrame(tb.div(tb.sum(axis=1),axis=0)).plot(kind='bar',stacked=True,ax=axs[1],color=[blue,orange],rot=0,legend=False);axs[1].set_ylabel('Proportion');axs[1].set_title('Target proportions by sex');save(f,'17_age_and_sex_distribution.png')
# 18 lab violins (subsets avoid huge)
f,axs=plt.subplots(2,3,figsize=(10.6,6.8));
for ax,c in zip(axs.ravel(),LABS):
 subset=D[['target',c]].dropna(); cap=subset[c].quantile(.99);subset=subset.assign(v=subset[c].clip(upper=cap));sns.violinplot(data=subset,x='target',y='v',ax=ax,inner='quart',palette=[blue,orange],cut=0,linewidth=.7,hue='target',legend=False);ax.set_title(c+' (≤99th pct shown)');ax.set_xlabel('0=dash / 1=non-dash');ax.set_ylabel('Observed value')
f.suptitle('Laboratory values by target (clipped for visualization only)',y=1);f.tight_layout();save(f,'18_laboratory_violin_plots.png')
# 19 Heatmap of model scores (without prespecified threshold effects misleading)
metrics=['pr_auc_average_precision','roc_auc','balanced_accuracy','f1','recall_sensitivity','specificity'];display={'pr_auc_average_precision':'AP','roc_auc':'ROC-AUC','balanced_accuracy':'Balanced Acc.','f1':'F1','recall_sensitivity':'Sensitivity','specificity':'Specificity'};heat=T.set_index('model')[metrics].rename(columns=display).sort_values('AP',ascending=False);f,ax=plt.subplots(figsize=(10,5.1));sns.heatmap(heat,annot=True,fmt='.3f',cmap='YlGnBu',vmin=0.50,vmax=1.0,ax=ax,cbar_kws={'label':'Locked-test score'},linewidths=.4);ax.set_title('Eight-model test metrics (descriptive; no test-based selection)');ax.set_ylabel('');save(f,'19_model_metrics_heatmap.png')
# 20 radar (top five and naive reference) normal scale 0-1
items=['LightGBM','CatBoost','RandomForest','XGBoost','MLP','GaussianNB'];mc=['pr_auc_average_precision','roc_auc','f1','recall_sensitivity','specificity','balanced_accuracy'];rnames=['AP','ROC-AUC','F1','Sensitivity','Specificity','B-Acc'];angles=np.linspace(0,2*np.pi,len(mc),endpoint=False).tolist();ang=angles+[angles[0]];f,ax=plt.subplots(figsize=(7.1,6.1),subplot_kw={'polar':True});
for j,name in enumerate(items):r=T.set_index('model').loc[name,mc].to_numpy(float).tolist();r+=r[:1];ax.plot(ang,r,linewidth=1.7,label=name,color=COLORS[j]);ax.fill(ang,r,alpha=.035,color=COLORS[j]);
ax.set_xticks(angles,rnames);ax.set_ylim(0,1);ax.set_yticks([.2,.4,.6,.8,1.0]);ax.set_title('Performance radar (original values; no cross-metric scaling)',pad=17);ax.legend(loc='upper left',bbox_to_anchor=(1.04,1),fontsize=8);save(f,'20_model_performance_radar.png')
# 21 Fold CV confidence visual
cc=CV.sort_values('cv_ap_mean');f,ax=plt.subplots(figsize=(8,4.6));ax.errorbar(cc.cv_ap_mean,cc.model,xerr=cc.cv_ap_sd,fmt='o',markersize=7,capsize=4,color=blue,ecolor='#b17a5b');ax.set_xlim(.6,1.02);ax.set_xlabel('Training three-fold grouped CV: mean AP ± SD');ax.set_title('Grouped CV stability (SD, not a 95% confidence interval)');save(f,'21_cross_validation_stability.png')
# 22 validation vs test paired slope
vv=V.set_index('model');tt=T.set_index('model');order=list(V.model)
f,ax=plt.subplots(figsize=(8.4,4.8));n=np.arange(len(order));ax.scatter(vv.loc[order,'pr_auc_average_precision'],n,s=52,label='Validation',marker='o',c=blue);ax.scatter(tt.loc[order,'pr_auc_average_precision'],n,s=54,label='Locked test',marker='s',c=orange);[ax.plot([vv.loc[m,'pr_auc_average_precision'],tt.loc[m,'pr_auc_average_precision']],[i,i],color='#a5a5a5',linewidth=1) for i,m in enumerate(order)];ax.set_yticks(n,order);ax.invert_yaxis();ax.set_xlim(.62,1.01);ax.set_xlabel('Average precision (AP)');ax.set_title('Validation vs. test AP (validation-only model selection)');ax.legend();save(f,'22_validation_test_comparison.png')
# 23 ablation horizontal bars
f,ax=plt.subplots(figsize=(7.9,3.9));aa=A.iloc[::-1];ax.barh(aa.ablation,aa.validation_ap,color=[orange if x=='minus_labs' else blue for x in aa.ablation]);ax.set_xlim(0,1.05);ax.set_xlabel('Validation AP');ax.set_title('Feature-group ablation: contribution of laboratory values');[ax.text(t+.009,i,f'{t:.3f}',va='center') for i,t in enumerate(aa.validation_ap)];save(f,'23_feature_group_ablation.png')
# 24 importance bar
z=I.head(16).iloc[::-1];f,ax=plt.subplots(figsize=(7.2,5.4));ax.barh(z.feature,z.ap_decrease,xerr=z.sd,color=blue,error_kw={'capsize':2});ax.set_xlabel('Validation AP decrease (mean ± SD; 5 permutations)');ax.set_title('LightGBM permutation importance');save(f,'24_permutation_importance.png')
# 25 SHAP bar exists; use in atlas and paper
# 26 all model test CI bootstrap AP (descriptive)
# 27 clinical subgroups confusion sensitivity by sex and age for final best fixed predictions
P=pd.read_csv(RES/'locked_test_predictions.csv');DI=D.iloc[P.row_index.to_numpy()].reset_index(drop=True).copy();DI['y']=P.true_label.to_numpy();DI['pred']=P.predicted.to_numpy();DI['age_group']=pd.cut(DI.age,[-np.inf,30,50,70,np.inf],labels=['≤30','31–50','51–70','>70']);grp=[]
for typ,k in [('Age group','age_group'),('Sex','sex'),('Referral source','referral_source')]:
 for v,g in DI.groupby(k,dropna=False,observed=True):
  if len(g)<25:continue
  npos=(g.y==1).sum();nneg=(g.y==0).sum();sens=((g.pred==1)&(g.y==1)).sum()/npos if npos else np.nan;spec=((g.pred==0)&(g.y==0)).sum()/nneg if nneg else np.nan
  grp.append({'stratum':typ,'category':str(v),'n':len(g),'n_positive':npos,'sensitivity':sens,'specificity':spec})
groups=pd.DataFrame(grp);groups.to_csv(RES/'test_subgroup_descriptive.csv',index=False);sel=groups[(groups.stratum.isin(['Age group','Sex']))&(groups.n_positive>=15)];f,ax=plt.subplots(figsize=(8.6,4.5));x=np.arange(len(sel));ax.bar(x-.18,sel.sensitivity,.35,color=orange,label='Sensitivity');ax.bar(x+.18,sel.specificity,.35,color=blue,label='Specificity');ax.set_xticks(x,[x+'\n(n='+str(n)+')' for x,n in zip(sel.category,sel.n)]);ax.set_ylim(0,1.05);ax.set_ylabel('Proportion');ax.set_title('Selected model: descriptive test subgroups by age and sex');ax.legend();save(f,'25_test_age_sex_subgroups.png')
# 28 labels co-occur
codes=list('ABCDEFGHIJKLMNOPQRST');mat=np.zeros((20,20),int)
for s in D.diagnosis:
 zz=set(s.replace('|','').replace('-',''))
 for a in zz:
  for b in zz:
   if a in codes and b in codes:mat[codes.index(a),codes.index(b)]+=1
order=np.argsort(-np.diag(mat))[:14];f,ax=plt.subplots(figsize=(7.1,5.9));sns.heatmap(np.log1p(mat[np.ix_(order,order)]),xticklabels=np.array(codes)[order],yticklabels=np.array(codes)[order],cmap='Blues',square=True,ax=ax,cbar_kws={'label':'log(1 + co-occurrence count)'},linewidths=.3);ax.set_title('Diagnosis code co-occurrence (overlapping labels)');save(f,'26_diagnosis_code_cooccurrence.png')
# 29 target split stacked proportions
merged=S.merge(D[['target']],left_on='row_index',right_index=True);xt=pd.crosstab(merged.split,merged.target).reindex(['train','validation','test']);z=xt.div(xt.sum(axis=1),axis=0);f,ax=plt.subplots(figsize=(6.7,4));z.plot(kind='bar',stacked=True,color=[blue,orange],ax=ax,rot=0);ax.set_ylabel('Within-split proportion');ax.set_title('Target proportions across train, validation, and test');ax.legend(['No additional code','Additional code present'],loc='center left',bbox_to_anchor=(1,.5));save(f,'27_split_target_proportions.png')
# 30 model AP/F1 scatter
f,ax=plt.subplots(figsize=(7.2,5.3));
for j,r in enumerate(T.itertuples()):ax.scatter(r.pr_auc_average_precision,r.f1,s=85,color=COLORS[j%len(COLORS)]);ax.annotate(r.model,(r.pr_auc_average_precision,r.f1),xytext=(4,5),textcoords='offset points',fontsize=8)
ax.set_xlabel('AP');ax.set_ylabel('F1 (validation-selected threshold)');ax.set_title('Discrimination and classification performance (test)');save(f,'28_test_ap_f1_scatter.png')
# threshold sweep ONLY validation (re-fit exactly fixed chosen model, not using test)
X=D[COLS];tr=merged.loc[merged.split=='train','row_index'].to_numpy();va=merged.loc[merged.split=='validation','row_index'].to_numpy();yy=D.target.to_numpy();from sklearn.pipeline import Pipeline
best='LightGBM';mdl=Pipeline([('prep',preprocessor(COLS)),('clf',estimator_fns(yy[tr])[best]())]);mdl.fit(X.iloc[tr],yy[tr]);vp=mdl.predict_proba(X.iloc[va])[:,1];ths=np.arange(.05,.951,.01);thdf=pd.DataFrame([{'threshold':t,'f1':f1_score(yy[va],vp>=t,zero_division=0),'sensitivity':recall_score(yy[va],vp>=t,zero_division=0),'precision':precision_score(yy[va],vp>=t,zero_division=0)} for t in ths]);thdf.to_csv(RES/'validation_threshold_sweep.csv',index=False)
f,ax=plt.subplots(figsize=(7.8,4.4));
for col,c in [('f1',orange),('sensitivity',blue),('precision','#6a9c81')]:ax.plot(thdf.threshold,thdf[col],label=col,color=c)
ax.axvline(float(V.iloc[0].threshold),ls='--',color='#888',label=f'Validation-selected threshold {float(V.iloc[0].threshold):.2f}');ax.set_ylim(.72,1.0);ax.set_xlim(.05,.95);ax.set_xlabel('Validation classification threshold');ax.set_ylabel('Metric');ax.set_title('Threshold selection using validation only');ax.legend();save(f,'29_validation_threshold_sensitivity.png')
# save publication summary
summary={'n':len(D),'target_count':D.target.value_counts().to_dict(),'important':I.head(6)[['feature','ap_decrease']].to_dict('records'),'top_numeric_fdr':rank.sort_values('q_bh').head(6).to_dict('records'),'pca_sample_n':len(inds),'age_range':[float(D.age.min()),float(D.age.max())],'TBG_missing':float(D.TBG.isna().mean()),'T3_missing':float(D.T3.isna().mean()),'shap_available':(FIG/'09_optional_validation_shap.png').exists()};(RES/'publication_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print('Done figures',len(list(FIG.glob('*.png'))),flush=True)
