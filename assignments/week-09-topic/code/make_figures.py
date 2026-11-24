"""Reproducible, publication-style figures for the ensemble-learning assignment.

All empirical plots read saved results. Conceptual illustrations are labelled.
Export: vector PDF and 600-dpi PNG for the course reports.
"""
from pathlib import Path
import argparse
import json
import string
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, precision_recall_curve

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = ROOT/'results'
OUT = ROOT/'figures'
DATA = ROOT/'data'
COL = {'single_tree':'#777777','deep_tree':'#B6B6B6','bagging':'#D98B31',
       'random_forest':'#008A78','gradient_boosting':'#2674A5',
       'adaboost':'#C2A225','logistic':'#7662A0','rbf_svm':'#C76E93'}
NAME = {'single_tree':'Tree (d = 4)','deep_tree':'Tree (full)',
        'bagging':'Bagging','random_forest':'Random forest','gradient_boosting':'Gradient boost.',
        'adaboost':'AdaBoost','logistic':'Logistic reg.','rbf_svm':'RBF-SVM'}
ORDER=['single_tree','bagging','random_forest','adaboost','gradient_boosting','logistic','rbf_svm']
BLUE='#2674A5'; TEAL='#008A78'; ORANGE='#D98B31'; GRAY='#777777'; LIGHT='#E6E8EB'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
    'axes.titlesize':9,'axes.titleweight':'normal','axes.linewidth':.7,
    'xtick.labelsize':8,'ytick.labelsize':8,'xtick.major.width':.7,'ytick.major.width':.7,
    'xtick.major.size':3,'ytick.major.size':3,'legend.fontsize':8,'legend.frameon':False,
    'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
    'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,
    'axes.spines.right':False,'lines.linewidth':1.4,'lines.markersize':4,
    'figure.constrained_layout.use':False})

def save(fig,name):
    OUT.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','png']:
        fig.savefig(OUT/f'{name}.{ext}',dpi=600 if ext=='png' else None)
    plt.close(fig)
    print('Saved',name,flush=True)

def panel(ax, letter, title):
    ax.set_title(title,loc='left',pad=10)
    ax.text(-.12,1.055,letter,transform=ax.transAxes,fontweight='bold',fontsize=12,va='bottom')

def clean(ax,grid='x'):
    if grid: ax.grid(axis=grid,color='#E8EAED',lw=.6,zorder=0)
    ax.set_axisbelow(True)

def box(ax,xy,w,h,text,color=BLUE,fontsize=8,fill='#F7F9FA'):
    ax.add_patch(FancyBboxPatch(xy,w,h,boxstyle='round,pad=0.012,rounding_size=0.025',
       edgecolor=color,facecolor=fill,lw=.85))
    ax.text(xy[0]+w/2,xy[1]+h/2,text,ha='center',va='center',fontsize=fontsize,color='#26313A')

def arrow(ax,a,b,color='#58636B',lw=1.1):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=8,lw=lw,color=color))

def plot_design():
    fig=plt.figure(figsize=(7.09,5.3),layout='constrained')
    gs=fig.add_gridspec(2,2,height_ratios=[1,.96],hspace=.13,wspace=.13)
    ax=fig.add_subplot(gs[0,0]); panel(ax,'a','Fixed data partitions')
    for i,(folder,label,tag) in enumerate([('wdbc','malignant','WDBC'),('ctg','pathologic','CTG')]):
        for j,(part,partname) in enumerate([('train','Train'),('validation','Validation'),('locked-test-features','Locked test')]):
            df=pd.read_csv(DATA/folder/'student'/f'{part}.csv')
            y=5-(i*3+j)
            if label in df:
                pos=df[label].sum(); neg=len(df)-pos
                ax.barh(y,neg/len(df),height=.62,color=BLUE)
                ax.barh(y,pos/len(df),left=neg/len(df),height=.62,color=ORANGE)
                text=f'{len(df):,} rows / {int(pos)} positive'
            else:
                ax.barh(y,1,height=.62,color='#CDD2D8',hatch='///',edgecolor='white')
                text=f'{len(df):,} rows / labels unopened'
            ax.text(.02,y,text,va='center',fontsize=7.6,color='white' if label in df else '#3B4148')
    ax.set_yticks([5,4,3,2,1,0],['WDBC · train','WDBC · validation','WDBC · locked','CTG · train','CTG · validation','CTG · locked'])
    ax.set_xlim(0,1);ax.set_xticks([0,.5,1],['0','0.5','1']);ax.set_xlabel('Fraction of records')
    ax.text(0,-.25,'Blue = negative; orange = positive',transform=ax.transAxes,fontsize=8)
    ax.spines[['left','bottom']].set_visible(False);ax.tick_params(axis='y',length=0)
    ax=fig.add_subplot(gs[0,1]);panel(ax,'b','Evaluation protocol');ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
    box(ax,(.06,.75),.88,.19,'Development = train + validation\nWDBC: 455 · CTG: 1,703')
    arrow(ax,(.5,.74),(.5,.65))
    box(ax,(.06,.42),.88,.21,'Shared 5-fold cross-validation\nWDBC: stratified · CTG: grouped',TEAL)
    arrow(ax,(.5,.41),(.5,.33))
    box(ax,(.06,.08),.88,.23,'Train → fixed validation\nMechanisms and sensitivity analyses\nNo locked-test evaluation',ORANGE)
    ax=fig.add_subplot(gs[1,0]);panel(ax,'c','Bagging / random forest (schematic)');ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
    box(ax,(.28,.79),.44,.14,'Training records',BLUE)
    for x in [.05,.37,.69]:
        arrow(ax,(.5,.78),(x+.13,.64))
        box(ax,(x,.45),.25,.18,'Bootstrap\nsample',BLUE)
        arrow(ax,(x+.125,.44),(x+.125,.35))
        box(ax,(x,.17),.25,.17,'Tree',TEAL)
        arrow(ax,(x+.125,.16),(.5,.05))
    ax.text(.5,.015,'Average probabilities',ha='center',va='center',fontsize=9,fontweight='bold')
    # Random-feature subsets are described in the caption to keep the flow clear.
    ax=fig.add_subplot(gs[1,1]);panel(ax,'d','Boosting (schematic)');ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
    for k,x in enumerate([.02,.37,.72]):
        box(ax,(x,.53),.26,.19,f'Tree {k+1}',ORANGE,8)
        if k<2:arrow(ax,(x+.27,.62),(x+.34,.62))
    ax.text(.5,.83,'Each stage depends on previous predictions',ha='center',fontsize=8)
    ax.text(.5,.38,'AdaBoost: update sample weights\nGradient boosting: fit negative gradients',ha='center',va='center',fontsize=8.5,linespacing=1.65)
    box(ax,(.06,.08),.88,.15,'Fₘ(x) = Fₘ₋₁(x) + η hₘ(x)',ORANGE,10)
    save(fig,'Fig01_design')

# Empirical panels are defined below; data files are generated by run_experiments.py.

def read(ds,name):
    return pd.read_csv(RESULTS/ds/name)

def plot_tree_bootstrap():
    d=read('wdbc','tree-depth.csv'); roots=read('wdbc','bootstrap-roots.csv')
    sim=read('wdbc','bootstrap-simulation.csv'); su=read('wdbc','bootstrap-simulation-summary.csv')
    fig,axs=plt.subplots(2,2,figsize=(7.09,5.3),layout='constrained')
    ax=axs[0,0];panel(ax,'a','Single-tree depth and discrimination')
    depths=['1','2','3','4','unlimited']
    for split,col,style in [('train',GRAY,'--'),('validation',BLUE,'-')]:
        s=d[d.split==split].set_index('max_depth').reindex(depths)
        ax.plot(range(5),s.roc_auc,marker='o',color=col,ls=style,label=split.title())
    ax.set_xticks(range(5),['1','2','3','4','Full']);ax.set(xlabel='Maximum tree depth',ylabel='ROC-AUC');ax.legend(loc='lower right');clean(ax,'y')
    ax=axs[0,1];panel(ax,'b','Root split after training resampling')
    rt=roots.groupby('feature').agg(count=('refit','size'),lo=('threshold','min'),hi=('threshold','max')).sort_values('count')
    yy=np.arange(len(rt));ax.barh(yy,rt['count'],height=.5,color=BLUE,alpha=.85)
    ax.set_yticks(yy,[s.replace('worst ','worst\n').replace('mean ','mean\n') for s in rt.index],fontsize=8)
    for i,(_,r) in enumerate(rt.iterrows()):
        ax.text(r['count']+.12,i,f"{int(r['count'])}/10\ncut: {r.lo:.3g}–{r.hi:.3g}",va='center',fontsize=7.5)
    ax.set(xlim=(0,max(rt['count'])+3),xlabel='Number of depth-2 fits')
    ax.set_xticks(range(0,int(max(rt['count']))+1,2));clean(ax)
    for ax,nn,letter in [(axs[1,0],5,'c'),(axs[1,1],341,'d')]:
        panel(ax,letter,f'Bootstrap coverage: n = {nn}')
        vals=sim.loc[sim.n==nn,'unique_fraction']; th=su.loc[su.n==nn,'theory_unique_fraction'].iloc[0]
        bins=np.arange(.1,1.11,.2) if nn==5 else np.linspace(.55,.72,25)
        ax.hist(vals,bins=bins,color=TEAL,alpha=.8,edgecolor='white',linewidth=.7)
        ax.axvline(th,color=ORANGE,ls='--',lw=1.4,label=f'Theory: {th:.4f}')
        ax.axvline(vals.mean(),color='#27323A',lw=1,label=f'Simulation: {vals.mean():.4f}')
        ax.set(xlabel='Fraction of unique records',ylabel='Number of draws')
        ax.legend(loc='upper left',fontsize=7.5);clean(ax,'y')
    save(fig,'Fig02_tree_bootstrap')

def cv_panel(ax,df,metric,letter,title,order=ORDER,xlim=None):
    panel(ax,letter,title)
    y=np.arange(len(order));offset=np.linspace(-.11,.11,5)
    for i,m in enumerate(order):
        v=df.loc[df.model==m,metric].to_numpy()
        ax.scatter(v,i+offset[:len(v)],s=10,color=COL[m],alpha=.55,zorder=3,edgecolors='none')
        ax.errorbar(np.mean(v),i,xerr=np.std(v,ddof=1),fmt='o',ms=4,color=COL[m],capsize=2,lw=1.25,zorder=4)
    ax.set_yticks(y,[NAME[m] for m in order]);ax.invert_yaxis()
    if xlim:ax.set_xlim(xlim)
    clean(ax)

def plot_wdbc_comparison():
    cv=read('wdbc','cv-fold-metrics.csv'); p=read('wdbc','validation-predictions.csv');y=p.malignant
    fig=plt.figure(figsize=(7.09,5.3),layout='constrained');gs=fig.add_gridspec(2,6,height_ratios=[1.12,1],hspace=.15)
    for j,(metric,title,xlim) in enumerate([('roc_auc','CV: ROC-AUC',(.85,1.005)),('average_precision','CV: AP',(.75,1.005)),('balanced_accuracy','CV: balanced accuracy',(.79,1.005))]):
        ax=fig.add_subplot(gs[0,j*2:j*2+2]);cv_panel(ax,cv,metric,string.ascii_lowercase[j],title,xlim=xlim)
        if j>0:ax.set_yticklabels([])
        ax.set_xlabel('Mean ± SD')
    for j,kind in enumerate(['roc','pr']):
        ax=fig.add_subplot(gs[1,j*3:j*3+3]);panel(ax,'de'[j],'Validation: ROC' if kind=='roc' else 'Validation: precision–recall')
        for m in ORDER:
            if kind=='roc':xx,yy,_=roc_curve(y,p[m+'_score'])
            else:yy,xx,_=precision_recall_curve(y,p[m+'_score'])
            ax.plot(xx,yy,color=COL[m],label=NAME[m],lw=1.4,alpha=.95)
        if kind=='roc':
            ax.plot([0,1],[0,1],ls=':',color='#AAB0B6',lw=.9)
            ax.set(xlabel='False-positive rate',ylabel='True-positive rate',xlim=(-.01,1.01),ylim=(-.01,1.02))
        else:
            ax.axhline(y.mean(),color='#AAB0B6',ls=':',lw=.9)
            ax.text(.03,y.mean()+.03,f'Prevalence = {y.mean():.3f}',fontsize=7,color='#71787F')
            ax.set(xlabel='Recall',ylabel='Precision',xlim=(-.01,1.01),ylim=(-.01,1.02))
        clean(ax,None)
        if j==0:ax.legend(loc='lower right',fontsize=8,ncol=1,handlelength=1.4,labelspacing=.3)
    save(fig,'Fig03_wdbc_comparison')

def plot_aggregation():
    d=read('wdbc','tree-count.csv'); cube=np.load(RESULTS/'wdbc'/'individual-tree-predictions.npz')
    fig,axs=plt.subplots(2,2,figsize=(7.09,5.3),layout='constrained')
    for j,(metric,title) in enumerate([('roc_auc','Validation discrimination'),('balanced_accuracy','Thresholded performance')]):
        ax=axs[0,j];panel(ax,'ab'[j],title)
        for m in ['bagging','random_forest']:
            s=d[d.model==m];ax.plot(s.n_trees,s[metric],marker='o',color=COL[m],label=NAME[m])
        ax.set_xscale('log');ax.set_xticks([1,10,50,100,300],[1,10,50,100,300]);ax.set_xlabel('Number of trees (log scale)')
        ax.set_ylabel('ROC-AUC' if metric=='roc_auc' else 'Balanced accuracy');ax.legend(loc='lower right',fontsize=8);clean(ax,'y')
    ax=axs[1,0];panel(ax,'c','Tree-error correlation at B = 100')
    for m in ['bagging','random_forest']:
        err=cube[m+'_error'];err=err[err.std(axis=1)>0];corr=np.corrcoef(err);v=corr[np.triu_indices(len(corr),1)]
        ax.hist(v,bins=np.linspace(-.25,1,26),histtype='step',density=True,lw=1.4,color=COL[m],label=f'{NAME[m]}: mean {v.mean():.3f}')
        ax.axvline(v.mean(),color=COL[m],ls='--',lw=.8)
    ax.set(xlabel='Pairwise correlation of error indicators',ylabel='Density');ax.legend(loc='upper left',fontsize=7);clean(ax,'y')
    ax=axs[1,1];panel(ax,'d','Variance formula (illustration)')
    B=np.arange(1,301)
    for rho,col in zip([0,.1,.5,.9],[BLUE,TEAL,ORANGE,'#9C4F7B']):
        ax.plot(B,rho+(1-rho)/B,color=col,label=f'ρ = {rho:g}')
    ax.set_xscale('log');ax.set(xlabel='Number of models B (log scale)',ylabel='Var(mean prediction) / σ²',ylim=(0,1.02));ax.set_xticks([1,10,100,300],[1,10,100,300])
    ax.text(.04,.95,'ρ + (1 − ρ) / B',transform=ax.transAxes,va='top',fontsize=9)
    ax.legend(loc='center right',fontsize=7.5);clean(ax,'y')
    save(fig,'Fig04_aggregation')

def paired_panel(ax,ds,metric,letter,title):
    d=read(ds,'validation-paired-differences.csv');d=d[d.metric==metric]
    panel(ax,letter,title)
    for i,(_,r) in enumerate(d.iterrows()):
        other=r.contrast.split(' - ')[1];color=COL[other]
        ax.plot([r.lower,r.upper],[i,i],color=color,lw=1.4)
        ax.plot([r.lower,r.lower],[i-.07,i+.07],color=color,lw=.9)
        ax.plot([r.upper,r.upper],[i-.07,i+.07],color=color,lw=.9)
        ax.plot(r.estimate,i,'o',color=color,ms=4)
    ax.set_yticks(range(len(d)),['RF − '+NAME[s.split(' - ')[1]].replace('Gradient boost.','GB').replace('Logistic reg.','LR') for s in d.contrast])
    ax.axvline(0,color='#9DA4AA',ls='--',lw=.8);ax.invert_yaxis();ax.set_xlabel('Difference (95% bootstrap interval)');clean(ax)

def stability_panel(ax,ds,letter,title):
    d=read(ds,'stability-per-sample.csv');order=['single_tree','deep_tree','bagging','random_forest','gradient_boosting','logistic']
    panel(ax,letter,title)
    arr=[d[m+'_prob_sd'].to_numpy() for m in order]
    b=ax.boxplot(arr,vert=False,tick_labels=[NAME[m] for m in order],widths=.45,showfliers=False,
       patch_artist=True,medianprops={'color':'#20272C','lw':1.1},whiskerprops={'lw':.8},capprops={'lw':.8},boxprops={'lw':.8})
    for patch,m in zip(b['boxes'],order):patch.set_facecolor(COL[m]);patch.set_alpha(.3);patch.set_edgecolor(COL[m])
    for i,(m,v) in enumerate(zip(order,arr),1):
        ax.scatter(v,np.full(len(v),i)+np.random.default_rng(100+i).uniform(-.13,.13,len(v)),s=4,c=COL[m],alpha=.3,lw=0)
        ax.plot(v.mean(),i,'D',color=COL[m],ms=3)
    ax.invert_yaxis();ax.set_xlabel('Prediction SD (30 refits)');clean(ax)

def flip_panel(ax,ds,letter,title):
    d=read(ds,'stability-summary.csv').set_index('model');order=['single_tree','deep_tree','bagging','random_forest','gradient_boosting','logistic'];panel(ax,letter,title)
    for i,m in enumerate(order):
        ax.plot([d.loc[m,'mean_class_flip_vs_full_train'],d.loc[m,'fraction_samples_any_class_change']],[i,i],color=COL[m],alpha=.5,lw=1)
        ax.scatter(d.loc[m,'mean_class_flip_vs_full_train'],i,s=22,color=COL[m],marker='o')
        ax.scatter(d.loc[m,'fraction_samples_any_class_change'],i,s=24,facecolor='white',edgecolor=COL[m],marker='s',lw=1)
    ax.set_yticks(range(len(order)),[NAME[m] for m in order]);ax.invert_yaxis();ax.set_xlabel('Fraction');clean(ax)
    ax.set_xlabel('Fraction (● mean flip; □ any change)',fontsize=8)

def plot_stability():
    fig,axs=plt.subplots(2,2,figsize=(7.09,5.3),layout='constrained')
    stability_panel(axs[0,0],'wdbc','a','Training-resample stability')
    flip_panel(axs[0,1],'wdbc','b','Two measures of class change')
    paired_panel(axs[1,0],'wdbc','roc_auc','c','Validation: Δ ROC-AUC')
    paired_panel(axs[1,1],'wdbc','average_precision','d','Validation: Δ AP')
    save(fig,'Fig05_stability_uncertainty')

def plot_boosting():
    d=read('wdbc','boosting-stages.csv');fig,axs=plt.subplots(2,3,figsize=(7.09,5.1),layout='constrained')
    for j,(rate,col) in enumerate(zip([.03,.1,.3],[BLUE,TEAL,ORANGE])):
        for k,(metric,label) in enumerate([('log_loss','Log loss'),('roc_auc','ROC-AUC')]):
            ax=axs[k,j];panel(ax,string.ascii_lowercase[k*3+j],f'Learning rate η = {rate:g}')
            for split,ls,alpha in [('train','--',.5),('validation','-',1.)]:
                s=d[(d.learning_rate==rate)&(d.split==split)]
                ax.plot(s.stage,s[metric],color=col,ls=ls,alpha=alpha,label=split.title())
            if k==0:
                s=d[(d.learning_rate==rate)&(d.split=='validation')];row=s.loc[s.log_loss.idxmin()]
                ax.plot(row.stage,row.log_loss,'o',color=col,ms=4)
                ax.text(.97,.93,f'Min. validation loss\nat stage {int(row.stage)}',transform=ax.transAxes,ha='right',va='top',fontsize=7.1)
                ax.set_ylim(0,max(.15,d[d.learning_rate==rate].log_loss.max()*1.07))
            else:ax.set_ylim(max(0,d[d.learning_rate==rate].roc_auc.min()-.007),1.002)
            ax.set_xlabel('Boosting stage');ax.set_xticks([1,100,200,300]);clean(ax,'y')
            if j==0:ax.set_ylabel(label)
            if j==0 and k==0:ax.legend(loc='center right',fontsize=7)
    save(fig,'Fig06_boosting')

def shortfeature(f):
    return f.replace('mean ','mean ').replace('worst ','worst ').replace(' error',' SE').replace('concave points','concave pts').replace('fractal dimension','fractal dim.')

def plot_importance():
    d=read('wdbc','feature-importance.csv'); corr=pd.read_csv(RESULTS/'wdbc'/'train-spearman.csv',index_col=0)
    selected=set(d.nlargest(8,'mdi').feature) | set(d.nlargest(3,'permutation_auc_drop_mean').feature)
    top=d[d.feature.isin(selected)].sort_values('mdi')
    nfeat=len(top)
    feats=top.feature.to_list();labels=[shortfeature(x) for x in feats]
    fig=plt.figure(figsize=(7.09,5.3),layout=None)
    ax=fig.add_axes([.19,.53,.45,.39]);panel(ax,'a','Correlation of selected features')
    mat=corr.loc[feats,feats].to_numpy();im=ax.imshow(mat,cmap='RdBu_r',vmin=-1,vmax=1,aspect='equal')
    ax.set_xticks(range(nfeat),range(1,nfeat+1));ax.set_yticks(range(nfeat),[f'{i+1}. '+x for i,x in enumerate(labels)],fontsize=8)
    ax.tick_params(length=0);ax.spines[['left','bottom']].set_visible(False)
    ax.set_xlabel('Feature number (same order as rows)',fontsize=8)
    cax=fig.add_axes([.665,.56,.018,.31]);cb=fig.colorbar(im,cax=cax);cb.set_label('Spearman correlation',fontsize=8)
    ax=fig.add_axes([.19,.10,.31,.29]);panel(ax,'b','Training impurity importance')
    ax.barh(range(nfeat),top.mdi,color=TEAL,height=.6);ax.set_yticks(range(nfeat),[str(i) for i in range(1,nfeat+1)],fontsize=8)
    ax.set_ylabel('Feature number');ax.set_xlabel('Normalized MDI',fontsize=8);clean(ax)
    ax=fig.add_axes([.64,.10,.31,.29]);panel(ax,'c','Validation permutation importance')
    ax.errorbar(top.permutation_auc_drop_mean,range(nfeat),xerr=top.permutation_auc_drop_sd,fmt='o',color=BLUE,capsize=2,ms=4,lw=1)
    ax.set_yticks(range(nfeat),[str(i) for i in range(1,nfeat+1)],fontsize=8);ax.axvline(0,color=GRAY,ls='--',lw=.8)
    ax.set_xlabel('AUC decrease (mean ± shuffle SD)',fontsize=8);clean(ax)
    save(fig,'Fig07_feature_importance')


def plot_ctg():
    cv=read('ctg','cv-fold-metrics.csv')
    fig=plt.figure(figsize=(7.09,5.3),layout=None)
    for j,(metric,title,xlim) in enumerate([('roc_auc','Group CV: ROC-AUC',(.6,1.01)),('average_precision','Group CV: AP',(.4,1.01)),('balanced_accuracy','Group CV: BA',(.6,1.01))]):
        ax=fig.add_axes([.18+j*.27,.59,.22,.32]);cv_panel(ax,cv,metric,string.ascii_lowercase[j],title,xlim=xlim)
        if j>0:ax.set_yticklabels([])
        ax.set_xlabel('Mean ± SD',fontsize=8)
    ax=fig.add_axes([.18,.12,.29,.30]);stability_panel(ax,'ctg','d','Recording-resample stability')
    ax=fig.add_axes([.68,.12,.28,.30]);paired_panel(ax,'ctg','average_precision','e','Validation: Δ AP')
    ax.set_xlabel('Difference (95% bootstrap interval)',fontsize=8)
    save(fig,'Fig08_ctg_grouped')


def plot_adaboost_weights():
    from matplotlib.colors import LogNorm
    stages=read('wdbc','adaboost-stage-weights.csv');trace=read('wdbc','adaboost-sample-trace.csv')
    fig=plt.figure(figsize=(7.09,3.5),layout='constrained');gs=fig.add_gridspec(1,2,width_ratios=[.85,1.3],wspace=.1)
    ax=fig.add_subplot(gs[0,0]);panel(ax,'a','SAMME weight redistribution')
    ax.plot(stages.stage,stages.misclassified_weight_mass_before,'o-',color=BLUE,label='Before update')
    ax.plot(stages.stage,stages.misclassified_weight_mass_after,'s--',color=ORANGE,label='After update')
    ax.set(xlabel='Stage',ylabel='Weight mass on current errors',ylim=(0,.6));ax.set_xticks([1,3,5,7,10]);ax.legend(loc='lower right',fontsize=8);clean(ax,'y')
    ax=fig.add_subplot(gs[0,1]);panel(ax,'b','Eight illustrative training records')
    ids=trace[trace.stage==1].sample_id.to_list();m=trace.pivot(index='sample_id',columns='stage',values='weight_after').loc[ids]*341
    wrong=trace.pivot(index='sample_id',columns='stage',values='wrong_this_stage').loc[ids]
    im=ax.imshow(m.to_numpy(),aspect='auto',cmap='viridis',norm=LogNorm(vmin=.02,vmax=50))
    for i in range(len(ids)):
        for j in range(10):
            if wrong.iloc[i,j]:ax.text(j,i,'×',ha='center',va='center',color='white',fontsize=9)
    ax.set_xticks(range(10),range(1,11));ax.set_yticks(range(8),ids,fontsize=8)
    ax.set_xlabel('Stage (×: misclassified at this stage)',fontsize=8);ax.tick_params(length=0)
    cb=fig.colorbar(im,ax=ax,pad=.025,shrink=.8);cb.set_label('Weight / initial uniform weight',fontsize=8)
    save(fig,'FigS01_adaboost_weights')


def main():
    global RESULTS,OUT,DATA
    ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,default=RESULTS);ap.add_argument('--out',type=Path,default=OUT)
    ap.add_argument('--data-root',type=Path,default=DATA);ap.add_argument('--only',nargs='*',type=int)
    args=ap.parse_args();RESULTS=args.results;OUT=args.out;DATA=args.data_root
    fns=[plot_design,plot_tree_bootstrap,plot_wdbc_comparison,plot_aggregation,plot_stability,plot_boosting,plot_importance,plot_ctg,plot_adaboost_weights]
    for i,fn in enumerate(fns,1):
        if not args.only or i in args.only:fn()
if __name__=='__main__':main()
