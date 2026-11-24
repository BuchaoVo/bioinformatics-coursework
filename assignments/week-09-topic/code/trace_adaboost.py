"""Reconstruct and verify actual sklearn SAMME training weights; no tuning.

The fitted 50-stump ensemble is identical to the WDBC baseline model. Its first
10 stages are reconstructed from its fitted stump predictions, with each
weighted error checked against estimator_errors_. No teacher files are read.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import AdaBoostClassifier

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-root',type=Path,default=Path(__file__).resolve().parents[1] / 'data')
    ap.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parents[1] / 'results')
    args=ap.parse_args(); out=args.output_root/'wdbc'; out.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(args.input_root/'wdbc/student/train.csv')
    feats=[x for x in d if x not in {'sample_id','malignant'}]
    X=d[feats].to_numpy(); y=d.malignant.to_numpy()
    model=AdaBoostClassifier(n_estimators=50,learning_rate=1.,random_state=1010).fit(X,y)
    first_wrong=model.estimators_[0].predict(X)!=y
    selected=np.r_[np.flatnonzero(first_wrong)[:4],np.flatnonzero(~first_wrong)[:4]]
    w=np.ones(len(y))/len(y); rows=[]; trace=[]; all_weights=[]
    for k,estimator in enumerate(model.estimators_[:10]):
        # sklearn clips positive weights at machine epsilon before each fit.
        w=np.clip(w,np.finfo(w.dtype).eps,None)
        incorrect=estimator.predict(X)!=y
        error=np.average(incorrect,weights=w)
        alpha=model.estimator_weights_[k]
        assert np.isclose(error,model.estimator_errors_[k],atol=1e-12,rtol=1e-12)
        assert np.isclose(alpha,np.log((1-error)/error),atol=1e-12,rtol=1e-12)
        nxt=np.exp(np.log(w)+alpha*incorrect); nxt/=nxt.sum()
        rows.append({'stage':k+1,'weighted_error':error,'sklearn_stored_error':model.estimator_errors_[k],
          'alpha_samme':alpha,'wrong_to_correct_weight_multiplier_ratio':np.exp(alpha),
          'misclassified_weight_mass_before':w[incorrect].sum()/w.sum(),
          'misclassified_weight_mass_after':nxt[incorrect].sum(),
          'n_wrong':int(incorrect.sum()),
          'mean_wrong_next_over_current':np.mean(nxt[incorrect]/w[incorrect]),
          'mean_correct_next_over_current':np.mean(nxt[~incorrect]/w[~incorrect]),
          'stump_feature':feats[estimator.tree_.feature[0]],'stump_threshold':estimator.tree_.threshold[0]})
        for i in selected:
            trace.append({'stage':k+1,'sample_id':d.sample_id.iloc[i],'label':y[i],
               'first_stage_wrong':bool(first_wrong[i]),'wrong_this_stage':bool(incorrect[i]),
               'weight_before':w[i],'weight_after':nxt[i],'weight_ratio_after_before':nxt[i]/w[i]})
        all_weights.append(w.copy()); w=nxt
    pd.DataFrame(rows).to_csv(out/'adaboost-stage-weights.csv',index=False)
    pd.DataFrame(trace).to_csv(out/'adaboost-sample-trace.csv',index=False)
    np.savez_compressed(out/'adaboost-all-training-weights.npz',sample_id=d.sample_id.to_numpy(),weight_before=np.array(all_weights))
    note={
        'sklearn_version':sklearn.__version__,'algorithm':'Discrete SAMME, binary labels 0/1',
        'model':'AdaBoostClassifier(n_estimators=50,learning_rate=1,random_state=1010), default depth-1 stumps',
        'stages_recorded':10,'all_recorded_errors_match_fitted_model':True,'selected_samples':'First four first-stage misclassified rows and first four correctly classified rows, by supplied training order.',
        'update':'alpha=ln((1-e)/e); unnormalized w_next=w*exp(alpha*I[wrong]); normalize all weights.',
        'relation_to_classic_binary_adaboost':'With labels -1/+1, classic alpha=0.5*ln((1-e)/e) and update w*exp(-alpha*y*h). After normalization, its relative wrong/correct multiplier exp(2*alpha) equals exp(alpha_SAMME); the normalized weights agree.',
        'interpretation':'After a full learning_rate=1 binary SAMME update, current stump mistakes jointly carry weight 0.5. A particular row can lose weight when later classified correctly. Weight measures training attention, not disease risk.',
        'test_labels_read':False
    }
    (out/'adaboost-trace-notes.json').write_text(json.dumps(note,indent=2,ensure_ascii=False))
    print(pd.DataFrame(rows).round(6).to_string(index=False))

if __name__=='__main__': main()
