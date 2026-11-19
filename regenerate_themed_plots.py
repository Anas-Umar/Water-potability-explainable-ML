"""Regenerate themed ROC and SHAP plots using the main analysis model-selection rule.

The selected SHAP model is the model with the higher mean 5-fold training CV ROC-AUC.
All imputation is fit on the outer training data only. SHAP plots explain the fitted
model on the held-out test observations; they are not causal explanations.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from matplotlib.colors import LinearSegmentedColormap
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'water_potability.csv')
if not os.path.exists(DATA_PATH):
    raise SystemExit('water_potability.csv not found. Place the Kaggle CSV beside this script.')

PALETTE = {'navy':'#1E2A3A','rf':'#1a7a4c','xgb':'#2563a8','accent_orange':'#e67e22','muted':'#6b7785','grid':'#e3e6ea'}
plt.rcParams.update({'font.family':'DejaVu Sans','axes.titleweight':'bold','axes.titlesize':13,
    'axes.titlecolor':PALETTE['navy'],'axes.labelcolor':PALETTE['navy'],'xtick.color':PALETTE['muted'],
    'ytick.color':PALETTE['muted'],'axes.spines.top':False,'axes.spines.right':False,
    'figure.facecolor':'white','axes.edgecolor':PALETTE['grid']})

class FoldAwareXGBClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, n_estimators=300, max_depth=5, learning_rate=0.05,
                 random_state=RANDOM_STATE, eval_metric='logloss'):
        self.n_estimators=n_estimators; self.max_depth=max_depth; self.learning_rate=learning_rate
        self.random_state=random_state; self.eval_metric=eval_metric
    def fit(self, X, y):
        y_arr=np.asarray(y); n0=int(np.sum(y_arr==0)); n1=int(np.sum(y_arr==1))
        if n0==0 or n1==0: raise ValueError('Both classes must occur in each training fold.')
        self.scale_pos_weight_=n0/n1
        self.model_=xgb.XGBClassifier(n_estimators=self.n_estimators,max_depth=self.max_depth,
            learning_rate=self.learning_rate,scale_pos_weight=self.scale_pos_weight_,
            random_state=self.random_state,eval_metric=self.eval_metric)
        self.model_.fit(X,y_arr); self.classes_=self.model_.classes_; return self
    def predict(self,X): return self.model_.predict(X)
    def predict_proba(self,X): return self.model_.predict_proba(X)


df=pd.read_csv(DATA_PATH)
if 'Potability' not in df.columns: raise ValueError("CSV must contain 'Potability'.")
X=df.drop(columns='Potability'); y=df['Potability']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=RANDOM_STATE)
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=RANDOM_STATE)
models={
 'Random Forest':Pipeline([('imputer',SimpleImputer(strategy='median')),
   ('model',RandomForestClassifier(n_estimators=300,class_weight='balanced',random_state=RANDOM_STATE))]),
 'XGBoost':Pipeline([('imputer',SimpleImputer(strategy='median')),
   ('model',FoldAwareXGBClassifier(random_state=RANDOM_STATE))])}
cv_auc={}
for name,pipe in models.items():
    scores=cross_validate(pipe,X_train,y_train,cv=cv,scoring='roc_auc',n_jobs=None)['test_score']
    cv_auc[name]=float(np.mean(scores))
    pipe.fit(X_train,y_train)
    print(f'{name}: CV ROC-AUC={np.mean(scores):.6f} +/- {np.std(scores):.6f}; test ROC-AUC={roc_auc_score(y_test,pipe.predict_proba(X_test)[:,1]):.6f}')
selected_name=max(cv_auc,key=cv_auc.get)
selected_pipe=models[selected_name]
print(f'SHAP model selected by training CV ROC-AUC: {selected_name}')

# ROC comparison from held-out test set
fig,ax=plt.subplots(figsize=(7.5,6.2))
for name,pipe in models.items():
    prob=pipe.predict_proba(X_test)[:,1]; fpr,tpr,_=roc_curve(y_test,prob)
    ax.plot(fpr,tpr,linewidth=2.5,color=PALETTE['rf'] if name=='Random Forest' else PALETTE['xgb'],
            label=f'{name} (AUC={auc(fpr,tpr):.2f})')
ax.plot([0,1],[0,1],linestyle='--',color=PALETTE['muted'],linewidth=1.3,label='Random guess (AUC=0.50)')
ax.set(xlabel='False Positive Rate',ylabel='True Positive Rate',title='ROC Curve Comparison: Random Forest vs. XGBoost')
ax.legend(loc='lower right',frameon=False); ax.grid(alpha=.4); plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(DATA_PATH),'model_comparison_roc_v2.png'),dpi=200); plt.close()

# Explain the selected model using its own fitted imputer and estimator
X_test_imp=pd.DataFrame(selected_pipe.named_steps['imputer'].transform(X_test),columns=X.columns,index=X_test.index)
model_step=selected_pipe.named_steps['model']
tree_model=model_step.model_ if isinstance(model_step,FoldAwareXGBClassifier) else model_step
explainer=shap.TreeExplainer(tree_model)
sv=explainer.shap_values(X_test_imp)
if isinstance(sv,list): sv=sv[1]
elif isinstance(sv,np.ndarray) and sv.ndim==3: sv=sv[:,:,1]
custom_cmap=LinearSegmentedColormap.from_list('thesis_theme',[PALETTE['xgb'],'#f2f2f2',PALETTE['accent_orange']])
plt.figure(); shap.summary_plot(sv,X_test_imp,show=False,cmap=custom_cmap)
plt.gcf().axes[0].set_title(f'SHAP Summary — {selected_name}',fontsize=13,fontweight='bold',color=PALETTE['navy'],pad=10)
plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(DATA_PATH),'shap_summary_plot_v2.png'),dpi=200,bbox_inches='tight'); plt.close()
plt.figure(); shap.summary_plot(sv,X_test_imp,plot_type='bar',show=False,color=PALETTE['rf'])
plt.gcf().axes[0].set_title(f'Mean |SHAP Value| — {selected_name}',fontsize=13,fontweight='bold',color=PALETTE['navy'],pad=10)
plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(DATA_PATH),'shap_bar_plot_v2.png'),dpi=200,bbox_inches='tight'); plt.close()
mean_abs=np.abs(sv).mean(axis=0); top=X_test_imp.columns[np.argsort(mean_abs)[::-1][:4]]
fig,axes=plt.subplots(2,2,figsize=(13,9))
for ax,feat in zip(axes.flat,top):
    shap.dependence_plot(feat,sv,X_test_imp,ax=ax,show=False,cmap=custom_cmap)
    ax.set_title(str(feat),fontsize=12,fontweight='bold',color=PALETTE['navy'])
fig.suptitle(f'SHAP Dependence Plots — Top 4 Features ({selected_name})',fontsize=15,fontweight='bold',color=PALETTE['navy'],y=1.02)
plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(DATA_PATH),'shap_dependence_grid_v2.png'),dpi=200,bbox_inches='tight'); plt.close()
print('Saved ROC and selected-model SHAP plots. Interpret SHAP as model explanations, not chemical causation.')
