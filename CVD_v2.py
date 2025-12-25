import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import precision_recall_curve, brier_score_loss, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Random Forest model trained on Kaggle dataset
df_k = pd.read_csv("https://raw.githubusercontent.com/akhilchibber/Cardiovascular-Disease-Detection/master/cardio_train.csv", sep=';')
df_k['age'] = (df_k['age'] / 365).astype(int)
df_k = df_k[(df_k['ap_hi'] >= 80) & (df_k['ap_hi'] <= 250)]
df_k['bmi'] = df_k['weight'] / ((df_k['height'] / 100) ** 2)
Xk = df_k[['age', 'ap_hi', 'ap_lo', 'cholesterol', 'bmi']]
yk = df_k['cardio']

Xk_train, Xk_test, yk_train, yk_test = train_test_split(Xk, yk, test_size=0.2, random_state=1, stratify=yk)
sk = StandardScaler()
Xk_train_s = sk.fit_transform(Xk_train)
Xk_test_s = sk.transform(Xk_test)

mk = CalibratedClassifierCV(RandomForestClassifier(n_estimators=100, max_depth=8, random_state=1), cv=5).fit(Xk_train_s, yk_train)
yk_probs = mk.predict_proba(Xk_test_s)[:, 1]
pk, rk, tk_list = precision_recall_curve(yk_test, yk_probs)
tk = tk_list[np.argmax((5 * pk * rk) / (4 * pk + rk))] # F2-score used as a threshold in order to minimize False Negatives due to medical context.

# Random Forest model trained on dataset provided by UC Irvine
cols_c = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
df_c = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", names=cols_c, na_values="?").dropna()
df_c['target'] = (df_c['target'] > 0).astype(int)
Xc = df_c.drop("target", axis=1)
yc = df_c["target"]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=1, stratify=yc)
sc = StandardScaler()
Xc_train_s = sc.fit_transform(Xc_train)
Xc_test_s = sc.transform(Xc_test)

mc = CalibratedClassifierCV(RandomForestClassifier(n_estimators=100, max_depth=8, random_state=1), cv=5).fit(Xc_train_s, yc_train)
yc_probs = mc.predict_proba(Xc_test_s)[:, 1]
pc, rc, tc_list = precision_recall_curve(yc_test, yc_probs)
tc = tc_list[np.argmax((5 * pc * rc) / (4 * pc + rc))] # Same reasoning as on the line 28

# Test patient
patient_stats = {'age': 67, 'sex': 1, 'ap_hi': 158, 'ap_lo': 92, 'chol_cat': 3, 'bmi': 29.8,
                 'cp': 4, 'trestbps': 158, 'chol': 285, 'fbs': 0, 'restecg': 2, 'thalach': 120,
                 'exang': 1, 'oldpeak': 1.8, 'slope': 2, 'ca': 2, 'thal': 7}

p_df_k = pd.DataFrame([[patient_stats['age'], patient_stats['ap_hi'], patient_stats['ap_lo'], patient_stats['chol_cat'], patient_stats['bmi']]], columns=Xk.columns)
p_df_c = pd.DataFrame([[patient_stats['age'], patient_stats['sex'], patient_stats['cp'], patient_stats['trestbps'], patient_stats['chol'], patient_stats['fbs'], patient_stats['restecg'], patient_stats['thalach'], patient_stats['exang'], patient_stats['oldpeak'], patient_stats['slope'], patient_stats['ca'], patient_stats['thal']]], columns=Xc.columns)

# Outlier detection using 2.5 standard deviations 
outliers = []
for col in Xc.columns:
    mean, std = Xc[col].mean(), Xc[col].std()
    if abs(patient_stats[col] - mean) > 2.5 * std:
        outliers.append(col.upper())

pk_s, pc_s = sk.transform(p_df_k), sc.transform(p_df_c)

mk_preds = np.array([clf.predict_proba(pk_s)[0, 1] for clf in mk.calibrated_classifiers_])
mc_preds = np.array([clf.predict_proba(pc_s)[0, 1] for clf in mc.calibrated_classifiers_])

risk_k, ci_k = np.mean(mk_preds), np.std(mk_preds) * 1.96
risk_c, ci_c = np.mean(mc_preds), np.std(mc_preds) * 1.96

# SHAP for the feature importance analysis
internal_model = mc.calibrated_classifiers_[0].estimator
explainer = shap.TreeExplainer(internal_model)
shap_v = explainer.shap_values(pc_s)
importance_vals = np.array(shap_v[1]).flatten() if isinstance(shap_v, list) else shap_v[0, :, 1].flatten()


# Patient results and model summary
fig = plt.figure(figsize=(18, 12))
plt.suptitle(f"Cardiovascular CDSS. Patient is {patient_stats['age']} years old, {'Male' if patient_stats['sex']==1 else 'Female'}", 
             fontsize=22, fontweight='bold')


ax1 = plt.subplot(2, 3, 1)
labs = ['Kaggle', 'Cleveland']
ax1.bar(labs, [risk_k, risk_c], yerr=[ci_k, ci_c], capsize=10, color=['#440154', '#21918c'], alpha=0.7)
for i, (lab, thresh) in enumerate(zip(labs, [tk, tc])):
    ax1.hlines(thresh, xmin=i-0.4, xmax=i+0.4, colors='black', linestyles='--', lw=2, label=f'{lab} threshold')
ax1.set_ylim(0, 1)
ax1.set_title("1. Risk Assessment & Limits")
ax1.legend(loc='upper left', fontsize=9)


ax2 = plt.subplot(2, 3, 2)
shap_df = pd.DataFrame({'feature': list(Xc.columns), 'importance': importance_vals}).sort_values(by='importance', ascending=False)
sns.barplot(data=shap_df.head(10), x='importance', y='feature', hue='feature', palette='RdBu_r', ax=ax2, legend=False)
ax2.set_title("2. Why this Patient? (Important Features)")


ax3 = plt.subplot(2, 3, 3)
ax3.axis('off')
status = "Dangerous" if risk_c > tc and risk_k > tk else "NORMAL"
alert_text = f"OUTLIERS DETECTED: {', '.join(outliers)}" if outliers else "NO OUTLIERS"

report = (f"CDSS SUMMARY\n\n"
          f"DIAGNOSTIC STATUS: {status}\n\n"
          f"Risk based on Kaggle dataset: {risk_k:.1%} (±{ci_k:.1%})\n"
          f"Warning threshold: {tk:.1%}\n\n"
          f"Risk based on Cleveland model: {risk_c:.1%} (±{ci_c:.1%})\n"
          f"Warning threshold: {tc:.1%}\n\n"
          f"Significant feature: {shap_df.iloc[0]['feature'].upper()}\n\n"
          f"DATA INTEGRITY CHECK: {alert_text}\n"
          f"Confidence: {'LOW' if ci_c > 0.15 or outliers or ci_k > 0.15 else 'HIGH'}")
ax3.text(0, 0.5, report, fontsize=13, family='monospace', va='center', 
         bbox=dict(facecolor='whitesmoke', alpha=0.9, edgecolor='#21918c' if not outliers else 'red'))


ax4 = plt.subplot(2, 2, 3)
fk_roc, tk_roc, _ = roc_curve(yk_test, yk_probs)
fc_roc, tc_roc, _ = roc_curve(yc_test, yc_probs)
ax4.plot(fk_roc, tk_roc, label=f'Kaggle (AUC: {roc_auc_score(yk_test, yk_probs):.2f})', color='#440154')
ax4.plot(fc_roc, tc_roc, label=f'Cleveland (AUC: {roc_auc_score(yc_test, yc_probs):.2f})', color='#21918c')
ax4.plot([0,1],[0,1], 'k--')
ax4.set_title("4. ROC curve")
ax4.legend()


ax5 = plt.subplot(2, 2, 4)
pk_cal, fk_cal = calibration_curve(yk_test, yk_probs, n_bins=8)
pc_cal, fc_cal = calibration_curve(yc_test, yc_probs, n_bins=8)
ax5.plot(fk_cal, pk_cal, "s-", label='Kaggle', color='#440154')
ax5.plot(fc_cal, pc_cal, "o-", label='Cleveland', color='#21918c')
ax5.plot([0,1],[0,1], 'k--')
ax5.set_title("5. Calibration (Reliability Validation)")
ax5.legend()


plt.tight_layout()
plt.savefig('summary.pdf')
plt.show()
