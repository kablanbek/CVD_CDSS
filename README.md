# CVD_CDSS
Clinical-grade ML dual model for cardiovascular risk assessments with high emphasis on medical safety and interpretability.

# Main features:
1. **Two-Model Architecture**: Combines a **Lifestyle Model** (5-feature screening using a large dataset) with a **Diagnostic Model** (13-feature clinical analysis based on a small dataset) to track patient risk across different stages of care.
2. **Interpretability**: Using **SHAP** (SHapley Additive exPlanations) to identify the important features that lie behind an individual patient's risk score.
3. **Safety prioritization**:
   3.a. Calibrated Probabilities: Uses _CalibratedClassifierCV_ to ensure a predicted 70% risk corresponds to a 70% real-world incidence.
   3.b. $F_2$-scored Thresholds: Thresholds are tuned to prioritize Recall (Sensitivity), reducing the risk of life-threatening _False Negatives_. No one would like to be a False Negative in such case, right?
   3.c. Statstical Alerts: Automatically detects patient data outliers that fall outside $2.5$ standard deviations of the training population.

# Generated Dashboard description: 
Risk assessment with model-specific safety limits.
1. **SHAP Analysis**: Local interpretability showing why a specific patient is high risk.
2. **Clinical Summary**: Automated report with status alerts and data integrity checks.
3. **Model Validation**: Integrated ROC and Calibration curves to prove model reliability.
