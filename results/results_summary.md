On a random stratified segment-level test split, the best model by PR-AUC was Random Forest (sensitivity 0.95, specificity 1.00, PR-AUC 1.00, ROC-AUC 1.00). These numbers describe performance on segments similar to the training data, not performance on new patients.

- Rows x columns: 11233 x 36865; class counts {'0': 9799, '1': 1434}
- Row layout verdict: **blocks** (Equal-length contiguous segments detected from regular jumps in adjacent-column difference.)
- Sampling rate: unknown - analysis in normalised frequency
- Label meaning: inferred: observed class counts match the counts reported by the dataset creator
- No patient/recording identifiers: patient-independent evaluation is not possible.
- Relative band power, seizure / non-seizure median ratio: nb1 x0.99, nb2 x1.29, nb3 x0.45, nb4 x1.62, nb5 x1.38
- Most discriminative single features (|Cohen's d|): blk15_std (+1.96), blk15_rms (+1.96), blk10_std (+1.84), blk10_rms (+1.84), blk06_rms (+1.82)
- Random Forest top features: blk18_var, blk06_line_length, blk18_rms, blk18_std, blk18_bp_nb1

| model | sensitivity | specificity | precision | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.976 | 0.973 | 0.841 | 0.903 | 0.975 | 0.996 |
| SVM (RBF) | 0.972 | 0.987 | 0.918 | 0.944 | 0.993 | 0.999 |
| Random Forest | 0.951 | 0.998 | 0.989 | 0.970 | 0.996 | 0.999 |

Best model by PR-AUC: **Random Forest** (test set, random stratified segment-level split).
- Ordered-split sensitivity check (PR-AUC random -> ordered): Logistic Regression 0.975 -> 0.964, SVM (RBF) 0.993 -> 0.993, Random Forest 0.996 -> 1.000