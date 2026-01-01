## PBII — ROC–AUC Evaluation (Zero-Shot)

PBII (Prime Base Instability Index) is a **structural instability measure**, not a classifier.

By construction:
- **lower PBII → primes**
- **higher PBII → composites**

Therefore, PBII scores are **anti-correlated** with the conventional label
`prime = 1`.

### Raw ROC–AUC (incorrect polarity)

ROC–AUC = 0.184

This value reflects **reversed ordering**, not lack of signal.

### Polarity-corrected ROC–AUC
By inverting the score (`-PBII`):

ROC–AUC = 0.816

### Interpretation
- Zero-shot
- No training
- No feature learning
- Pure numeric structure

PBII **separates primes from composites by rank ordering**, which is the
appropriate evaluation criterion for a structural metric.

