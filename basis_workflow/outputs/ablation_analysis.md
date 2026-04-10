# Pipeline Ablation Analysis Report

**Reference**: gse20194.csv | **Target**: gse58644.csv ($N=1$ mode)

| Experiment | ER Centroid Shift | ER AUC | HER2 AUC |
| :--- | :---: | :---: | :---: |
| 1. Global Ranking (Baseline) | 0.517 | 0.930 | 0.902 |
| 2. + Anchor-Only | 1.834 | 0.929 | 0.894 |
| 3. + Lamé Warp | 1.827 | 0.929 | 0.892 |
| 4. + Sentinel Tare | 0.431 | 0.941 | 0.865 |
| 5. + BGN (Full Pipeline) | 0.468 | 0.933 | 0.879 |