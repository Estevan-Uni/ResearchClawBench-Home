### T1: Together_1 random 80/20 split

| Feature set | Behaviour | Threshold | Labels | n seeds | AP | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|
| engineered | Attack | 0.5 | original | 5 | 0.709 ± 0.014 | 0.918 ± 0.008 | 0.846 ± 0.009 | 0.818 ± 0.013 | 0.876 ± 0.011 |
| engineered | Attack | 0.5 | gap_closed_eval | 5 | 0.999 ± 0.001 | 0.999 ± 0.002 | 0.945 ± 0.005 | 1.000 ± 0.000 | 0.896 ± 0.010 |
| engineered | Attack | oob_maxF1 | original | 5 | 0.709 ± 0.014 | 0.918 ± 0.008 | 0.905 ± 0.007 | 0.833 ± 0.010 | 0.990 ± 0.008 |
| engineered | Attack | oob_maxF1 | gap_closed_eval | 5 | 0.999 ± 0.001 | 0.999 ± 0.002 | 0.994 ± 0.005 | 0.997 ± 0.004 | 0.991 ± 0.007 |
| engineered | Sniffing | 0.5 | original | 5 | 0.761 ± 0.049 | 0.979 ± 0.005 | 0.787 ± 0.026 | 0.822 ± 0.034 | 0.757 ± 0.037 |
| engineered | Sniffing | 0.5 | gap_closed_eval | 5 | 0.996 ± 0.005 | 0.999 ± 0.001 | 0.873 ± 0.030 | 0.995 ± 0.010 | 0.779 ± 0.047 |
| engineered | Sniffing | oob_maxF1 | original | 5 | 0.761 ± 0.049 | 0.979 ± 0.005 | 0.899 ± 0.024 | 0.826 ± 0.031 | 0.987 ± 0.026 |
| engineered | Sniffing | oob_maxF1 | gap_closed_eval | 5 | 0.996 ± 0.005 | 0.999 ± 0.001 | 0.982 ± 0.013 | 0.975 ± 0.018 | 0.989 ± 0.022 |
| pose48 | Attack | 0.5 | original | 3 | 0.750 ± 0.017 | 0.927 ± 0.006 | 0.901 ± 0.006 | 0.838 ± 0.010 | 0.975 ± 0.007 |
| pose48 | Attack | 0.5 | gap_closed_eval | 3 | 0.999 ± 0.001 | 0.999 ± 0.001 | 0.989 ± 0.003 | 1.000 ± 0.000 | 0.979 ± 0.006 |
| pose48 | Attack | oob_maxF1 | original | 3 | 0.750 ± 0.017 | 0.927 ± 0.006 | 0.911 ± 0.006 | 0.839 ± 0.008 | 0.997 ± 0.004 |
| pose48 | Attack | oob_maxF1 | gap_closed_eval | 3 | 0.999 ± 0.001 | 0.999 ± 0.001 | 0.998 ± 0.002 | 0.998 ± 0.003 | 0.998 ± 0.003 |
| pose48 | Sniffing | 0.5 | original | 3 | 0.750 ± 0.033 | 0.979 ± 0.004 | 0.862 ± 0.024 | 0.824 ± 0.036 | 0.906 ± 0.010 |
| pose48 | Sniffing | 0.5 | gap_closed_eval | 3 | 0.998 ± 0.001 | 1.000 ± 0.000 | 0.953 ± 0.006 | 0.987 ± 0.009 | 0.920 ± 0.006 |
| pose48 | Sniffing | oob_maxF1 | original | 3 | 0.750 ± 0.033 | 0.979 ± 0.004 | 0.897 ± 0.023 | 0.814 ± 0.039 | 1.000 ± 0.000 |
| pose48 | Sniffing | oob_maxF1 | gap_closed_eval | 3 | 0.998 ± 0.001 | 1.000 ± 0.000 | 0.979 ± 0.019 | 0.960 ± 0.035 | 1.000 ± 0.000 |
| provided50 | Attack | 0.5 | original | 3 | 0.749 ± 0.010 | 0.927 ± 0.004 | 0.902 ± 0.002 | 0.838 ± 0.009 | 0.977 ± 0.008 |
| provided50 | Attack | 0.5 | gap_closed_eval | 3 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.990 ± 0.004 | 1.000 ± 0.000 | 0.981 ± 0.007 |
| provided50 | Attack | oob_maxF1 | original | 3 | 0.749 ± 0.010 | 0.927 ± 0.004 | 0.911 ± 0.006 | 0.839 ± 0.008 | 0.997 ± 0.004 |
| provided50 | Attack | oob_maxF1 | gap_closed_eval | 3 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.998 ± 0.002 | 0.998 ± 0.003 | 0.998 ± 0.003 |
| provided50 | Sniffing | 0.5 | original | 3 | 0.757 ± 0.037 | 0.979 ± 0.004 | 0.878 ± 0.028 | 0.828 ± 0.036 | 0.935 ± 0.018 |
| provided50 | Sniffing | 0.5 | gap_closed_eval | 3 | 0.998 ± 0.002 | 1.000 ± 0.000 | 0.966 ± 0.011 | 0.987 ± 0.009 | 0.945 ± 0.013 |
| provided50 | Sniffing | oob_maxF1 | original | 3 | 0.757 ± 0.037 | 0.979 ± 0.004 | 0.897 ± 0.023 | 0.814 ± 0.039 | 1.000 ± 0.000 |
| provided50 | Sniffing | oob_maxF1 | gap_closed_eval | 3 | 0.998 ± 0.002 | 1.000 ± 0.000 | 0.979 ± 0.019 | 0.960 ± 0.035 | 1.000 ± 0.000 |
| index_only | Attack | 0.5 | original | 3 | 0.665 ± 0.012 | 0.906 ± 0.006 | 0.866 ± 0.011 | 0.822 ± 0.007 | 0.915 ± 0.018 |
| index_only | Attack | 0.5 | gap_closed_eval | 3 | 0.999 ± 0.000 | 1.000 ± 0.000 | 0.959 ± 0.006 | 0.992 ± 0.006 | 0.929 ± 0.015 |
| index_only | Attack | oob_maxF1 | original | 3 | 0.665 ± 0.012 | 0.906 ± 0.006 | 0.907 ± 0.006 | 0.834 ± 0.006 | 0.994 ± 0.008 |
| index_only | Attack | oob_maxF1 | gap_closed_eval | 3 | 0.999 ± 0.000 | 1.000 ± 0.000 | 0.994 ± 0.002 | 0.993 ± 0.006 | 0.995 ± 0.007 |
| index_only | Sniffing | 0.5 | original | 3 | 0.681 ± 0.029 | 0.973 ± 0.005 | 0.866 ± 0.013 | 0.824 ± 0.025 | 0.913 ± 0.018 |
| index_only | Sniffing | 0.5 | gap_closed_eval | 3 | 0.995 ± 0.006 | 0.999 ± 0.001 | 0.955 ± 0.017 | 0.987 ± 0.019 | 0.926 ± 0.017 |
| index_only | Sniffing | oob_maxF1 | original | 3 | 0.681 ± 0.029 | 0.973 ± 0.005 | 0.897 ± 0.023 | 0.814 ± 0.039 | 1.000 ± 0.000 |
| index_only | Sniffing | oob_maxF1 | gap_closed_eval | 3 | 0.995 ± 0.006 | 0.999 ± 0.001 | 0.979 ± 0.019 | 0.960 ± 0.035 | 1.000 ± 0.000 |
| engineered | Attack | oob_maxF1 | gap_closed_train_and_eval | 3 | 0.999 ± 0.000 | 1.000 ± 0.000 | 0.987 ± 0.004 | 0.986 ± 0.010 | 0.988 ± 0.003 |
| engineered | Sniffing | oob_maxF1 | gap_closed_train_and_eval | 3 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.994 ± 0.004 | 1.000 ± 0.000 | 0.988 ± 0.009 |

### T2: Reference recording CV

| Feature set | Behaviour | Scheme | AP | ROC-AUC | F1@0.5 | prevalence | bouts |
|---|---|---|---|---|---|---|---|
| reference_engineered_recomputed | Attack | blocked_5fold | 0.179 | 0.197 | 0.167 | 0.163 | 3 |
| reference_engineered_recomputed | Attack | stratified_random_5fold | 0.989 ± 0.001 | 0.998 ± 0.000 | 0.937 ± 0.016 | 0.163 | 3 |
| reference_engineered_recomputed | Sniffing | blocked_5fold | 0.037 | 0.194 | 0.000 | 0.037 | 1 |
| reference_engineered_recomputed | Sniffing | stratified_random_5fold | 0.975 ± 0.000 | 0.999 ± 0.000 | 0.874 ± 0.024 | 0.037 | 1 |
| reference_simba517 | Attack | blocked_5fold | 0.152 | 0.235 | 0.113 | 0.163 | 3 |
| reference_simba517 | Attack | stratified_random_5fold | 0.990 ± 0.000 | 0.998 ± 0.000 | 0.917 ± 0.008 | 0.163 | 3 |
| reference_simba517 | Sniffing | blocked_5fold | 0.037 | 0.235 | 0.000 | 0.037 | 1 |
| reference_simba517 | Sniffing | stratified_random_5fold | 0.979 ± 0.004 | 0.999 ± 0.000 | 0.874 ± 0.024 | 0.037 | 1 |

### T3: Transfer

| Direction | Behaviour | target prevalence | AP | ROC-AUC | F1@0.5 | predicted positives | max p |
|---|---|---|---|---|---|---|---|
| Together1_to_Reference | Attack | 0.163 | 0.128 | 0.363 | 0.000 | 0 | 0.406 |
| Reference_to_Together1 | Attack | 0.338 | 0.250 | 0.309 | 0.007 | 5 | 0.530 |
| Together1_to_Reference | Sniffing | 0.037 | 0.099 | 0.812 | 0.000 | 0 | 0.362 |
| Reference_to_Together1 | Sniffing | 0.133 | 0.178 | 0.588 | 0.000 | 0 | 0.182 |

### T4: Blocked CV pooled

| Feature set | Behaviour | AP | ROC-AUC | F1@0.5 | prevalence |
|---|---|---|---|---|---|
| engineered | Attack | 0.220 | 0.209 | 0.026 | 0.338 |
| engineered | Sniffing | 0.100 | 0.378 | 0.000 | 0.133 |
| provided50 | Attack | 0.256 | 0.307 | 0.244 | 0.338 |
| provided50 | Sniffing | 0.138 | 0.310 | 0.139 | 0.133 |
| index_only | Attack | 0.408 | 0.503 | 0.321 | 0.338 |
| index_only | Sniffing | 0.175 | 0.488 | 0.155 | 0.133 |