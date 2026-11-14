# Decisions over H1-H8: generated summary

Holm over 8 hypotheses at family-wise 0.05; complete: True (pending or not-evaluable hypotheses enter Holm with p = 1; the adjusted p-values are upper bounds of the final ones). Code defect: False.

Pending (input missing): none.

| hypothesis | raw p (IU) | Holm-adjusted p | decision | contradicted | null reading (if not supported) |
| --- | --- | --- | --- | --- | --- |
| H1 | 0.0232 | 0.186 | not supported | - | three-way reading of the removal threshold per network (see readings) |
| H2w | 0.989 | 1 | not supported | - | three-way reading of the removal threshold per network (see readings) |
| H2b | 0.0454 | 0.318 | not supported | - | three-way reading of the removal threshold per network (see readings) |
| H3b | 0.83 | 1 | not supported | - | BTP bounds Tier >= 1 admission per attack edge (H3a); under the count gate, the attacker's best gain per attack edge is not reduced. |
| H6a | 0.896 | 1 | not supported | - | reported as not supported in the section that makes the claim |
| H6b | 0.771 | 1 | not supported | - | reported as not supported in the section that makes the claim |
| H7 | 1 | 1 | contradicted | bitcoinalpha/max_auc, bitcoinotc/max_auc, wiki-rfa/max_auc | criterion G3 is unmet |
| H8 | 0.0577 | 0.346 | not supported | - | three-way reading of the removal threshold per network (see readings) |

## Components

| hypothesis | network | component | rule | estimate | 95% interval | p | reading | Holm-adjusted p | decision | missing runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| H1 | bitcoinalpha | a_difference | interval > 0 | 0.1734 | [0.1340, 0.2207] | 0.0001 | met | 0.186 | not supported | 0 |
| H1 | bitcoinalpha | b_cut | lower bound >= 0.8 | 0.9834 | [0.8036, 1.1632] | 0.0232 | removal met | 0.186 | not supported | 0 |
| H1 | bitcoinotc | a_difference | interval > 0 | 0.1741 | [0.1437, 0.2076] | 0.0001 | met | 0.186 | not supported | 0 |
| H1 | bitcoinotc | b_cut | lower bound >= 0.8 | 1.1540 | [0.9675, 1.3406] | 0.001 | removal met | 0.186 | not supported | 0 |
| H1 | wiki-rfa | a_difference | interval > 0 | 0.2255 | [0.1870, 0.2650] | 0.0001 | met | 0.186 | not supported | 0 |
| H1 | wiki-rfa | b_cut | lower bound >= 0.8 | 1.0087 | [0.8997, 1.1178] | 0.000952 | removal met | 0.186 | not supported | 0 |
| H2w | bitcoinalpha | a_difference | interval > 0 | 0.0248 | [-0.0139, 0.0635] | 0.0907 | not met | 1 | not supported | 0 |
| H2w | bitcoinalpha | b_suppression | lower bound >= 0.8 | 0.3854 | [-0.2135, 0.8742] | 0.989 | removal not established | 1 | not supported | 0 |
| H2w | bitcoinotc | a_difference | interval > 0 | 0.0466 | [0.0224, 0.0711] | 0.000911 | met | 1 | not supported | 0 |
| H2w | bitcoinotc | b_suppression | lower bound >= 0.8 | 0.6672 | [0.4409, 0.8935] | 0.9 | removal not established | 1 | not supported | 0 |
| H2w | wiki-rfa | a_difference | interval > 0 | 0.3231 | [0.2636, 0.3866] | 0.0001 | met | 1 | not supported | 0 |
| H2w | wiki-rfa | b_suppression | lower bound >= 0.8 | 0.9624 | [0.8539, 1.0709] | 0.00403 | removal met | 1 | not supported | 0 |
| H2b | bitcoinalpha | a_difference | interval > 0 | 0.0750 | [0.0666, 0.0838] | 0.0001 | met | 0.318 | not supported | 0 |
| H2b | bitcoinalpha | b_suppression | lower bound >= 0.8 | 0.9434 | [0.9124, 0.9744] | 0.0001 | removal met | 0.318 | not supported | 0 |
| H2b | bitcoinotc | a_difference | interval > 0 | 0.0831 | [0.0740, 0.0915] | 0.0001 | met | 0.318 | not supported | 0 |
| H2b | bitcoinotc | b_suppression | lower bound >= 0.8 | 0.8826 | [0.8378, 0.9273] | 0.0012 | removal met | 0.318 | not supported | 0 |
| H2b | wiki-rfa | a_difference | interval > 0 | 0.1590 | [0.1336, 0.1861] | 0.0001 | met | 0.318 | not supported | 0 |
| H2b | wiki-rfa | b_suppression | lower bound >= 0.8 | 0.8625 | [0.7879, 0.9371] | 0.0454 | removal not established | 0.318 | not supported | 0 |
| H3b | bitcoinalpha | difference | interval > 0 | -0.0095 | [-0.0429, 0.0410] | 0.83 | not met | 1 | not supported | 0 |
| H3b | bitcoinotc | difference | interval > 0 | 0.0197 | [-0.0367, 0.0621] | 0.167 | not met | 1 | not supported | 0 |
| H3b | wiki-rfa | difference | interval > 0 | 0.0535 | [-0.0162, 0.1314] | 0.0677 | not met | 1 | not supported | 0 |
| H6a | bitcoinalpha | whitewash | interval > 0 | 0.0090 | [-0.0377, 0.0558] | 0.336 | not met | 1 | not supported | 0 |
| H6a | bitcoinalpha | badmouth | interval > 0 | 0.0056 | [-0.0045, 0.0158] | 0.125 | not met | 1 | not supported | 0 |
| H6a | bitcoinotc | whitewash | interval > 0 | 0.0194 | [-0.0114, 0.0501] | 0.0967 | not met | 1 | not supported | 0 |
| H6a | bitcoinotc | badmouth | interval > 0 | 0.0008 | [-0.0085, 0.0106] | 0.433 | not met | 1 | not supported | 0 |
| H6a | wiki-rfa | whitewash | interval > 0 | 0.0180 | [-0.0311, 0.0671] | 0.219 | not met | 1 | not supported | 0 |
| H6a | wiki-rfa | badmouth | interval > 0 | -0.0074 | [-0.0203, 0.0051] | 0.896 | not met | 1 | not supported | 0 |
| H6b | bitcoinalpha | whitewash | interval > 0 | -0.0239 | [-0.1049, 0.0459] | 0.771 | not met | 1 | not supported | 0 |
| H6b | bitcoinalpha | badmouth | interval > 0 | 0.0005 | [-0.0101, 0.0116] | 0.47 | not met | 1 | not supported | 0 |
| H6b | bitcoinotc | whitewash | interval > 0 | 0.0133 | [-0.0962, 0.1228] | 0.408 | not met | 1 | not supported | 0 |
| H6b | bitcoinotc | badmouth | interval > 0 | 0.0142 | [-0.0127, 0.0412] | 0.131 | not met | 1 | not supported | 0 |
| H6b | wiki-rfa | whitewash | interval > 0 | 0.0945 | [0.0088, 0.1802] | 0.0171 | met | 1 | not supported | 0 |
| H6b | wiki-rfa | badmouth | interval > 0 | 0.0186 | [-0.0115, 0.0487] | 0.0978 | not met | 1 | not supported | 0 |
| H7 | bitcoinalpha | max_auc | interval < 0.6 | 0.7989 | [0.7748, 0.8230] | 1 | contradicted | 1 | contradicted | 0 |
| H7 | bitcoinotc | max_auc | interval < 0.6 | 0.7596 | [0.7403, 0.7788] | 1 | contradicted | 1 | contradicted | 0 |
| H7 | wiki-rfa | max_auc | interval < 0.6 | 0.7155 | [0.6915, 0.7395] | 1 | contradicted | 1 | contradicted | 0 |
| H8 | bitcoinalpha | a | interval > 0 | 0.2681 | [0.1898, 0.3464] | 0.000342 | met | 0.346 | not supported | 0 |
| H8 | bitcoinalpha | b | lower bound >= 0.5 | 0.7262 | [0.4660, 0.9863] | 0.0366 | removal not established | 0.346 | not supported | 0 |
| H8 | bitcoinotc | a | interval > 0 | 0.3401 | [0.1152, 0.5649] | 0.00685 | met | 0.346 | not supported | 0 |
| H8 | bitcoinotc | b | lower bound >= 0.5 | 1.0251 | [0.2981, 1.7521] | 0.0577 | removal not established | 0.346 | not supported | 0 |
| H8 | wiki-rfa | a | interval > 0 | 0.4349 | [0.3404, 0.5295] | 0.000108 | met | 0.346 | not supported | 0 |
| H8 | wiki-rfa | b | lower bound >= 0.5 | 0.7613 | [0.6110, 0.9117] | 0.00424 | removal met | 0.346 | not supported | 0 |

## Exact properties and the deterministic rule

- H3a: holds (150 capped-BTP runs, 2700 groups, 0 violations).
- H5: supported; untested: 21/bitcoinalpha/exponential, 21/bitcoinotc/exponential, 21/wiki-rfa/exponential.
- H4: supported (census mean of attack - clean <= 0.01; null reading: M-B reduces, not removes, witness laundering, with the measured residual).
  - bitcoinalpha: holds (k=5 ring=False: mean -0.0000, p95 0.0000, max 0.0021; k=5 ring=True: mean -0.0000, p95 0.0001, max 0.0022; k=50 ring=False: mean 0.0000, p95 0.0000, max 0.0000; k=50 ring=True: mean 0.0001, p95 0.0005, max 0.0095)
  - bitcoinotc: holds (k=5 ring=False: mean -0.0001, p95 0.0000, max 0.0008; k=5 ring=True: mean -0.0001, p95 0.0001, max 0.0010; k=50 ring=False: mean -0.0000, p95 0.0000, max 0.0000; k=50 ring=True: mean 0.0001, p95 0.0005, max 0.0095)
  - wiki-rfa: holds (k=5 ring=False: mean 0.0003, p95 0.0003, max 0.0258; k=5 ring=True: mean 0.0006, p95 0.0010, max 0.0263; k=50 ring=False: mean 0.0001, p95 0.0000, max 0.0097; k=50 ring=True: mean 0.0034, p95 0.0095, max 0.0102)

## Inputs

- results/analysis/cppr_confirm.json: present, status ok
- results/analysis/e2.json: present
- results/analysis/e6.json: present, status ok
- results/analysis/e9a_e8.json: present, status ok
- results/analysis/e1.json: present

## Missing runs

- H1: 0
- H2w: 0
- H2b: 0
- H3b: 0
- H6a: 0
- H6b: 0
- H7: 0
- H8: 0
- H3a: 0
- H4: 0
- H5: 0

## Notes

- none
