# Gate k-sweep addendum: precondition at k in {7, 8}, corrected headroom basis

Addendum of: `results/ksweep-20260801T184604Z.json`
Reason: original grid bracketed the coverage/purity crossover at k=5 (coverage not yet complete) and k=10 (coverage complete, purity collapsed) without measuring k in {7, 8} = the largest cluster's sibling count; added to measure rather than infer whether the precondition is satisfiable at all

Corpus SHA unchanged: True

## 1. F15/I4 precondition, measured at k in {7, 8}

| arm | k | sibling coverage | competitor purity | precondition satisfied |
|---|---|---|---|---|
| bm25 | 7 | 0.8083 [0.5741, 0.9356] | 0.5667 [0.4488, 0.6989] | False |
| bm25 | 8 | 0.9972 [0.9870, 1.0000] | 0.5139 [0.3989, 0.6580] | False |
| plain_vector | 7 | 0.8000 [0.5678, 0.9290] | 0.5653 [0.4477, 0.6978] | False |
| plain_vector | 8 | 0.9944 [0.9822, 1.0000] | 0.5135 [0.3976, 0.6585] | False |

## 2. Corrected headroom -- single basis, ceiling varied only

Basis: control and both numerators fixed at F24's 180-query subsample values (A_ctrl=0.6000, llm_pick=0.9444, verify_veto=0.8833); only the ceiling R varies across rows

| k | ceiling source | R (ceiling) | A_ctrl (0.6000, subsample) | gap | H(llm_pick) | H(verify_veto) |
|---|---|---|---|---|---|---|
| 5 | paper-quoted (0.958), same subsample-adjacent basis | 0.9580 | 0.6000 | 0.3580 | 0.9620 | 0.7913 |
| 10 | full-corpus plain_vector recall@k, measured this sweep -- NOT measured on the 180-query subsample; combining it with the subsample control/numerators below is a stated assumption, not a second measurement | 1.0000 | 0.6000 | 0.4000 | 0.8610 | 0.7082 |
| 20 | full-corpus plain_vector recall@k, measured this sweep -- NOT measured on the 180-query subsample; combining it with the subsample control/numerators below is a stated assumption, not a second measurement | 1.0000 | 0.6000 | 0.4000 | 0.8610 | 0.7082 |
| 50 | full-corpus plain_vector recall@k, measured this sweep -- NOT measured on the 180-query subsample; combining it with the subsample control/numerators below is a stated assumption, not a second measurement | 1.0000 | 0.6000 | 0.4000 | 0.8610 | 0.7082 |

Wall clock: 521.9s
