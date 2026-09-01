# Branch-Change Mechanism Probe

- samples: 2
- scorable positions: 2218
- baseline near-miss positions: 104 (4.69%)

## Delta Alignment

| metric | baseline | branch-change |
|---|---:|---:|
| mse | 5.9669 | 5.0641 |
| cos | 0.1074 | 0.2241 |
| top10_overlap | 0.0462 | 0.0635 |

## Near-Miss Repair

| metric | value |
|---|---:|
| teacher_top1_rank_baseline | 77.0000 |
| teacher_top1_rank_branch_change | 51.8600 |
| teacher_top1_logit_baseline | 7.4554 |
| teacher_top1_logit_branch_change | 9.5772 |
| branch_change_top1_correct_on_baseline_near_miss | 0.4500 |

## Ambiguity Buckets

| bucket | positions | baseline top1 agree | branch-change top1 agree | delta |
|---|---:|---:|---:|---:|
| low | 2002 | 84.52% | 90.06% | +5.54 pts |
| medium | 105 | 74.29% | 72.38% | -1.90 pts |
| high | 111 | 44.14% | 49.55% | +5.41 pts |
