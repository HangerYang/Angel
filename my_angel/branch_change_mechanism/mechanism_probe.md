# Branch-Change Mechanism Probe

- samples: 32
- scorable positions: 28613
- baseline near-miss positions: 1632 (5.70%)

## Delta Alignment

| metric | baseline | branch-change |
|---|---:|---:|
| mse | 5.6479 | 4.4082 |
| cos | 0.1446 | 0.3061 |
| top10_overlap | 0.0589 | 0.0787 |

## Near-Miss Repair

| metric | value |
|---|---:|
| teacher_top1_rank_baseline | 162.9317 |
| teacher_top1_rank_branch_change | 64.0961 |
| teacher_top1_logit_baseline | 7.9126 |
| teacher_top1_logit_branch_change | 9.2402 |
| branch_change_top1_correct_on_baseline_near_miss | 0.3647 |

## Ambiguity Buckets

| bucket | positions | baseline top1 agree | branch-change top1 agree | delta |
|---|---:|---:|---:|---:|
| low | 24042 | 88.27% | 95.50% | +7.22 pts |
| medium | 1827 | 65.63% | 77.61% | +11.99 pts |
| high | 2744 | 42.75% | 42.78% | +0.04 pts |
