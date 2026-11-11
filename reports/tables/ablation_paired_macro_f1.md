# H2GCN ablations versus full H2GCN, macro_f1 (paired, Holm corrected)

| dataset | condition | variant | pairs | mean Δ | SD Δ | 95% CI | t | p (raw) | p (Holm) | p (Wilcoxon) | d_z | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chameleon | filtered | H2GCN[no_one_hop] | 10 | -0.0259 | 0.0379 | [-0.0531, +0.0012] | -2.1631 | 0.0588 | 0.1175 | 0.0840 | -0.6840 | not significant |
| chameleon | raw | H2GCN[no_one_hop] | 10 | -0.1197 | 0.0347 | [-0.1445, -0.0949] | -10.9111 | 0.0000 | 0.0000 | 0.0020 | -3.4504 | significant |
| squirrel | filtered | H2GCN[no_one_hop] | 10 | 0.0068 | 0.0235 | [-0.0100, +0.0236] | 0.9136 | 0.3847 | 0.3847 | 0.4922 | 0.2889 | not significant |
| squirrel | raw | H2GCN[no_one_hop] | 10 | -0.0601 | 0.0298 | [-0.0814, -0.0388] | -6.3742 | 0.0001 | 0.0004 | 0.0020 | -2.0157 | significant |
| chameleon | filtered | H2GCN[no_two_hop] | 10 | 0.0040 | 0.0353 | [-0.0213, +0.0292] | 0.3560 | 0.7301 | 0.7301 | 0.8457 | 0.1126 | not significant |
| chameleon | raw | H2GCN[no_two_hop] | 10 | -0.0071 | 0.0215 | [-0.0225, +0.0083] | -1.0434 | 0.3240 | 0.6480 | 0.3750 | -0.3299 | not significant |
| squirrel | filtered | H2GCN[no_two_hop] | 10 | -0.0088 | 0.0195 | [-0.0228, +0.0052] | -1.4228 | 0.1885 | 0.5656 | 0.2754 | -0.4499 | not significant |
| squirrel | raw | H2GCN[no_two_hop] | 10 | 0.0135 | 0.0167 | [+0.0016, +0.0255] | 2.5587 | 0.0308 | 0.1230 | 0.0273 | 0.8091 | not significant |
