# H2GCN ablations versus full H2GCN, accuracy (paired, Holm corrected)

| dataset | condition | variant | pairs | mean Δ | SD Δ | 95% CI | t | p (raw) | p (Holm) | p (Wilcoxon) | d_z | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| chameleon | filtered | H2GCN[no_one_hop] | 10 | -0.0163 | 0.0229 | [-0.0327, +0.0001] | -2.2460 | 0.0513 | 0.0513 | 0.0547 | -0.7102 | not significant |
| chameleon | raw | H2GCN[no_one_hop] | 10 | -0.1064 | 0.0281 | [-0.1264, -0.0863] | -11.9904 | 0.0000 | 0.0000 | 0.0020 | -3.7917 | significant |
| squirrel | filtered | H2GCN[no_one_hop] | 10 | 0.0133 | 0.0156 | [+0.0022, +0.0245] | 2.7106 | 0.0240 | 0.0479 | 0.0293 | 0.8572 | significant |
| squirrel | raw | H2GCN[no_one_hop] | 10 | -0.0576 | 0.0241 | [-0.0749, -0.0404] | -7.5500 | 0.0000 | 0.0001 | 0.0020 | -2.3875 | significant |
| chameleon | filtered | H2GCN[no_two_hop] | 10 | 0.0118 | 0.0269 | [-0.0075, +0.0311] | 1.3834 | 0.1999 | 0.5357 | 0.1934 | 0.4375 | not significant |
| chameleon | raw | H2GCN[no_two_hop] | 10 | -0.0072 | 0.0202 | [-0.0217, +0.0072] | -1.1311 | 0.2873 | 0.5357 | 0.2656 | -0.3577 | not significant |
| squirrel | filtered | H2GCN[no_two_hop] | 10 | -0.0091 | 0.0097 | [-0.0160, -0.0021] | -2.9623 | 0.0159 | 0.0636 | 0.0273 | -0.9368 | not significant |
| squirrel | raw | H2GCN[no_two_hop] | 10 | 0.0058 | 0.0125 | [-0.0032, +0.0147] | 1.4590 | 0.1786 | 0.5357 | 0.1523 | 0.4614 | not significant |
