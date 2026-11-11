# Raw versus filtered accuracy (descriptive)

| dataset | model | raw | filtered | Δ (filtered − raw) | n raw | n filtered |
|---|---|---|---|---|---|---|
| actor | MLP | 0.3514 | 0.3577 | 0.0063 | 10 | 10 |
| actor | GCN | 0.2978 | 0.3007 | 0.0029 | 10 | 10 |
| actor | GAT | 0.2936 | 0.2964 | 0.0028 | 10 | 10 |
| actor | H2GCN | 0.3570 | 0.3633 | 0.0064 | 10 | 10 |
| chameleon | MLP | 0.5009 | 0.3757 | -0.1252 | 10 | 10 |
| chameleon | GCN | 0.6246 | 0.3592 | -0.2654 | 10 | 10 |
| chameleon | GAT | 0.6314 | 0.3875 | -0.2439 | 10 | 10 |
| chameleon | H2GCN | 0.6048 | 0.4133 | -0.1915 | 10 | 10 |
| squirrel | MLP | 0.3329 | 0.3474 | 0.0146 | 10 | 10 |
| squirrel | GCN | 0.4626 | 0.3468 | -0.1159 | 10 | 10 |
| squirrel | GAT | 0.4316 | 0.3442 | -0.0874 | 10 | 10 |
| squirrel | H2GCN | 0.4212 | 0.3789 | -0.0423 | 10 | 10 |
