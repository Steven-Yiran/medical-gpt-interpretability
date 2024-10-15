## Background

To perform causal analysis in a language model G, We view G as a causal graph, where attention heads and MLPs are nodes in this graph. Edges are implicitly given by a direct path between the nodes.

## Run Experiments
Activate environment
```bash
python3 -m experiments.causal_trace
```

```bash
python3 -m experiments.inference --interactive
```