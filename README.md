# Medical GPT Interpretability

## Background

To perform causal analysis in a language model G, We view G as a causal graph, where attention heads and MLPs are nodes in this graph. Edges are implicitly given by a direct path between the nodes.

## Run Experiments

Activate environment and install required dependencies.

```bash
bash scripts/causal_trace.sh
```

## Todos

- [x] Integrate the PubMedQA dataset
- [ ] Proprocess data and implement corruption method
- [ ] Modify the activation patching framework for the new task
- [ ] Design and run age-information experiment
