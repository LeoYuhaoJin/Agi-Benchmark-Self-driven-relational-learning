# Agi-Benchmark-Self-driven-relational-learning# Cognitive AGI Benchmark

A Kaggle Community Benchmark measuring structured reasoning and learnability
in AI models, adapted from primate and computational neuroscience paradigms.

## Tasks

### Transitive Inference (TI)
Tests whether a model can infer the relative rank of two items never directly
compared by chaining through items it has seen. Measures two classic signatures:
- **Symbolic Distance Effect**: accuracy rises with rank distance between items
- **Terminal Item Effect**: endpoint items are easier than interior pairs

### List Linking (LL)
Two separate 4-item ranked lists are learned independently. A single linking
probe (D > E) connects them. Tests whether the model can infer cross-list
rankings requiring multi-step inference through the probe link.

### Learnability — Reward yoking design
Two interleaved 5-item stimulus sets. Learnable (L) items have a consistent
hidden ranking. Unlearnable (U) items use the PN schedule: reward probability
yoked to the rolling mean of recent L outcomes, equating average feedback while
removing any learnable structure. Key metric: **OP discrimination** = OP_L − OP_U
(slope of choose_frequency ~ item_rank). A negative OP_disc means the model
correctly distinguishes learnable from unlearnable stimulus structure.

## Background

Tasks are adapted from:
- Monkey dACC electrophysiology experiments on learnability signals
- Plastic RNN (RetroModulRNN) superstitious learning simulations
- Classical transitive inference and list-linking paradigms from comparative cognition

## Kaggle Benchmark

[Link to Kaggle benchmark] <!-- add your benchmark URL here -->

## Files

| File | Description |
|---|---|
| `task_ti_sde_tie.py` | Transitive inference — SDE + TIE combined |
| `task_list_linking.py` | List linking — non-probe cross-list inference |
| `task_learnability_pn.py` | Learnability task — PN schedule, OP discrimination |
| `benchmark_writeup.md` | Full task descriptions and metric explanations |
