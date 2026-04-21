# ARISE Automatic-Relational-Inference-Structural-Evaluation

A Cognitive-AGI Benchmark measuring structured reasoning and learnability
in AI models (Gemini,Claude,GPT, Deepseek), inspired by cognitive and computational neuroscience

Ran in Kaggle/local python notebook

## Tasks
### Transitive Inference (TI)
Tests whether a model can infer the relative rank of two items never directly
compared by chaining through items it has seen. Measures two classic signatures:
- **Symbolic Distance Effect**: accuracy rises with rank distance between items
- **Terminal Item Effect**: endpoint items are easier than interior pairs

### Learnability — Reward yoking design
Two interleaved 5-item stimulus sets. **Learnable (L)** stimuli have a **consistent
hidden ranking**. **Unlearnable (U)** stimuli have **random feedback indepedent of choice**: reward probability
yoked to the rolling mean of recent 10 L outcomes, equating average feedback while
removing any learnable structure. Key metric: **OP discrimination** = OP_L − OP_U
(slope of choose_frequency/stimulus ~ item_rank). A negative OP_disc (<-0.1) means the model
correctly distinguishes learnable from unlearnable stimulus structure.

## Reference
Tasks are adapted from:
- Monkey dACC electrophysiology experiments on learnability signals
  * Jin, Y., Jensen, G., Gottlieb, J., & Ferrera, V. (2022). Superstitious learning of abstract order from random reinforcement. Proceedings of the National Academy of Sciences, 119(35), e2202789119. https://doi.org/10.1073/pnas.2202789119
  
  * Jin, Y., Jensen, G., Ferrera, V., & Gottlieb, J. (2025). Single-neuron encoding of learnability in the dorsal anterior cingulate cortex. bioRxiv. https://doi.org/10.1101/2025.09.29.679390

- For any more questions, reach out to yj2525@columbia.edu/leoyuhaojin@gmail.com

## Kaggle Benchmark

https://www.kaggle.com/competitions/kaggle-measuring-agi/writeups/https

## Files

| File | Description |
|---|---|
| `benchmark-task-transitive-inference.py` | Transitive inference — SDE + TIE combined, help by Vivian Peng @vivianpengdev|
| `benchmark-task-learnability.py` | Learnability task — PN schedule, OP discrimination, help by Vivian Peng |
| `benchmark_writeup.md` | Full task descriptions and metric explanations |
