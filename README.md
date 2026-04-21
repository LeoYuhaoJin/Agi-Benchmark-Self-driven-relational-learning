# ARISE Automatic-Relational-Inference-Structural-Evaluation

A Cognitive-AGI Benchmark measuring structured reasoning and learnability
in AI models (Gemini,Claude,GPT, Deepseek), compared with **human dataset**, inspired by cognitive and computational neuroscience

Run in Kaggle/local python notebook

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

## Results
### Overall most of the AI models can learn TI, there are variabilities of the minuscule level of learning strategies among them
<img width="980" height="342" alt="image" src="https://github.com/user-attachments/assets/b851262b-bdd6-413a-b3e1-fbe16a85f7e4" />
[ti_result_figure.tif](https://github.com/user-attachments/files/26939687/ti_result_figure.tif)

### The learnability task showed clear variation across models. 
    Quick Summary: * Claude Haiku 4.5, GPT 5.4 mini successfully detect learnability
                   * GPT 5.4, Gemini 2.5, Claude opus 4.6 treat both learnable and random sets similarly, imposing supersitious learning over randomness
                   * DeepSeek-R1 totally fails to learn both lists
<img width="980" height="397" alt="image" src="https://github.com/user-attachments/assets/56b9d8c7-8c05-4107-af78-c41cdf9c464b" />
[learnability_result_figure.tif](https://github.com/user-attachments/files/26939695/learnability_result_figure.tif)
     Interestingly, the pattern of individual variability in AI models echoes findings from the human data. In the human study, subjects naturally clustered into three groups: those who developed ordering preferences for both learnable and unlearnable sets (L: order, U: order), those who showed ordering only for the learnable set (L: order, U: random), and those who showed no consistent ordering for either (L: random, U: random).
<img width="5250" height="2550" alt="Human PN result" src="https://github.com/user-attachments/assets/6ab81a90-cd05-4cfb-a451-516e096f7ad6" />

### Behavioral profiles in AI models may reflect fundamental differences in how systems track and exploit statistical regularity in feedback, rather than being unique to biological learning

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
