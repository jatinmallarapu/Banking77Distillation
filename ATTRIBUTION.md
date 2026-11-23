# Data and model attribution

Banking77 is by PolyAI and is distributed under **Creative Commons Attribution
4.0 International**. Source: https://huggingface.co/datasets/PolyAI/banking77
at revision `1846984189f2bc0ac51c6fc3a600fa9e1714e0c6`.
License: https://creativecommons.org/licenses/by/4.0/

Casanueva, I., Temčinas, T., Gerz, D., Henderson, M., and Vulić, I. (2020).
*Efficient Intent Detection with Dual Sentence Encoders.*
https://arxiv.org/abs/2003.04807

This project redistributes a 300-query gate subset with original intent labels
and stable source row IDs. It creates a stratified train/validation split, removes
11 duplicate/overlapping training records, and applies identifier masking before
model input or prediction logging. Official test membership and labels are unchanged.
Future teacher paraphrases are adaptations of the training queries and must retain
this attribution and an explicit statement of modification when redistributed.

Qwen2.5 models are by the Qwen team, Alibaba Cloud. The base student and AWQ teacher
cards identify **Apache 2.0** licensing:

- https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-AWQ

No Qwen weights or generated training dataset are redistributed in the initial
GitHub repository. Preserve the base model's license and notices in a future merged
model release. Authored project code has a separate MIT license. There is no
affiliation with Artefact, PolyAI, or Qwen.
