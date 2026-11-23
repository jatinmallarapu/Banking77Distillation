# Complete-label likelihood scoring

Evaluation scores every one of the 77 intent strings, including its EOS token. A greedy sample is not substituted for these scores. Confidence is the softmax of these sequence scores, with temperature and abstention selected on validation data.

The student implementation calculates each query prefix once. It copies and repeats the prefix KV cache for groups of 16 candidate labels. The prefix predicts the first label token; suffix logits predict the rest of the label and EOS. Query batches use left padding, explicit logical positions and attention masks. Batch size 32 therefore processes multiple query prefixes together rather than reporting 32 sequential calls as a parallel batch.

`StudentScorer.score_reference` retains the uncached full-sequence implementation for checks. `python scripts/verify_decoder.py` compares both implementations and separate versus batched queries using a tiny random FP32 Qwen2 transformer, including unequal query and label lengths. This checks the math, not trained-model quality. On the allocated T4, the maximum absolute log-probability difference was 0.0000038147 across three synthetic queries and four labels. That is a software verification result, not a Banking77 accuracy benchmark.

An earlier three-query check of the real FP16 Student A adapter produced identical predicted labels but maximum log-score difference 0.37361 between differently shaped matrix operations. FP16 scores are not claimed to be bitwise identical. Held-out quality and confidence must be measured through the same released scoring path. All benchmark reports retain the decoding definition and batch size.

Reported input token counts for the cached student count each logical prefix once; output token counts count all candidate label tokens. vLLM reports its own request token usage, which can count cached prefixes again. These are accounting scopes, not directly comparable amounts of GPU work. Cost is derived from measured wall time and the explicitly supplied GPU price, rather than inferred from token counts.
