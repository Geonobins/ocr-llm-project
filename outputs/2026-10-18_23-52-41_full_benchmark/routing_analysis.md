# Routing threshold analysis

Source runs: `B-allfast` (every document on the FAST 4-bit route) and `B-allfull` (every document on the FULL fp32 route).
Documents: 600. Field accuracy is averaged over all runs of each configuration.

## Reference policies

| Policy | FAST share | Field accuracy | Mean LLM time (s) |
|---|---|---|---|
| all FULL | 0.00 | 0.9361 | 0.1151 |
| all FAST | 1.00 | 0.9361 | 0.1154 |

## Accuracy by token count bin

| token_count   |   documents |   acc_fast |   acc_full |   llm_fast |   llm_full |
|:--------------|------------:|-----------:|-----------:|-----------:|-----------:|
| (0, 100]      |         306 |     0.9749 |     0.9771 |  0.0899905 |  0.0693502 |
| (100, 200]    |          74 |     0.8514 |     0.8468 |  0.118592  |  0.146411  |
| (200, 300]    |         129 |     0.8682 |     0.8656 |  0.148514  |  0.166959  |
| (300, 400]    |          56 |     0.9702 |     0.9702 |  0.152095  |  0.171252  |
| (400, 600]    |          35 |     0.9714 |     0.9714 |  0.149711  |  0.167894  |

## Accuracy by OCR confidence bin

| confidence   |   documents |   acc_fast |   acc_full |
|:-------------|------------:|-----------:|-----------:|
| (0, 80]      |           2 |     1      |     1      |
| (80, 85]     |           2 |     0.6667 |     0.6667 |
| (85, 90]     |          52 |     0.9231 |     0.9231 |
| (90, 95]     |         472 |     0.9364 |     0.935  |
| (95, 100]    |          71 |     0.9624 |     0.9718 |

Documents where FAST beats FULL: 5; FULL beats FAST: 5.

## Chosen thresholds

Admissible policies keep field accuracy within 0.005 of the all-FULL policy (0.9311).
Among them the policy with the largest FAST share is `max_tokens_fast=525`, `min_confidence_fast=70.0` with FAST share 1.00, field accuracy 0.9361 and mean LLM time 0.1154s.

## Top 15 admissible policies

|   max_tokens_fast |   min_confidence_fast |   fast_share |   field_accuracy |   mean_llm_time_s |
|------------------:|----------------------:|-------------:|-----------------:|------------------:|
|               525 |                    70 |       0.9983 |           0.9361 |            0.1154 |
|               525 |                    75 |       0.9983 |           0.9361 |            0.1154 |
|               550 |                    70 |       0.9983 |           0.9361 |            0.1154 |
|               550 |                    75 |       0.9983 |           0.9361 |            0.1154 |
|               575 |                    70 |       0.9983 |           0.9361 |            0.1154 |
|               575 |                    75 |       0.9983 |           0.9361 |            0.1154 |
|               600 |                    70 |       0.9983 |           0.9361 |            0.1154 |
|               600 |                    75 |       0.9983 |           0.9361 |            0.1154 |
|               625 |                    70 |       0.9983 |           0.9361 |            0.1154 |
|               625 |                    75 |       0.9983 |           0.9361 |            0.1154 |
|               650 |                    70 |       0.9983 |           0.9361 |            0.1154 |
|               650 |                    75 |       0.9983 |           0.9361 |            0.1154 |
|               675 |                    70 |       0.9983 |           0.9361 |            0.1154 |
|               675 |                    75 |       0.9983 |           0.9361 |            0.1154 |
|               700 |                    70 |       0.9983 |           0.9361 |            0.1154 |
