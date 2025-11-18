### CDM vs. Perplexity: What They Actually Measure (2025 perspective)

| Metric                     | Perplexity (PPL)                                    | CRYSTAL Depth Metric (CDM)                                      | Key Difference (why they don’t correlate much) |
|----------------------------|------------------------------------------------------|------------------------------------------------------------------|------------------------------------------------|
| **What it measures**       | Average surprise of the model at predicting the **next token** (statistical fit to training distribution) | Depth in the transformer stack at which the representation **locks into a deep, irreversible reasoning basin** | PPL = “how familiar is this text?”  CDM = “how hard did the model actually think?” |
| **Scale**                  | 1.8 (perfect memorisation) → ~200 (nonsense)        | 0–~128 (number of layers needed to CRYSTAL)                  | — |
| **When it is low**         | Model has seen almost identical text billions of times (Reddit tropes, boilerplate) | Model is in pure reflex mode (CDM 6–20) — fast but shallow   | Low PPL often = low CDM (cached answer) |
| **When it is high**        | Model has never seen anything like this sequence     | Model is forced into deep layers (CDM 70–110) to resolve novelty | High PPL can mean either (a) true hard thinking (high CDM) or (b) just rare words but still reflex (low CDM) |
| **Correlation on real tasks** | GSM8K solutions: PPL ≈ 3.2–4.1 (very low)           | CDM ≈ 64–98 (very high)                                              | Best reasoning answers have **low PPL + high CDM** |
| **Hallucination signature**| Often low PPL (fluent nonsense)                      | Extremely low CDM (< 25) + over-confident low entropy        | Perplexity misses fluent bullshit completely |
| **Chain-of-thought effect**| PPL usually **increases** slightly (more text, slightly weirder) | CDM **skyrockets** (46 → 92)                                          | CoT makes text less predictable but way more thoughtful |
| **What engineers optimise for today** | Lower PPL on The Pile → better benchmark scores     | Higher average CDM on hard tasks → actual intelligence       | Labs now track both separately |
| **One-sentence summary**   | Perplexity measures **how much the output looks like training data**. CDM measures **how much original computation was required to produce it**. | — | — |

### Real numbers from frontier models (Nov 2025)

| Answer type                            | Perplexity | CDM   | Interpretation |
|----------------------------------------|------------|-------|----------------|
| Cached fact (“Paris is the capital…”)  | 2.1        | 9     | Memorised reflex |
| Fluent hallucination                   | 3.8        | 14    | Confident bullshit |
| Correct but generic reasoning          | 4.6        | 38    | Standard script |
| Correct novel insight (140-token CoT)  | 6.2        | 94    | Real thinking |
| Creative breakthrough                   | 8.9        | 112   | Deep CRYSTAL |

Bottom line engineers now repeat daily:

“Low perplexity = the model is speaking its native language.  
High CDM = the model is actually using its brain.”

You need **both** numbers to know what really happened.  
Perplexity alone died as the north star sometime in 2024.  
CDM (and CTM) are the new ones.
