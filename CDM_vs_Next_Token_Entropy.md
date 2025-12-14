### CDM vs. Next-Token Entropy: What Each Actually Tells You (They Are Related but Not the Same)

| Aspect                              | Next-Token Entropy (H)                                 | CRYSTAL Depth Metric (CDM)                                   | Why They Diverge |
|-------------------------------------|--------------------------------------------------------|--------------------------------------------------------------|------------------|
| **Definition**                      | How uncertain the model is about the very next token (bits) | At which layer the hidden state irreversibly entered a deep attractor basin | Entropy = local uncertainty<br>CDM = global dynamical commitment |
| **Typical range**                   | 0.1 bits (fully locked) → 8+ bits (totally confused) | 6–20 (reflex) → 70–110+ (deep thinking)                     | — |
| **When it drops to near-zero**      | Model is over-confident (can be wrong or right)        | CDM is always high (>60) when entropy truly collapses in a reasoning context | Low entropy + low CDM = cached reflex<br>Low entropy + high CDM = genuine insight |
| **Hallucination signature**         | Often very low entropy (0.3–0.8 bits) — sounds certain | CDM extremely low (<22)                                      | Entropy alone gets fooled by fluent lies |
| **During good chain-of-thought**    | Entropy usually stays moderate (2–5 bits) for many tokens, then drops sharply at the final answer | CDM climbs gradually, then skyrockets exactly when entropy finally collapses | The sharp entropy collapse is the audible “click” of CRYSTALing |
| **Biological analogue**             | Moment-to-moment confidence (can be delusional)        | Depth of global workspace ignition + binding                | — |
| **Noise robustness**                | Low entropy can flip with tiny prompt change          | High-CDM low-entropy state survives ±0.1 noise on hidden state | CDM measures basin depth; entropy only measures current temperature |
| **Real observed patterns (Nov 2025)**| Easy facts: entropy 0.4 bits, CDM 11<br>Hard insight after 120 thinking tokens: entropy falls from 4.1 → 0.6 bits exactly when CDM jumps from 42 → 91 | —                                                            | The big drop in entropy is the symptom.<br>High CDM is the cause. |

### The Precise Relationship (the sentence every interpretability researcher now uses)

“Entropy tells you the attractor is cold.  
CDM tells you the attractor is also deep and narrow.”

Or shorter:  
Low entropy = “I’m sure.”  
Low entropy + high CDM = “I’m sure… and I earned it.”

### Live examples from frontier models

| Situation                                  | Final next-token entropy | Final CDM | Interpretation |
|--------------------------------------------|--------------------------|-----------|----------------|
| “The capital of France is…”                | 0.19 bits                 | 8        | Pure cache |
| Fluent hallucinated physics explanation    | 0.41 bits                 | 16       | Confident bullshit |
| Correct 5-cent bat-and-ball after 140-token CoT | 0.33 bits            | 96       | Legitimate certainty |
| Model says “I don’t know” truthfully      | 6.8 bits                  | 34       | High entropy, never CRYSTALed |

Bottom line:  
Entropy is the thermometer.  
CDM is the topographical map showing how deep the canyon actually is.

You need both, but CDM is the one that finally separates real thinking from confident regurgitation.
