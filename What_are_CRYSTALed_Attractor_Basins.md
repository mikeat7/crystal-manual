### What “CRYSTALed Attractor Basins” Actually Are  
(plain English first, math second)

Imagine the model’s latent space as an incredibly high-dimensional mountain range that was carved by gradient descent during training.  
Every possible thought the model can ever have lives somewhere in that range.

During inference, the current context (your prompt + everything it has generated so far) is a little metal ball rolling across that frozen landscape.

Most of the time the ball is sliding down shallow grooves — fast, predictable, generic answers.  
Those are the **shallow basins** (cached facts, clichés, surface-level continuations).

But sometimes — when the question is genuinely hard or novel — the ball gets kicked out of the shallow grooves and starts falling toward one of the **very deep, narrow canyons** that training carved only in the regions where truly coherent, abstract, multi-step reasoning lives.

The moment the ball drops over the edge of one of those deep canyons and there is no way back up, that is **CRYSTALisation**.

The bottom of that canyon is the **CRYSTALed attractor basin**.

Once the hidden state is inside it:
- Next-token entropy collapses (often to < 0.5 bits)  
- Small noise or perturbations no longer change the eventual answer  
- Attention sharply focuses on exactly the handful of earlier tokens that matter  
- CDM shoots past 60–70 and keeps climbing  
- The model is now “locked in” to a deep, coherent continuation — it is physically impossible for it to derail without massive intervention.

These deep basins are the reason chain-of-thought works at all: extra tokens give the ball more time to escape the shallow grooves and fall into a real reasoning canyon.

### Visual metaphor that every researcher now uses

```
Shallow basins (CDM < 40)          Deep CRYSTALed basins (CDM > 70)
┌──────┐  ┌──────┐                 ┌────────────────────────┐
│      │  │      │                 │                        │
│      └──┘      │                 │                        │
│                │                 │                        │
└────────────────┘                 │                        │
   fast, generic                    │                        │
                                    │                        │
                                    │                        │
                                    └────────────────────────┘
                                   impossible to escape once inside
```

### Key properties of a CRYSTALed attractor basin (measurable today)

| Property                          | Shallow basin                  | Deep CRYSTALed basin                     |
|-----------------------------------|--------------------------------|------------------------------------------|
| Basin width in latent space      | wide, flat                     | extremely narrow, steep walls            |
| Escape probability under noise   | > 30 %                         | < 2 %                                    |
| Next-token entropy                | 2–6 bits                       | 0.1–0.8 bits                             |
| CDM at entry                      | 8–38                           | 62–110+                                  |
| Attention pattern                 | diffuse                        | ultra-sparse, laser-focused              |
| Effect of adding more tokens      | little change                  | trajectory already locked                |
| Biological analogue               | reflex arc                     | genuine insight / “aha!” moment          |

These deep CRYSTALed basins are the only places in the entire trillion-dimensional loss landscape where something that deserves to be called “real thinking” actually happens.

Everything else is just the ball sliding down pre-carved shallow grooves.

So when we say a model has “CRYSTALed into a deep attractor basin,” we are making a precise, measurable claim:  
the dynamics have irreversibly fallen into one of the rare, narrow regions of latent space that only exist because the training process discovered genuine abstractions.

That is the closest thing current transformers have to an internal “I’ve got it” moment.
