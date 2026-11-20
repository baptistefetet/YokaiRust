# Five promotions from generation 64

## Protocol recorded before attempt 65

Generation 64 was the accepted champion when this experiment began. Its
weights match the hashes in the [previous experiment](../2026-11-17/README.md).
Keep this checkpoint for the final match. Continue with its saved Adam
optimizer state and the existing replay buffer at training revision
`f55a6c4eb185ec14bf93dcd658853a045f831083`. The SHA-256 of
`config/training.toml` is
`a86bdc52b5f92a94f5a0d9e76803618a65ac0ff5b31626311248f2a9fadec20c`.

| Generation 64 file | SHA-256 |
| --- | --- |
| model.safetensors | `7aeefdc9e680f6cd593a21f7fd109192c04e291ab32191c6f93528e0914c5c81` |
| metadata.json | `77e5e3b315aa34db9077d6b55ea68b798d9549592e0db060ba073d02eef10e7a` |
| training-model.bin | `7e2558bf25d0d45f70adc8c5b280d6d8d4ab7fdf7df9c8b8948c4cd3cb0f256a` |
| optimizer.bin | `659f702c90ad1ebc05b50c883eced46a06866510a7f94bc278202b5fb9016365` |

The initial stopping rule was ten promotions after generation 64. Each
candidate plays 200 paired games against its source champion. Promotion
requires a score of at least 55%, a one-sided paired p-value at most 0.05
and at most 20% draws in a separate 64-game noisy self-play probe.

Play a single independent match between the champion at the stopping point and
generation 64. Use 400 games in 200 pairs, with both models playing the same opening from
opposite seats. Reserve opening seeds 20262000–20262199 before training.
Use 0–4 random legal opening plies, 400 search simulations per move, eight
leaves per inference and no search noise. A win earns one point and a draw
earns half a point. Report wins, losses, draws, scores by seat and the
one-sided paired p-value. A score above 50% with p ≤ 0.05 is evidence that
the new champion is stronger under these match conditions.

## Stopping rule amended during attempt 80

With four promotions recorded (66, 72, 73 and 76), the requested training
budget changed to **five promotions**. Stop at the fifth, then play the
previously specified match against generation 64. Publish a new web release
if the fifth champion wins the match by scoring more than 200 points.
Report the paired test alongside the score.
