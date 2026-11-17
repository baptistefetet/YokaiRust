# Three promotions from the published 5×6 champion

## Protocol recorded before attempt 58

The reference is generation 53 of `models/expert-5x6`, distributed with
[web-v2.2.1](https://github.com/baptistefetet/YokaiRust/releases/tag/web-v2.2.1).
The local weights match the weights in the release archive. Retain generation
53 throughout training and evaluation.

| Reference file | SHA-256 |
| --- | --- |
| Release champion archive | `574777b4cdcf3ed3ca24ca4313e478a25d2b8bf6a613254ae3c41b55f9f15f8f` |
| model.safetensors | `7cc61a3ce24f423516b9fe85bab97efe247117381863e806be4daf15bc3ae4c2` |
| metadata.json | `651656c66836d8e4d35b4f91a0d508eba5f6dafb3272dcb6660170ecf8e4b65e` |

Resume after completed attempt 57, with the accepted champion’s Adam state
and the existing replay buffer. The training revision is `fc2b219`.
The SHA-256 of `config/training.toml` is
`a86bdc52b5f92a94f5a0d9e76803618a65ac0ff5b31626311248f2a9fadec20c`.
The only change to the learning objective is that neutral draw supervision
preserves all original visits for both players. Previously a position in a
drawn game could discard a third-repetition move for the player who moved
second. All search budgets, optimizer settings and promotion gates carry over.

Stop at the third new promotion. Each candidate must score at least 55% in
its 200-game paired arena, attain one-sided paired p ≤ 0.05 and draw at most
20% of its 64 noisy probe games. Play the resulting champion once against
generation 53 over 400 games, using 200 color-swapped opening pairs, 400
simulations per move, eight leaves per inference, no search noise, 0–4 random
opening plies and base seed `20261750`. A win earns one point and a draw half.
Publish a new web release if the new champion scores more than 200 points.
Report the paired p-value separately to quantify the evidence of a strength gain.
