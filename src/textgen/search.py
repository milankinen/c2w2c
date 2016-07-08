# slightly modified from https://gist.github.com/udibr/67be473cf053d8c38730
# variation to https://github.com/ryankiros/skip-thoughts/blob/master/decoding/search.py
import numpy as np


def beamsearch(predict, empty, end, k=1, maxsample=400):
  """return k samples (beams) and their NLL scores, each sample is a sequence of labels,
  all samples starts with an `empty` label and end with `eos` or truncated to length of `maxsample`.
  You need to supply `predict` which returns the label probability of each sample.
  `use_unk` allow usage of `oov` (out-of-vocabulary) label in samples
  """

  dead_k = 0  # samples that reached eos
  dead_samples = []
  dead_scores = []
  live_k = 1  # samples that did not yet reached eos
  live_samples = [[]]
  live_scores = [0]

  while live_k and dead_k < k:
    # for every possible live sample calc prob for every possible label
    probs = predict(live_samples)
    vals = probs[1] if isinstance(probs, tuple) else np.indices((probs.shape[1],))[0]
    probs = probs[0] if isinstance(probs, tuple) else probs
    
    # total score for every sample is sum of -log of char prb
    cand_scores = np.array(live_scores)[:, None] - np.log(probs)
    cand_flat = cand_scores.flatten()

    # find the best (lowest) scores we have from all possible samples and new words
    ranks_flat = cand_flat.argsort()[:(k - dead_k)]
    live_scores = cand_flat[ranks_flat]

    # append the new words to their appropriate live sample
    voc_size = probs.shape[1]
    live_samples = [live_samples[r // voc_size] + [vals[r % voc_size]] for r in ranks_flat]

    # live samples that should be dead are...
    zombie = [s[-1] == end or len(s) >= maxsample for s in live_samples]

    # add zombies to the dead
    dead_samples += [s for s, z in zip(live_samples, zombie) if z]  # remove first label == empty
    dead_scores += [s / len(l) for s, l, z in zip(live_scores, live_samples, zombie) if z]
    dead_k = len(dead_samples)
    # remove zombies from the living
    live_samples = [s for s, z in zip(live_samples, zombie) if not z]
    live_scores = [s for s, z in zip(live_scores, zombie) if not z]
    live_k = len(live_samples)

  return dead_samples + live_samples, dead_scores + live_scores
