# vector-cache-optimizer

> Comparing learned eviction policies against LRU/LFU for embedding caches.

## The question I'm exploring

Belady's MIN algorithm is the theoretical optimum for cache eviction — but it's
clairvoyant (it requires knowing the future). LRU is the "good enough" stand-in
that most production systems use. There's a small but interesting body of work
(Google's "Learning Memory Access Patterns", Mihail's *LeCaR*, Berkeley's *Glider*)
showing that learned policies can close part of the gap to MIN on real workloads.

I wanted to know: does any of that translate to **embedding caches**, where the
"value" of a cached entry isn't just access frequency but also semantic
overlap with future queries?

## Why I care

Eviction is one of those problems that looks solved until you have a workload
that doesn't fit the textbook assumptions. RAG embedding caches don't fit:

- Entries aren't independent — semantically similar entries can substitute
  for each other on a near-miss
- Compute cost per miss is uneven (re-embedding a long document costs more than
  a short query)
- Access patterns have strong session locality but weak global locality

If a learned policy can save even 5–10% over LRU on this workload, that's a
real reduction in GPU time and serving cost — and a real reduction in the
carbon footprint of every retrieval-heavy product.

## What's in here

A test harness for comparing eviction policies on the same trace:

- `src/core/cache.py` — base async cache with pluggable eviction
- `src/policies/` — LRU, LFU, ARC, and an ML-driven policy that scores
  entries by predicted reuse probability
- `src/clustering/` — vector clustering used as a feature for the predictor
  ("entries in dense clusters are more likely to be reused")
- `src/benchmarks/` — replay harness for synthetic traces

Infra is the usual: Terraform for GCP, k8s manifests, Prometheus + Grafana
dashboards. I included it because eviction policy decisions are only
interesting if you can observe their effects in production.

## What I'm finding (so far)

Honest answer: not enough yet. What I do see on synthetic Zipf-distributed
access patterns:

- **LFU beats LRU** by ~3-5% hit rate on long sessions but loses on bursty
  ones (well-known result, replicating it was a good sanity check).
- The ML policy as currently written is **slower than LRU** at runtime
  because of the per-eviction inference call. Amortizing predictions over
  batches helps but I haven't quantified by how much.
- Vector clustering as a feature is suspicious — the cluster IDs themselves
  drift as the cache contents change. I think I need a stable feature
  (e.g., distance to a fixed set of anchor points) instead.

## What I'd do next

- Run on real traces (production logs from any RAG system would work; failing
  that, MTEB query distributions)
- Measure the **end-to-end** cost: hit rate × miss penalty − policy overhead.
  Hit rate alone is misleading.
- Try the LeCaR-style "two-experts" approach (LRU + LFU mixed by reinforcement
  learning) before going deeper on the neural predictor
- Compare against a clairvoyant Belady oracle on the same trace to see how much
  headroom is left

## Status

Early experiment. The harness runs and the policies are real, but I haven't
done the comparison study honestly enough to publish numbers in this README.

## References

- Liu et al., *LeCaR: Cache Replacement with Reinforcement Learning* (HotStorage 2018)
- Shi et al., *Applying Deep Learning to the Cache Replacement Problem* (MICRO 2019)
- Hashemi et al., *Learning Memory Access Patterns* (ICML 2018)
- Belady, *A study of replacement algorithms* (1966)
