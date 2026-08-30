# Extending MoLAG

The public interfaces preserve the main extension points used in the implementation.

## Tracker architectures

Implement `TrackerGeometryBase` and `TrackerCodeBase`, combine them through
`TrackerBase`, and register the concrete tracker. Dataset generation resolves tracker
definitions through the shared registry, so a new architecture does not require
changes to the dataset core.

## Loss components

Subclass `AffinityLossComponentBase` for a new objective term. The component receives
an affinity-loss context and returns its scalar contribution. Define a smaller context
when the term needs only a subset of the full MoLAG batch state, or use
`FullAffinityLossContext` for the existing structured terms.

## Evaluation metrics

Subclass `MetricsBase` and register the metric class. Metrics consume prediction
batches incrementally, which keeps evaluation and threshold calibration independent
of a particular metric collection. Saved prediction caches allow a newly implemented
metric to be applied to an existing run.
