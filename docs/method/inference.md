# Inference

Inference applies the same coordinate normalisation and graph construction as
training. The model predicts pairwise affinities, a calibrated threshold converts
them to graph edges, and connected components become candidate tracker groups.

Threshold calibration is performed on a held-out frozen dataset. Every configured
threshold is evaluated for the selected objective metric, and ties are resolved in
favour of the higher threshold. Evaluation can alternatively use an explicit
threshold.

The prediction cache stores model outputs independently from metric results. This
allows new metrics or thresholds to be evaluated later without rerunning the model.
