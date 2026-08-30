# Model and loss

The model normalises each scene for translation and scale, builds its complete graph,
and applies configurable EdgeConv blocks. A symmetric edge head combines the two node
embeddings for an unordered pair and predicts a same-tracker affinity.

The scaled-conjunction affinity loss is assembled from independent components:

- supervised contrastive embedding structure;
- within-tracker connectivity;
- between-tracker separation;
- spurious attachment suppression; and
- spurious bridge suppression.

Each component implements `AffinityLossComponentBase` and declares the context it
needs. `FullAffinityLossContext` computes the shared quantities required by the MoLAG
loss once per batch. This separates the numerical objective from its orchestration and
makes alternative loss combinations straightforward to implement.

Architecture widths and loss hyperparameters are nested under `model_args` and
`loss_args`. The defaults reproduce the MoLAG configuration supplied with this code;
individual values remain configurable for further experiments.
