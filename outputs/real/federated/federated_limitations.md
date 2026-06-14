# Federated Learning Limitations

- These experiments are simulated federated learning on one machine using partitioned ClinVar data.
- They demonstrate training and aggregation feasibility but do not prove deployment-level privacy, secure aggregation, encrypted transport, authentication, or resistance to model inversion.
- Clients are synthetic partitions rather than real hospitals or laboratories.
- Non-IID heterogeneity is label-skew simulation; it does not reproduce every institutional ascertainment pattern.
- A shared preprocessing transform is fit on the global training partition so all simulated clients use one feature space; a deployment would need fixed encoders or federated aggregation of preprocessing statistics.
- The global test set is a balanced random ClinVar-derived holdout, not temporal, gene-held-out, or external validation.
- FL-B and FL-C include prior predictor fields and therefore have high circularity risk.
- FL-C is limited to 5k, 10k, and 20k because exact SpliceAI sidecars were already available only for those samples.
- SwissProt/HUMSAVAR remains external validation and is not used in client or server training.
- Communication volume, client dropout, adversarial clients, differential privacy, and secure aggregation are not modeled.
