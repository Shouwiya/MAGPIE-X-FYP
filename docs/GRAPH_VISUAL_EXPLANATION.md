# Graph Visual Explanation

MAGPIE-X uses a variant graph so each variant can be interpreted with local biological context instead of only as an isolated row in a table.

## Node Meaning

Each node is one exonic variant.

The node can carry features such as:

- Allele frequency.
- CADD and dbNSFP predictor scores.
- Conservation scores.
- Consequence type.
- AlphaMissense score and availability for missense variants.
- Optional tabular baseline predictions for residual mode.

Labels are not node features during training.

## Edge Meaning

Edges encode biological or annotation-based relatedness:

- **Same gene:** two variants affect the same annotated gene.
- **Same consequence:** two variants share the same consequence class, such as missense or frameshift.
- **kNN similarity:** two variants are close in numeric annotation space using AF, CADD, REVEL, SIFT, PolyPhen, GERP, phyloP, and phastCons.

These edges are capped per node. This keeps the graph small enough for reference 5k smoke tests and prevents high-frequency genes or consequences from creating a near-complete graph.

## Why GNN Helps

A GNN can aggregate signals from neighboring variants. For example, a missense variant with uncertain tabular probability may be adjusted if nearby variants in the same gene and similar annotation profile show consistent pathogenic or benign patterns in the training fold.

In residual mode, the GNN output is not the final answer directly. It becomes `delta_gnn`:

```text
z_final = z_tabular + delta_gnn
p_final = sigmoid(z_final)
```

## Sir-Friendly Visuals

The visualization script samples a small subgraph, usually 100 to 200 nodes. It should never attempt to draw the full graph.

Recommended visual encoding:

- Node color by label or consequence.
- Same-gene edges in one color.
- Same-consequence edges in another color.
- kNN similarity edges in a third color.

This gives a clear explanation of how MAGPIE-X connects tabular annotation evidence to graph context.
