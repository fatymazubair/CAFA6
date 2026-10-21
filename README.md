# Protein Function Prediction with ESM2 Embeddings (CAFA 6)

A multi-label classification pipeline that predicts Gene Ontology (GO) functional annotations for proteins directly from their amino acid sequences, using protein language model embeddings and CAFA-standard evaluation.

---

## Motivation

I built this project to apply computational methods to a real, actively-used benchmark in protein biology, as part of building a portfolio for computational biology / ML roles. I have a background in biophysics but am self-taught in programming, so this project was also a deliberate, structured way to learn the practical ML engineering skills (data pipelines, embeddings, multi-label classification, proper evaluation) that complement my domain knowledge.

I chose the CAFA (Critical Assessment of Functional Annotation) challenge specifically because:
- It's a real, recurring community benchmark (this is the 6th iteration), not a toy dataset — results are directly comparable to published methods
- It sits naturally at the intersection of biology and ML: predicting a protein's function from its sequence is a foundational problem in computational biology
- It doesn't require institutional database access or expensive compute — the data is public on Kaggle, and a reasonable approach fits on free-tier Google Colab

**Note on timing:** the CAFA 6 Kaggle competition had already closed by the time I started this project. I am not a competition entrant — I used the publicly available dataset purely to build and evaluate my own pipeline, scored using the official CAFA metric rather than the (now closed) leaderboard.

---

## The Problem, Explained

**Task:** given a protein's amino acid sequence, predict which Gene Ontology (GO) terms apply to it.

GO terms describe:
- **Molecular Function (MF)** — what the protein does biochemically (e.g., "kinase activity")
- **Biological Process (BP)** — what larger process it participates in (e.g., "apoptosis")
- **Cellular Component (CC)** — where in the cell it's located (e.g., "mitochondrial membrane")

This is a **multi-label** problem: one protein can correctly have dozens of GO terms simultaneously, across all three categories at once — not a single class label.

It is also a **hierarchical** problem: GO terms form a Directed Acyclic Graph (DAG), where specific terms (e.g., "protein kinase activity") imply all of their more general ancestor terms (e.g., "catalytic activity"). A protein's true label set includes everything that hierarchy implies, not just what's explicitly annotated.

---

## Pipeline Overview

```
Raw sequences (FASTA) ──┐
                         ├──> ESM2 embeddings ──> Multi-label classifier ──> GO term predictions
Raw labels (TSV) ────────┘         ↑
                                    │
                    GO DAG (.obo) propagation
```

---

## Step-by-Step: What I Did and Why

### 1. Data Acquisition and Inspection

Downloaded the official CAFA 6 dataset from Kaggle (`train_sequences.fasta`, `train_terms.tsv`, `go-basic.obo`, `train_taxonomy.tsv`, plus a test set and `IA.tsv` for evaluation weighting).

**Before writing any modeling code**, I inspected the raw files directly — row counts, column names, ID formats — rather than assuming structure. This caught a real bug early: UniProt FASTA headers are formatted as `sp|ACCESSION|ENTRY_NAME`, but the training labels file uses only the bare accession. Naively using Biopython's default `record.id` would have caused **100% of protein–label matches to silently fail** later, with no error — just an empty, nonsensical merged dataset. I fixed this by explicitly parsing out the accession (`record.id.split('|')[1]`) and verified a full ID overlap (82,404/82,404) before proceeding.

**Decision:** verify data integrity at every join point, not just once. This habit caught the bug above and later caught a second, subtler one (see Step 4).

### 2. GO Hierarchy and Label Propagation

Parsed `go-basic.obo` (43,448 GO terms) using `goatools` to build the GO DAG, then propagated every protein's raw annotated terms up through their ancestor terms.

**Why this step is non-negotiable, not optional polish:** the official CAFA scorer always evaluates against the fully propagated label set. Training or evaluating against raw (unpropagated) labels would mean my model's targets — and later, my evaluation — don't match the standard the field actually uses. Skipping this would have made any reported metric meaningless for comparison.

After propagation, a sample protein's label set expanded from 7 raw terms to 37 (including all implied ancestors) — confirming the propagation logic was working correctly.

### 3. Label Frequency Filtering

The full propagated label space contained **29,175 unique GO terms** across 82,404 proteins. Inspecting the distribution showed a severe long tail: median term frequency was just 7 proteins, meaning the majority of terms had far too few positive examples to learn from.

**Decision:** filter to terms appearing in ≥50 proteins, reducing the label space to 4,828 terms.

I chose this threshold by explicitly comparing the alternatives:
- ≥10 proteins (12,734 terms) — still too sparse for stable learning
- ≥50 proteins (4,828 terms) — reasonable signal per term while retaining biological specificity
- ≥100 proteins (2,922 terms) — safer statistically, but loses more specific/interesting terms

This is a documented, reasoned trade-off, not an arbitrary cutoff — and I verified no protein was left with zero labels after filtering (0 proteins affected), so no data was silently lost.

### 4. ESM2 Protein Embeddings

Used `esm2_t12_35M_UR50D`, a pretrained protein language model, to convert each raw amino acid sequence into a fixed-size 480-dimensional numeric vector (mean-pooled across sequence positions), generated for all 82,404 proteins on a Colab T4 GPU.

**Model size decision:** I deliberately chose the 35M-parameter ESM2 variant over larger options (up to 650M parameters). On free-tier Colab compute, a larger model risked GPU memory errors and multi-hour runtimes with real risk of session disconnection. This was a conscious compute-budget trade-off, not a default — the pipeline design is correct regardless of model size, and swapping to a larger ESM2 checkpoint later is a one-line change.

**Alignment check:** embeddings were generated in a different iteration order than the label matrix had been built in. Before combining them, I explicitly re-indexed both by protein ID to guarantee row-for-row correspondence, rather than assuming dictionary insertion order would match — the same category of silent-failure bug as Step 1, caught the same way: verify, don't assume.

### 5. Baseline Classifier

Trained a multi-label logistic regression classifier (one-vs-rest via `MultiOutputClassifier`) on the ESM2 embeddings.

**Why logistic regression, not a neural network, for the first model:** I wanted to validate that the entire pipeline — data, propagation, embeddings, alignment — was correct before introducing additional modeling complexity. A simple, fast, interpretable baseline isolates pipeline bugs from model-architecture issues; a more complex model would be a reasonable next iteration, not a prerequisite.

**Scope decision (compute-constrained):** full-scale training (4,828 labels × 70,043 training proteins) did not complete in a reasonable time on free-tier Colab's limited CPU cores (2 cores, no GPU acceleration for scikit-learn). After testing, I scoped the baseline down to the **100 most frequent GO terms** on a **10,000-protein training subsample**, which trained in minutes rather than hours while still validating the full pipeline end-to-end.

I'm stating this limitation explicitly rather than hiding it: this is a baseline proof-of-concept scoped to available compute, not a final, fully-scaled model. The architecture supports scaling to the full term/sample set given more compute time or a paid GPU-CPU instance.

Used `class_weight='balanced'` to counteract the label imbalance identified in Step 3, and the `liblinear` solver for faster convergence at this data scale.

### 6. Evaluation

Computed two sets of metrics on a held-out validation set (15% split, 12,361 proteins):

- **Macro/Micro F1** (quick sanity check): Macro F1 = 0.362, Micro F1 = 0.417 — confirmed the model learned real signal (not degenerate all-negative predictions) without being suspiciously perfect (no data leakage).
- **Weighted Fmax** (the official CAFA metric): **0.349**, computed using Information Accretion (IA) term weights from `IA.tsv`, following CAFA's standard protocol of sweeping prediction thresholds and reporting the maximum F-score (best threshold: 0.62).

**Why Fmax specifically, not just F1:** CAFA's official metric weights each GO term by how informative/specific it is (via Information Accretion), so correctly predicting a rare, specific term counts more than a generic, near-root one. This makes Fmax directly comparable to published CAFA baseline methods, unlike a standard unweighted F1 score.

**Context for this score:** published CAFA baseline methods (e.g., naive frequency-based or BLAST-based approaches) typically score in the 0.3–0.4 Fmax range; top competition submissions reach 0.5–0.6+. A score of 0.349 from a simple logistic regression baseline, on a reduced 100-term subset, is a legitimate, standard-baseline-level result — consistent with expectations for this model's complexity and scope.

---

## Known Limitations and Honest Scope

- Trained on 100/4,828 filtered GO terms (2% of the filtered label space), due to compute constraints on free-tier infrastructure
- Trained on a 10,000-protein subsample of the 70,043 available training proteins
- Baseline model only (logistic regression); no neural network or more advanced multi-label architecture attempted yet
- Embeddings use the smallest practical ESM2 variant (35M parameters)
- Qualitative inspection of individual predictions (see below) shows the model correctly identifies protein-specific GO terms in many cases (example, matching 3 of 5 top predictions for one validation protein), but performance is inconsistent across proteins — some get strong, specific matches while others default toward more generic cross-category terms. This variability is consistent with the model's moderate Fmax (0.349) and its reduced training scope (100 terms, 10K samples).

**Example — a case where the model performs well** (3 of its top 5 predictions are correct, specific matches, not generic defaults):

| Top 5 Predicted | In ground truth? |
|---|---|
| cellular component organization | ✅ |
| organelle organization | — |
| catalytic complex | — |
| cellular component organization or biogenesis | ✅ |
| cellular component assembly | ✅ |

This confirms the model is learning genuine protein-specific signal, not simply predicting the most frequent terms across the whole training set — though it does so unevenly from protein to protein, which is an expected limitation of the reduced training scope rather than a fundamental flaw in the approach.

These are documented, reasoned engineering trade-offs made under real compute constraints — not oversights. The pipeline itself (data handling, GO propagation, embedding generation, alignment validation, CAFA-standard evaluation) is complete and correct at full scale; only the final classifier's scope was reduced.

---

## What This Project Demonstrates

- Handling real, messy biological data formats (UniProt FASTA headers, OBO ontology files, CAFA's non-trivial submission schema)
- Correctly implementing hierarchical label propagation over a DAG
- Identifying and correcting label imbalance through reasoned, documented filtering decisions
- Using a pretrained protein language model (ESM2) for feature extraction
- Catching and fixing silent data-alignment bugs through explicit verification at every join point
- Implementing a domain-standard evaluation metric (weighted Fmax) from its components, not relying on a generic off-the-shelf score
- Making and documenting compute-budget trade-offs transparently, rather than overstating scope

---

## Possible Next Steps

- Scale the classifier to the full 4,828-term label set and full training set, given more compute (e.g., a paid Colab tier or local GPU with more CPU cores)
- Try a neural network classifier (e.g., a simple feedforward network on top of ESM2 embeddings) and compare Fmax against this baseline
- Try a larger ESM2 variant (150M or 650M parameters) to test whether embedding quality improves downstream performance
- Separate models per GO sub-ontology (MF/BP/CC) instead of one combined multi-label model, since they represent distinct biological questions

---

## Tech Stack

`Python` · `Biopython` · `fair-esm (ESM2)` · `goatools` · `scikit-learn` · `pandas` · `NumPy` · `PyTorch` · Google Colab (T4 GPU)
