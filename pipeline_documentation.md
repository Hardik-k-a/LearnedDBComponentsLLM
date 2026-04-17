# Learned Cardinalities Active Learning Pipeline
**Comprehensive Documentation**

The `training.pipeline` application serves as the core orchestration engine for evaluating Learned Cardinality Estimators (specifically MSCN) under an Active Learning framework. This pipeline completely automates dummy data generation, ground-truth label aggregation from PostgreSQL, feature encoding, neural network training, active query acquisition, and comprehensive graph-based statistical evaluation.

---

## 1. High-Level Architecture & Workflow
The pipeline execution is divided into **7 sequential steps**.

### Step 1: Database Connection
* **Module**: `config.db_config`
* The pipeline connects to a live PostgreSQL instance (default: `imdb` schema). Active transactions are instantiated to perform rapid operations (like `COUNT(*)`).

### Step 2: Materialized Samples Calculation
* **Module**: `labeling.bitmap_utils`
* To give the learned model a statistical summary of the underlying base tables, the pipeline utilizes a "vectorized bitmap" approach. Primary keys and foreign keys are dynamically read. For each table, a fast sequential subset (e.g., 1000 records) is pre-sampled into memory.

### Step 3: Query Generation (Unlabeled Pool)
* **Module**: `generation.query_generator`, `generation.format_converter`
* To train the AI, thousands of syntactically valid relational queries must be generated.
    * **Synthetic Mode (`--synthetic`)**: Fast, deterministic structural generation that recursively joins basic tables and guarantees valid join paths based on a hardcoded schema constraint.
    * **LLM Mode (`--model-name ollama`)**: Generates rich, highly varied queries via prompting an external language model wrapper (e.g. LLAMA/Ollama) and parses them back out using strict `SchemaValidator` constraints.

### Step 4: Initial Split & Labeling 
* **Module**: `labeling.db_labeler`
* Queries are partitioned into an **Unlabeled Pool** and a **Validation Set**. 
* The validation subset immediately has true cardinalities mapped by executing PostgreSQL `SELECT COUNT(*) FROM ...` loops.
* **Per-query Timeouts**: A timeout (e.g., `5000ms`) is enforced at the database level to prevent massive cross-joins from halting the pipeline globally.

### Step 5: Encoding & Bitmap Masking
* **Module**: `utils.encode`
* SQL strings cannot enter neural models directly.
* Vocabulary lists are constructed tracking every unique structural predicate (Tables, Joins, Operators). 
* **Feature Maps**: Categorical data is translated to `one-hot` tensors; exact numeric conditions (e.g., `< 2005`) are aggressively normalized using `build_column_min_max()`.

### Step 6: Active Learning Loop
This is the iterative core of the system. Rather than receiving $N$ labeled queries all at once (supervised baseline), the active learning framework only starts with a tiny fraction (e.g. 20% initialization) and *intelligently asks* for more query labels dynamically.
1. **Initialize**: A tiny base subset of the Unlabeled Pool is chosen randomly and passed to the Database Labeler to bootstrap basic model stability.
2. **Train Epochs**: The MSCN network evaluates backpropagation mapping string/bitmap tensors against Cardinality (using normalized log scales).
3. **Acquisition (If rounds continue)**: The model attempts to figure out what it *doesn't know*. 
    * **Random**: Default stochastic sampling.
    * **MC-Dropout**: The model infers across the unlabeled pool multiple times keeping the training dropout enabled. Queries exhibiting massive variance (`var(preds)`) are considered uncertain and fetched for labeling.
    * **Ensembles**: Fits $K$ parallel models iteratively; queries where the $K$ models heavily disagree are pulled for labeling.
4. **Append & Repeat**: Acquired queries get labeled, encoded, serialized to disk as `.bitmap`, and dumped over to the main `train_dataset`. Next round begins.

### Step 7: Final Metric & Evaluation Logging
* **Module**: `evaluation.pipeline_graphs`
* The pipeline drops extensive arrays: Validation Q-errors, Training Loss histories, Epoch trajectories, batch times. Every iteration creates visually aesthetic PDF/PNG outputs documenting:
    * Labeling Efficiency Curves vs Supervised baselines
    * Q-Error Cumulative Distributive Function (CDF) sweeps
    * Prediction v. Ground Truth log-scatter maps
    * Data Generation distributions (Keys, tables, aggregations)

---

## 2. Artifact Output Persistence
A single run creates a globally stamped output directory (`pipeline_results/YYYY-MM-DD_HH-MM-SS/`). Inside, it securely archives:

* `/graphs/` – Contains all 15 detailed MATPLOTLIB charts.
* `model.pt` – The final PyTorch model state dict from the concluding round.
* `learning_data.csv` – Serialized trace arrays of model median validation capabilities over time.
* `labeling_times.csv` – Exact timestamp deltas per learning iteration measuring cost/efficiency tradeoffs.
* `*_bitmaps.bitmap` – Intermediate serialized raw dictionary payloads mapping exact material schemas evaluated.

---

## 3. Invocation Configuration & CLI Parameters

You can trigger the pipeline directly via `python -m training.pipeline [arguments]`. 

### Key Flags:
* **--total-queries `<int>`**: Dictates the volume spanning across the un-labeled pool and evaluation validation set. Standard run uses `2500`.
* **--synthetic**: Boolean flag explicitly disabling LLM generation paths in favor of deterministic native synthetics mapping to `IMDB`. Default is False (meaning it attempts LLM calls).
* **--strategy `<string>`**: Determines Active Learning mechanism (`random`, `mc_dropout`, `ensemble`).
* **--acquire `<int>`**: The amount of labels actively pulled iteratively at the end of each Round.
* **--db-timeout `<int>`**: Enforced limit at the PostgreSQL level guaranteeing single complex SQL queries don't hang standard workflow completion.
* **--epochs `<int>`**: Local training convergence depth for `SetConv` loops per round.

## 4. Model Context (MSCN)
The pipeline specifically builds Multi-Set Convolutional Networks (MSCN) defined over in `mscn.model`. The data layer requires strict vector limits:
1. `max_joins` (dynamic array padding matching the biggest sequence)
2. `max_predicates` 
3. `max_tables`

All arrays are concatenated via standard DataLoader protocols locally (`torch.FloatTensor`). The output is fundamentally `unnormalized_label = exp(normalized_prediction * max - min + min)` to account for logarithmic volume differentials mapping database structures ranging from subsets of $10$ to queries covering sets of $25,000,000$.

## 5. Extensions
To introduce novel active learning heuristics (ex. Core-Set, Fisher Information), a new evaluation strategy must be appended under the **Active Learning Loop (Acquisition)** in `training.pipeline` (`line 647`). New mechanisms must extract the array tensors defined on `unlabeled_pool_idx`, evaluate scoring heuristics, and push back indexes mapping to the required `--acquire` length constraints.
