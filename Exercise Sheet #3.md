# Big Data Analytics – Exercise Sheet #3
**Prof. Dr. Martin Theobald | Summer Semester 2026 – University of Luxembourg**
**Due: May 21, 2026 (before the lecture starts)**

---

## General Submission Instructions

- Upload all source code as a **single `.zip` file** to the Moodle assignment, using your **group ID as the file name**.
- For each problem, include a file called **`Problem X.txt`** with:
  - Brief instructions on how to run your solution.
  - The most important results (e.g., number of results, runtimes).
- Solutions may be submitted in **groups of up to 3 students**.
- **AI usage (new for 2026):** You are explicitly encouraged to use Generative AI tools. Indicate which AI platform you used and summarize the most important prompting steps together with your solution for each problem.
- All solutions will be checked for **completeness and functionality**.

> **Note:** This exercise sheet involves intensive computational workloads. It is **highly recommended** to run all experiments on the **IRIS HPC cluster** using either the interactive or batching modes of the Spark launcher scripts (see the "Getting Started" guide on Moodle for detailed instructions).

---

## Problem 1 – Latent Semantic Analysis of Wikipedia Articles *(12 Points)*

**Dataset:** `Wikipedia-En-41784-Articles.tar.gz` — available on Moodle under Chapter 1.
**Reference script:** `RunLSA.scala` or `RunLSA.ipynb` — provided on Moodle as a starting basis.

---

### Part (a) – Setup, Tokenization Variants & Top-25 Results *(4 Points)*

1. **Translate the provided script into a runnable Spark job** using one of the following two options:
   - **Option A (Scala):** Translate `RunLSA.scala` into an actual Scala object and compile it into a JAR file called `RunLSA.jar`, which can be executed via `spark-submit` on the IRIS cluster.
   - **Option B (Python):** Extract the plain Python code from `RunLSA.ipynb` using `nbconvert --to script`, and submit the resulting Python script as a Spark job on the IRIS cluster.

2. **Create two versions of your code** (in whichever language you chose above):
   - **Version 1 – NLP Pipeline:** Uses a full NLP pipeline for parsing and lemmatizing the Wikipedia articles (using the `plainTextToLemmas` function provided in the reference script/notebook).
   - **Version 2 – Simple Tokenizer:** Uses a simple regex-based tokenizer matching words and numbers, specifically the regular expressions introduced in **Exercise Sheet #1** (i.e., a pattern such as `[a-z]+` for words and `[0-9]+(\.[0-9]*)?` for numbers).

3. **Compute the following outputs for both versions**, following the examples provided in the reference notebook and as shown in the lecture:
   - The **top-25 terms** under each of the **top-25 latent concepts**.
   - The **top-25 documents** under each of the **top-25 latent concepts**.

4. **Observe and document** how the results differ between the NLP pipeline version and the simple tokenizer version. Note any qualitative differences in the concepts discovered.

5. **Report your findings** in `Problem 1.txt`, including example output excerpts and observations on differences between the two tokenization strategies.

---

### Part (b) – Hyperparameter Comparison on the Full Dataset *(4 Points)*

Using the **full dataset of all 41,784 Wikipedia articles**, run your Spark job (from Part (a)) with every combination of the following hyperparameters and compare the runtimes:

- **`numFreq`** (number of frequent terms extracted from all Wikipedia articles): `{5000, 10000, 20000}`
- **`k`** (number of latent dimensions for the SVD): `{25, 100, 250}`

This gives **9 total configurations** (3 × 3). For each configuration:

1. Run the job with both the **NLP pipeline version** and the **simple tokenizer version** (from Part (a)), so **18 runs** in total.
2. Record the **wall-clock runtime** for each run.
3. **Manually inspect the top-25 terms and top-25 documents under the top-25 latent concepts** (as computed in Part (a)) to qualitatively assess the quality of the results.
4. Based on this inspection, **identify the best combination of hyperparameters** (`numFreq`, `k`, and tokenizer choice) for your setting.

> **Note:** The configuration `numFreq = 20000` and `k = 250` should complete in **less than one hour** on the IRIS cluster for all 41,784 articles.

5. **Report your findings** in `Problem 1.txt`, including a table of runtimes for all configurations and a justification for your chosen best hyperparameters.

---

### Part (c) – LSA-Based Search Engine *(4 Points)*

Using the **best hyperparameter configuration** identified in Part (b):

1. **Load the best SVD model** you computed in Part (b).

2. **Implement a basic keyword search engine** that:
   - Takes a **list of keywords** as input (a user query).
   - Represents the query in the SVD's latent space by projecting it using the term-concept matrix \( V \) (refer to **Slides 34–39 of Chapter 4** of the lecture for the exact mathematical steps, which involve:
     - Looking up each keyword's term vector in \( U \Sigma \),
     - Summing or averaging these term vectors to form a query vector in concept space,
     - Computing cosine similarity between the query vector and each document vector in \( V^T \)).
   - Returns the **most relevant documents** (Wikipedia articles) ranked by their similarity to the query vector.
   - **Prints the titles** of the retrieved Wikipedia articles alongside their similarity scores.

3. **Run 5–10 interesting keyword queries** of your choice and report the full results (article titles + similarity scores) for each query.

4. **Report your findings** in `Problem 1.txt`, including the chosen queries, the retrieved article titles, and any observations on search quality.

---

## Problem 2 – Latent Semantic Analysis of Wikipedia Movie Plots *(12 Points)*

**Dataset:** "Wikipedia Movie Plots" — available from Kaggle at:
`https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots`
(also available via Moodle)

The dataset is in CSV format and contains **134,164 Wikipedia movie articles** with 8 fields:
`Release Year`, `Title`, `Origin/Ethnicity`, `Director`, `Cast`, `Genre`, `Wiki Page`, `Plot`.

**Reference script:** `RunLSA.scala` or `RunLSA.ipynb` — the same reference script/notebook used in Problem 1, provided on Moodle.

Follow the general LSA methodology from the reference script/notebook for all parts below.

---

### Part (a) – Data Ingestion and Schema Definition *(4 Points)*

1. **Adapt the parsing functions** of `RunLSA.scala` or `RunLSA.ipynb` to read the CSV dataset into a Spark DataFrame. Specifically:
   - Read **only the `Title`, `Genre`, and `Plot` fields** from the CSV file (the other 5 fields — Release Year, Origin/Ethnicity, Director, Cast, Wiki Page — can be ignored or kept, but `Title`, `Genre`, and `Plot` are required).
   - Apply a **proper schema** to the DataFrame using Spark's `StructType` / `StructField` definitions, specifying at minimum:
     - `Title`: `StringType`
     - `Genre`: `StringType`
     - `Plot`: `StringType`

2. Verify the DataFrame loads correctly (e.g., by calling `.show()` and `.printSchema()`).

3. **Report your findings** in `Problem 2.txt` (e.g., record count, schema printout, any parsing issues encountered).

---

### Part (b) – NLP Tokenization and Full Spark Job *(2 Points)*

1. **Add a new column called `features`** to the DataFrame from Part (a). This column must contain a **list of lemmatized text tokens** extracted from each article's `Plot` field, using the **NLP-based `plainTextToLemmas` function** provided in the reference `RunLSA.scala` script or `RunLSA.ipynb` notebook. The `features` column should be of type `Array[String]` (Scala) or a list of strings (Python).

2. **Translate the full pipeline** — including the new steps from Part (a) (CSV ingestion, schema, field selection) and Part (b) (NLP tokenization into `features`) — into an actual Spark job that can be executed via `spark-submit` on the IRIS cluster.
   - **Scala users:** Compile into a JAR and submit with `spark-submit`.
   - **Python users:** Submit the `.py` script with `spark-submit`.

3. **Report your findings** in `Problem 2.txt`, including a sample of the `features` column output and the runtime for the tokenization step.

---

### Part (c) – SVD Decomposition and Top-25 Inspection *(2 Points)*

Using the DataFrame with the `features` column from Part (b), compute an SVD decomposition of the **134,164 movie plot documents** with the following fixed hyperparameters:

- **`numFreq = 5000`** — number of frequent terms extracted from all movie plots.
- **`k = 25`** — number of latent dimensions for the SVD.

Then, using the **`topTermsInTopConcepts`** and **`topDocsInTopConcepts`** functions from the reference `RunLSA.scala` script or `RunLSA.ipynb` notebook, compute:

- The **top-25 terms** under each of the **top-25 latent concepts**.
- The **top-25 documents (movie plots)** under each of the **top-25 latent concepts**.

**Manually inspect** the results:
- Do the top terms per concept form coherent themes (e.g., action, romance, historical)?
- Do the top documents per concept correspond to movies that match those themes?
- Does the SVD appear to improve the semantic representation of the documents compared to raw term frequency?

**Report your observations** in `Problem 2.txt`.

---

### Part (d) – Genre Labels in Top Documents *(2 Points)*

**Extend the `topDocsInTopConcepts` function** (from the reference `RunLSA.scala` script or `RunLSA.ipynb` notebook, and as used in Part (c)) so that, for each document returned in the top-25 documents per concept, it **also prints the top-5 original `Genre` labels** associated with that Wikipedia movie article (as stored in the `Genre` field of the DataFrame from Part (a)).

Specifically:
- For each of the top-25 documents under each of the top-25 latent concepts, display:
  - The **document title** (movie title).
  - Its **similarity score** within that concept.
  - The **top-5 genre labels** associated with that movie (from the `Genre` field).
- If a movie has fewer than 5 genre labels, display all available genre labels.

**Report sample output** in `Problem 2.txt`, illustrating how genre labels align with the discovered latent concepts.

---

### Part (e) – Keyword Queries for Movies *(2 Points)*

Using the SVD model computed in Part (c) (with `numFreq = 5000`, `k = 25`), implement or reuse the search engine approach (projecting a keyword query into concept space and ranking documents by cosine similarity — following the same methodology described in **Problem 1, Part (c)** and **Slides 34–39 of Chapter 4**):

1. Think of **5–10 interesting keyword queries** relevant to movies (e.g., genre-based queries, plot-based queries, character-based queries).
2. Run each query against the SVD model and retrieve the most relevant movies.
3. **Report the results** in `Problem 2.txt`, including:
   - The query keywords.
   - The titles of the top retrieved movies.
   - Their similarity scores.
   - Any observations about retrieval quality.

---

## Problem 3 – Outlier Detection for Gearbox Readings *(12 Points)*

**Dataset:** "Gearbox Fault Detection" dataset released by NASA in 2009 — available from Kaggle at:
`https://www.kaggle.com/datasets/hetarthchopra/gearbox-fault-detection-dataset-phm-2009-nasa`
(also available via Moodle)

The dataset is in CSV format and contains approximately **14 million gearbox sensor readings**.

**Reference script:** `RunKMeans.scala` or `RunKMeans.ipynb` — provided on Moodle.

---

### Full Task Description *(12 Points)*

1. **Load the dataset** into Spark using the reference `RunKMeans.scala` script or `RunKMeans.ipynb` notebook. Apply or adapt the provided parsing functions as needed to correctly read the ~14M gearbox readings from the CSV file.

2. **Run K-Means clustering** on the dataset for each value of **k from 2 to 12** (i.e., k = 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 — 11 total runs). For each value of k:
   - Train a K-Means model using the provided `RunKMeans` script as the basis.
   - Record the **Spark runtime** for each clustering run.
   - Compute the **distance of every data point to its assigned cluster centroid**.
   - Identify the **top-25 outliers** across all clusters — i.e., the 25 data points with the **largest distance to their respective cluster centroid** — and report these for each value of k.

3. **Produce a visualization** of a reasonably small sample of the gearbox readings (e.g., a random sample of a few thousand rows). Use a visualization technique of your choice (e.g., scatter plot of two sensor channels, time-series plot, PCA-reduced 2D plot). You only need to **submit the source code** that generates the visualization; submitting the image files themselves is not required.

4. **Report your findings** in `Problem 3.txt`, including:
   - A table of runtimes for each value of k.
   - The top-25 outliers for each value of k (data point identifiers + distance to centroid).
   - The source code used to generate the visualization (or a reference to the file in your submission).
   - Any observations about how the number of clusters k affects outlier detection quality.

---

## Summary of Deliverables

| File | Contents |
|------|----------|
| `Problem 1.txt` | Run instructions, runtimes for all 18 hyperparameter configurations, best hyperparameter justification, search engine query results |
| `Problem 2.txt` | Run instructions, DataFrame schema, NLP tokenization sample, SVD top-25 terms/docs, genre label output, keyword query results, runtimes |
| `Problem 3.txt` | Run instructions, runtime table for k=2..12, top-25 outliers per k, visualization source code reference, observations |
| Source code | All `.scala` / `.py` files, `RunLSA.jar` (if Scala), and visualization scripts, packaged as `<group_id>.zip` |
| AI usage notes | Included within each `Problem X.txt`: AI platform used and key prompting steps |

