# Move data out of /home
mkdir -p /scratch/users/eleiva/assignment3/data
cp -r /home/users/eleiva/uni/2_semester/big_data_analytics/assignment3/data/Wikipedia-En-41784-Articles /scratch/users/eleiva/assignment3/data/
cp /home/users/eleiva/uni/2_semester/big_data_analytics/assignment3/stopwords.txt /scratch/users/eleiva/assignment3/

# Pre-download NLTK data to the shared drive so compute nodes can access it
module load lang/Python/3.11.5-GCCcore-13.2.0
python -c "import nltk; nltk.download('punkt', download_dir='/scratch/users/eleiva/assignment3/nltk_data'); nltk.download('wordnet', download_dir='/scratch/users/eleiva/assignment3/nltk_data')"


 




You are an expert coding assistant and big data analytics following stricly the **assignment instructions** to complete it **based on the examples and theory**. 

Your help me to complete he Problem 1  of this assignment to get the full grade.


## Instructions:
1) Read Problem 1: Identify TODO's tasks, assignment goal, anotate specific instructions. You must follow stricly the assingment instructions.  
2) Analyze the theory and example: Use RunSLA.py as a guidance and chepter 4 34p-39p for the theory and practice approach.
3) Identify the tasks to resolve based in the  
4) Elaborate a detailed plan: describe what we are going to do, plan how are we are going to complete correctly,  Make the plan interactive for me to understand what we are doing, explaining the code and results. 
5) Implement the RunSLA.py in python code ensuring all the steps are completed correctly and ready for run. You can use the example material as a guidance. 



## Problem description
Problem 1. Consider the Wikipedia dataset which is available from Moodle under the link Wikipedia-En41784-Articles.tar.gz of Chapter 1. Also consider the sample script RunLSA.scala or RunLSA.ipynb
notebook provided on Moodle as basis to solve this exercise.
(a) Translate the provided RunLSA.scala script into an actual Scala object and compile this object
into a jar file called RunLSA.jar, which can also be executed via spark-submit on the IRIS cluster.
Alternatively, extract the plain Python code from the RunLSA.ipynb notebook if you prefer Python
by using nbconvert --to script and submit your Python script as a Spark job on the IRIS cluster.
In either case, create two versions of your code, one that performs an NLP pipeline for parsing the
articles, and one that uses a simple tokenizer (using the regular expressions for words and numbers
introduced in Exercise Sheet #1).
Next, compute the top-25 terms and the top-25 documents, each under the top-25 latent concepts,
based on the examples provided in the notebook (and as shown in the lecture), and observe how the
results vary depending on which tokenizer or lemmatizer you use. 4 Points
(b) For the entire dataset consisting of all the 41,784 Wikipedia articles, run your Spark job with the
following configurations and compare their runtimes:
– numF req ∈ {5000, 10000, 20000} for the number of frequent terms extracted from all Wikipedia
articles, and
– k ∈ {25, 100, 250} for the different numbers of latent dimensions used for the SVD.
Also compare among different runs with and without using the NLP pipeline. Manually inspect the
results as in (a) to find the best hyper-parameters for your setting. 4 Points
Note: The setting with numF req = 20000 and k = 250 should run in less than one hour for all of
the 41,784 articles on the IRIS cluster.
(c) After having found the best hyper-parameters for your setting, implement a basic search engine for
your framework, which takes a list of keywords as input and retrieves the most relevant documents
based on the best SVD you obtained in (b) (also refer to Slides 34–39 of Chapter 4 for an explanation
of these steps). Report the results (including the titles of the Wikipedia articles) for 5–10 interesting
keyword queries. 4 Points
Summarize the results (incl. the Spark runtimes) of this experiment in your Problem 1.txt file.

## Chapter 4 34-39 resources 



With the reduced decomposition of A into U × S × VT at hand, we can now
perform the following tasks in the "latent" concept space instead of using
the actual terms and documents:

} Document-to-document similarities:
} Compute row similarities of U × S

} Term-to-term similarities:
} Compute row similarities of V× S

} Single-term-to-document similarities:
} Multiply U × S with the row vector of term i in V

} Multi-term-to-document similarities:
} Transform q.

= q'×! × V!×$

} Multiply U × S with the transformed query vector q.T
Querying
Based on the reduced representation, we can find the most similar
documents to a given query document:
 import org.apache.spark.mllib.linalg.Matrices
 def topDocsForDoc(normalizedUS: RowMatrix, docId: Long)
 : Seq[(Double, Long)] = {
 val docRowArr = row(normalizedUS, docId)
 val docRowVec = Matrices.dense(docRowArr.length, 1, docRowArr)
 val docScores = normalizedUS.multiply(docRowVec)
 val allDocWeights = docScores.rows.map(_.toArray(0)).
 zipWithUniqueId()
 allDocWeights.filter(!_._1.isNaN).top(10)
 }
 val US = multiplyByDiagonalMatrix(svd.U, svd.s)
 val normalizedUS = rowsNormalized(US)
 topDocsForDoc(normalizedUS, idDocs(doc), docIds)


ilarly, we can also find the top similar terms to a given query term:
 import breeze.linalg.{SparseVector => BSparseVector}
 import breeze.linalg.{DenseVector => BDenseVector}
 import breeze.linalg.{DenseMatrix => BDenseMatrix}
 def topTermsForTerm(
 normalizedVS: BDenseMatrix[Double],
 termId: Int): Seq[(Double, Int)] = {
 val rowVec = new BDenseVector[Double](
 row(normalizedVS, termId).toArray)
 val termScores = (normalizedVS * rowVec).toArray.zipWithIndex
 termScores.sortBy(-_._1).take(10)
 }
 val VS = multiplyByDiagonalMatrix(svd.V, svd.s)
 val normalizedVS = rowsNormalized(VS)
 topTermsForTerm(normalizedVS, idTerms(term), termIds) 

 As in an actual Information Retrieval setting, we can find the most
similar documents for a given query term:
 def topDocsForTerm(US: RowMatrix, V: Matrix, termId: Int)
 : Seq[(Double, Long)] = {
 val termRowArr = row(V, termId).toArray
 val termRowVec = Matrices.dense(termRowArr.length, 1, termRowArr)
 val docScores = US.multiply(termRowVec)
 val allDocWeights = docScores.rows.map(_.toArray(0)).
 zipWithUniqueId()
 allDocWeights.top(10)
 }
 topDocsForTerm(normalizedUS, svd.V, idTerms(term))

First, we transform a query vector into a compatible SparseVector
representation using IDF values as term weights:
 def termsToQueryVector(
 terms: Seq[String],
 idTerms: Map[String, Int],
 idfs: Map[String, Double]): BSparseVector[Double] = {
 val indices = terms.map(idTerms(_)).toArray
 val values = terms.map(idfs(_)).toArray
 new BSparseVector[Double](indices, values, idTerms.size)
 }

 Finally, we multiply 𝑼× 𝑺 with the transformed query vector 𝒒*𝑇:
 def topDocsForTermQuery(
 US: RowMatrix,
 V: Matrix,
 query: BSparseVector[Double]): Seq[(Double, Long)] = {
 val breezeV = new BDenseMatrix[Double](V.numRows, V.numCols, V.toArray)
 val termRowArr = (breezeV.t * query).toArray
 val termRowVec = Matrices.dense(termRowArr.length, 1, termRowArr)
 val docScores = US.multiply(termRowVec)
 val allDocWeights = docScores.rows.map(_.toArray(0)).
 zipWithUniqueId()
 allDocWeights.top(10) }
 val queryVec = termsToQueryVector(terms, idTerms, idfs)
 topDocsForTermQuery(US, svd.V, queryVec)

## Example python code
In this example you are going to check how we initialize the spark session, how we run into the hpc. The RDD transformations and detailed functionality is just context from the previous assignment and is not related to this Problem 1.
from pathlib import Path
import time

from pyspark.sql import SparkSession
from pyspark.sql import types as T
from pyspark.sql.functions import col

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    GBTClassifier,
    OneVsRest,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit


def print_duration(name, start, end):
    print(f"({name}) Duration: {end-start:.3f} seconds")


t0 = time.perf_counter()
spark = (
    SparkSession.builder.appName("HeartDiseaseClassification")
    .master("local[*]")
    .config("spark.driver.memory", "16g")
    .config("spark.memory.offHeap.enabled", "true")
    .config("spark.memory.offHeap.size", "4g")
    .config("spark.driver.maxResultSize", "2g")
    .getOrCreate()
)
t1 = time.perf_counter()
print_duration("SparkSession creation", t0, t1)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "problem_1" / "heart_2020_cleaned.csv"
MODEL_PATH = REPO_ROOT / "solutions" / "problem_1" / "best_decision_tree_model"
MODEL_PATH_RF = REPO_ROOT / "solutions" / "problem_1" / "best_random_forest_model"
MODEL_PATH_TVS = REPO_ROOT / "solutions" / "problem_1" / "best_decision_tree_model_tvs"

# ----------------------------------------------------------------
# Step A.1: Load and preprocess the data

schema = T.StructType(
    [
        T.StructField("HeartDisease", T.StringType(), True),
        T.StructField("BMI", T.DoubleType(), True),
        T.StructField("Smoking", T.StringType(), True),
        T.StructField("AlcoholDrinking", T.StringType(), True),
        T.StructField("Stroke", T.StringType(), True),
        T.StructField("PhysicalHealth", T.DoubleType(), True),
        T.StructField("MentalHealth", T.DoubleType(), True),
        T.StructField("DiffWalking", T.StringType(), True),
        T.StructField("Sex", T.StringType(), True),
        T.StructField("AgeCategory", T.StringType(), True),
        T.StructField("Race", T.StringType(), True),
        T.StructField("Diabetic", T.StringType(), True),
        T.StructField("PhysicalActivity", T.StringType(), True),
        T.StructField("GenHealth", T.StringType(), True),
        T.StructField("SleepTime", T.DoubleType(), True),
        T.StructField("Asthma", T.StringType(), True),
        T.StructField("KidneyDisease", T.StringType(), True),
        T.StructField("SkinCancer", T.StringType(), True),
    ]
)

t0 = time.perf_counter()
data = spark.read.option("header", True).csv(str(DATA_PATH), schema=schema)
t1 = time.perf_counter()
print_duration("Load CSV", t0, t1)

t0 = time.perf_counter()
data = data.filter(col("HeartDisease").isNotNull())
data = data.filter((col("HeartDisease") == "Yes") | (col("HeartDisease") == "No"))
t1 = time.perf_counter()
print_duration("Preprocessing filters", t0, t1)

# ----------------------------------------------------------------
# Part (A): Load and preprocess data
# ----------------------------------------------------------------

print("\n" + "=" * 60)
print("PART (A): Load and preprocess data")
print("=" * 60)

# ----------------------------------------------------------------
# Part (B): DecisionTree with CrossValidator
# ----------------------------------------------------------------

print("\n" + "=" * 60)
print("PART (B): DecisionTree with CrossValidator")
print("=" * 60)

# ----------------------------------------------------------------
# Step B.1: 80%/20% split

t0 = time.perf_counter()
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
t1 = time.perf_counter()
print_duration("Train/Test randomSplit (lazy)", t0, t1)

t0 = time.perf_counter()
train_count = train_data.count()
test_count = test_data.count()
t1 = time.perf_counter()
print_duration("Counting train/test records", t0, t1)

print(f"\n(B.1) Training records: {train_count}")
print(f"(B.1) Testing records: {test_count}")

categorical_columns = [
    "Smoking",
    "AlcoholDrinking",
    "Stroke",
    "DiffWalking",
    "Sex",
    "AgeCategory",
    "Race",
    "Diabetic",
    "PhysicalActivity",
    "GenHealth",
    "Asthma",
    "KidneyDisease",
    "SkinCancer",
]

indexed_columns = [col + "_indexed" for col in categorical_columns]

string_indexers = [
    StringIndexer(inputCol=col, outputCol=col + "_indexed", handleInvalid="skip")
    for col in categorical_columns
]

label_indexer = StringIndexer(
    inputCol="HeartDisease", outputCol="label", handleInvalid="skip"
)

numeric_columns = ["BMI", "PhysicalHealth", "MentalHealth", "SleepTime"]

feature_columns = numeric_columns + indexed_columns

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

dt = DecisionTreeClassifier(labelCol="label", featuresCol="features")

pipeline = Pipeline(stages=string_indexers + [label_indexer, assembler, dt])

# ----------------------------------------------------------------
# Step B.2: Hyperparameter tuning with 5-fold cross-validation

paramGrid = (
    ParamGridBuilder()
    .addGrid(dt.impurity, ["entropy", "gini"])
    .addGrid(dt.maxDepth, [6, 12, 24])
    .addGrid(dt.maxBins, [20, 50, 100])
    .build()
)

# CrossValidator requires a pyspark.ml Evaluator. We use
# BinaryClassificationEvaluator(metricName="areaUnderROC"), which is equivalent
# to the areaUnderROC metric from pyspark.mllib.evaluation.BinaryClassificationMetrics.
cv_evaluator = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
)

cv = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=cv_evaluator,
    numFolds=5,
    seed=42,
)

print("\n(B.2) Starting 5-fold cross-validation with hyperparameter search...")
print(
    "(B.2) Parameters: impurity=[entropy, gini], maxDepth=[6, 12, 24], maxBins=[20, 50, 100]"
)
print("(B.2) This may take a while...")

t0 = time.perf_counter()
cv_model = cv.fit(train_data)
t1 = time.perf_counter()
print_duration("CrossValidator fit (5-fold)", t0, t1)

best_model = cv_model.bestModel
best_dt_model = best_model.stages[-1]

print(f"\n(B.2) Best hyperparameters:")
print(f"  impurity: {best_dt_model.getImpurity()}")
print(f"  maxDepth: {best_dt_model.getMaxDepth()}")
print(f"  maxBins: {best_dt_model.getMaxBins()}")

# ----------------------------------------------------------------
# Step B.3: Measure metrics on training data with best model


train_auc_evaluator = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
)
t0 = time.perf_counter()
train_predictions = best_model.transform(train_data)
train_auc = train_auc_evaluator.evaluate(train_predictions)
t1 = time.perf_counter()
print_duration("Training prediction + eval (AUC)", t0, t1)

print(f"\n(B.3) Training metrics (best model):")
print(f"  AUC: {train_auc}")

# Save the best DecisionTree model
t0 = time.perf_counter()
best_dt_model.write().overwrite().save(str(MODEL_PATH))
t1 = time.perf_counter()
print_duration("Save best DecisionTree model", t0, t1)
print(
    "\n(B.3) Best DecisionTree model saved to: solutions/problem_1/best_decision_tree_model"
)

# ----------------------------------------------------------------
# Step B.4: Evaluate best model on 20% test split

t0 = time.perf_counter()
test_predictions = best_model.transform(test_data)
test_auc = train_auc_evaluator.evaluate(test_predictions)
t1 = time.perf_counter()
print_duration("Test prediction + eval (AUC/PR)", t0, t1)

print(f"\n(B.4) Test set metrics (best model, 20% split):")
print(f"  AUC: {test_auc}")

# Compute per-label precision, recall, accuracy using MulticlassMetrics
t0 = time.perf_counter()
prediction_label_rdd_b = test_predictions.select("prediction", "label").rdd.map(
    lambda row: (float(row.prediction), float(row.label))
)
metrics_b = MulticlassMetrics(prediction_label_rdd_b)
accuracy = metrics_b.accuracy
precision_1 = metrics_b.precision(1.0)
recall_1 = metrics_b.recall(1.0)
precision_0 = metrics_b.precision(0.0)
recall_0 = metrics_b.recall(0.0)
t1 = time.perf_counter()
print_duration("Test MulticlassMetrics computation", t0, t1)

print(f"\n(B.4) Per-label metrics on test set (MulticlassMetrics):")
print(f"  Label 1 (HeartDisease=Yes):")
print(f"    Precision: {precision_1:.4f}")
print(f"    Recall: {recall_1:.4f}")
print(f"  Label 0 (HeartDisease=No):")
print(f"    Precision: {precision_0:.4f}")
print(f"    Recall: {recall_0:.4f}")
print(f"  Overall Accuracy: {accuracy:.4f}")

t0 = time.perf_counter()
bc_metrics_b = BinaryClassificationMetrics(prediction_label_rdd_b)
t1 = time.perf_counter()
print_duration("Test BinaryClassificationMetrics computation", t0, t1)
print(f"  AUC (BinaryClassificationMetrics): {bc_metrics_b.areaUnderROC:.4f}")
print(f"  Area under PR (BinaryClassificationMetrics): {bc_metrics_b.areaUnderPR:.4f}")

# ----------------------------------------------------------------
# Part (C): TrainValidationSplit instead of CrossValidator
# ----------------------------------------------------------------

print("\n" + "=" * 60)
print("PART (C): TrainValidationSplit vs CrossValidator Comparison")
print("=" * 60)

# ----------------------------------------------------------------
# Step C.1: Reuse 80/20 split (train_data, test_data) from Step B.1
print("\n(C.1) Reusing 80/20 split from Part B Step (B.1)")

# ----------------------------------------------------------------
# Step C.2: Replace CrossValidator with TrainValidationSplit (trainRatio=0.8)
# TrainValidationSplit requires a pyspark.ml Evaluator. We use
# BinaryClassificationEvaluator(metricName="areaUnderROC"), which is equivalent
# to the areaUnderROC metric from pyspark.mllib.evaluation.BinaryClassificationMetrics.
tvs = TrainValidationSplit(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=cv_evaluator,
    trainRatio=0.8,
    seed=42,
)

print("\n(C.2) Starting TrainValidationSplit with hyperparameter search...")
print(
    "(C.2) Parameters: impurity=[entropy, gini], maxDepth=[6,12,24], maxBins=[20,50,100]"
)
print("(C.2) trainRatio=0.8 (80% of training data for fitting, 20% for validation)")

t0 = time.perf_counter()
tvs_model = tvs.fit(train_data)
t1 = time.perf_counter()
print_duration("TrainValidationSplit fit", t0, t1)

best_model_c = tvs_model.bestModel
best_dt_model_c = best_model_c.stages[-1]

print(f"\n(C.2) Best hyperparameters (TrainValidationSplit):")
print(f"  impurity: {best_dt_model_c.getImpurity()}")
print(f"  maxDepth: {best_dt_model_c.getMaxDepth()}")
print(f"  maxBins: {best_dt_model_c.getMaxBins()}")

# ----------------------------------------------------------------
# Step C.3: Evaluate on training data, save model
t0 = time.perf_counter()
train_predictions_c = best_model_c.transform(train_data)
train_auc_c = train_auc_evaluator.evaluate(train_predictions_c)
t1 = time.perf_counter()
print_duration("TVS training prediction + eval", t0, t1)

print(f"\n(C.3) Training metrics (TrainValidationSplit best model):")
print(f"  AUC: {train_auc_c:.4f}")

# Save best TVS model
t0 = time.perf_counter()
best_dt_model_c.write().overwrite().save(str(MODEL_PATH_TVS))
t1 = time.perf_counter()
print_duration("Save TVS best DecisionTree model", t0, t1)
print(
    "(C.3) Best DecisionTree model (TrainValidationSplit) saved to: best_decision_tree_model_tvs"
)

# ----------------------------------------------------------------
# Step C.4: Evaluate on 20% test split
t0 = time.perf_counter()
test_predictions_c = best_model_c.transform(test_data)
test_auc_c = train_auc_evaluator.evaluate(test_predictions_c)
t1 = time.perf_counter()
print_duration("TVS test prediction + eval", t0, t1)

print(f"\n(C.4) Test set metrics (TrainValidationSplit, 20% split):")
print(f"  AUC: {test_auc_c:.4f}")

# Per-label precision, recall, accuracy using MulticlassMetrics
t0 = time.perf_counter()
prediction_label_rdd_c = test_predictions_c.select("prediction", "label").rdd.map(
    lambda row: (float(row.prediction), float(row.label))
)
metrics_c = MulticlassMetrics(prediction_label_rdd_c)
accuracy_c = metrics_c.accuracy
precision_1_c = metrics_c.precision(1.0)
recall_1_c = metrics_c.recall(1.0)
precision_0_c = metrics_c.precision(0.0)
recall_0_c = metrics_c.recall(0.0)
t1 = time.perf_counter()
print_duration("TVS MulticlassMetrics computation", t0, t1)

print(
    f"\n(C.4) Per-label metrics on test set (TrainValidationSplit, MulticlassMetrics):"
)
print(f"  Label 1 (HeartDisease=Yes):")
print(f"    Precision: {precision_1_c:.4f}")
print(f"    Recall: {recall_1_c:.4f}")
print(f"  Label 0 (HeartDisease=No):")
print(f"    Precision: {precision_0_c:.4f}")
print(f"    Recall: {recall_0_c:.4f}")
print(f"  Overall Accuracy: {accuracy_c:.4f}")

print("\n(C.5) Comparison: CrossValidator (Part B) vs TrainValidationSplit (Part C)")
print(f"  Metric           | CrossValidator | TrainValidationSplit")
print(f"  -----------------|----------------|---------------------")
print(f"  Accuracy         | {accuracy:.4f}         | {accuracy_c:.4f}")
print(f"  Precision (Yes)  | {precision_1:.4f}         | {precision_1_c:.4f}")
print(f"  Recall    (Yes)  | {recall_1:.4f}         | {recall_1_c:.4f}")
print(f"  Precision (No)   | {precision_0:.4f}         | {precision_0_c:.4f}")
print(f"  Recall    (No)   | {recall_0:.4f}         | {recall_0_c:.4f}")
print(f"  Best impurity    | {best_dt_model.getImpurity()}         | {best_dt_model_c.getImpurity()}")
print(f"  Best maxDepth    | {best_dt_model.getMaxDepth()}         | {best_dt_model_c.getMaxDepth()}")
print(f"  Best maxBins     | {best_dt_model.getMaxBins()}         | {best_dt_model_c.getMaxBins()}")
print("  Note: CrossValidator uses 5-fold CV (more reliable, slower).")
print("  TrainValidationSplit uses a single 80/20 split (faster, less stable).")

t0 = time.perf_counter()
bc_metrics_c = BinaryClassificationMetrics(prediction_label_rdd_c)
t1 = time.perf_counter()
print_duration("TVS BinaryClassificationMetrics computation", t0, t1)
print(f"  AUC (BinaryClassificationMetrics): {bc_metrics_c.areaUnderROC:.4f}")
print(f"  Area under PR (BinaryClassificationMetrics): {bc_metrics_c.areaUnderPR:.4f}")

# ----------------------------------------------------------------
# Part (D): Target=AgeCategory, RandomForest, MulticlassMetrics
# ----------------------------------------------------------------

print("\n" + "=" * 60)
print("PART (D): Target=AgeCategory, RandomForest, MulticlassMetrics")
print("=" * 60)

# ----------------------------------------------------------------
# Step D.1: Create 80/20 split for AgeCategory target
print("\n(D.1) Creating 80/20 split for AgeCategory target...")
data_d = data.filter(col("AgeCategory").isNotNull())
t0 = time.perf_counter()
train_data_d, test_data_d = data_d.randomSplit([0.8, 0.2], seed=42)
t1 = time.perf_counter()
print_duration("AgeCategory Train/Test randomSplit (lazy)", t0, t1)

t0 = time.perf_counter()
train_count_d = train_data_d.count()
test_count_d = test_data_d.count()
t1 = time.perf_counter()
print_duration("Counting AgeCategory train/test records", t0, t1)
print(f"(D.1) Training records: {train_count_d}")
print(f"(D.1) Testing records: {test_count_d}")

# Features: exclude AgeCategory (target), keep all other original features
categorical_columns_d = [
    "HeartDisease",
    "Smoking",
    "AlcoholDrinking",
    "Stroke",
    "DiffWalking",
    "Sex",
    "Race",
    "Diabetic",
    "PhysicalActivity",
    "GenHealth",
    "Asthma",
    "KidneyDisease",
    "SkinCancer",
]

numeric_columns_d = ["BMI", "PhysicalHealth", "MentalHealth", "SleepTime"]
indexed_columns_d = [col + "_indexed" for col in categorical_columns_d]

# String indexers for feature columns (excluding AgeCategory)
string_indexers_d = [
    StringIndexer(inputCol=col, outputCol=col + "_indexed", handleInvalid="skip")
    for col in categorical_columns_d
]

# Label indexer for AgeCategory (multiclass target)
label_indexer_d = StringIndexer(
    inputCol="AgeCategory", outputCol="label", handleInvalid="skip"
)

# Vector assembler for features
assembler_d = VectorAssembler(
    inputCols=numeric_columns_d + indexed_columns_d, outputCol="features"
)

# ----------------------------------------------------------------
# Step D.2: Train and evaluate GradientBoostedTrees (OvR) for AgeCategory

print("\n(D.2) Training GradientBoostedTrees (OvR) for AgeCategory target...")
dt_d = GBTClassifier(labelCol="label", featuresCol="features", maxDepth=6, maxIter=10)
ovr_gbt = OneVsRest(classifier=dt_d)
pipeline_dt = Pipeline(
    stages=string_indexers_d + [label_indexer_d, assembler_d, ovr_gbt]
)
t0 = time.perf_counter()
model_dt = pipeline_dt.fit(train_data_d)
t1 = time.perf_counter()
print_duration("GradientBoostedTrees (AgeCategory) fit", t0, t1)
t0 = time.perf_counter()
pred_dt = model_dt.transform(test_data_d)
t1 = time.perf_counter()
print_duration("GradientBoostedTrees (AgeCategory) transform", t0, t1)

# Use MulticlassMetrics for evaluation

t0 = time.perf_counter()
prediction_label_rdd_dt = pred_dt.select("prediction", "label").rdd.map(
    lambda row: (float(row.prediction), float(row.label))
)
metrics_dt = MulticlassMetrics(prediction_label_rdd_dt)
acc_dt = metrics_dt.accuracy
prec_dt = metrics_dt.weightedPrecision
rec_dt = metrics_dt.weightedRecall
t1 = time.perf_counter()
print_duration("GradientBoostedTrees (AgeCategory) MulticlassMetrics", t0, t1)

print(f"\n(D.2) GradientBoostedTrees test metrics (AgeCategory, MulticlassMetrics):")
print(f"  Accuracy: {acc_dt:.4f}")
print(f"  Weighted Precision: {prec_dt:.4f}")
print(f"  Weighted Recall: {rec_dt:.4f}")

# ----------------------------------------------------------------
# Step D.3: RandomForest with hyperparameter tuning (numTrees, maxDepth, maxBins, impurity)
print(
    "\n(D.3) Training RandomForest with hyperparameter tuning for AgeCategory target..."
)

rf_d = RandomForestClassifier(
    labelCol="label",
    featuresCol="features",
)

pipeline_rf = Pipeline(stages=string_indexers_d + [label_indexer_d, assembler_d, rf_d])

paramGrid_builder_rf = (
    ParamGridBuilder()
    .addGrid(rf_d.numTrees, [5, 10, 20])
    .addGrid(rf_d.maxDepth, [6, 12, 24])
    .addGrid(rf_d.maxBins, [20, 50, 100])
)

# RandomForest in Spark supports impurity=[gini, entropy].
paramGrid_rf = paramGrid_builder_rf.addGrid(rf_d.impurity, ["gini", "entropy"]).build()

# Use MulticlassClassificationEvaluator for tuning
multiclass_cv_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy",
    # Note: equivalent to MulticlassMetrics.accuracy; ml API required by CrossValidator
)

cv_rf = CrossValidator(
    estimator=pipeline_rf,
    estimatorParamMaps=paramGrid_rf,
    evaluator=multiclass_cv_eval,
    numFolds=5,
    seed=42,
)

print(
    "(D.3) Starting CrossValidator with numTrees=[5,10,20], "
    "maxDepth=[6,12,24], maxBins=[20,50,100], impurity=[gini,entropy]..."
)
t0 = time.perf_counter()
cv_rf_model = cv_rf.fit(train_data_d)
t1 = time.perf_counter()
print_duration("RandomForest CrossValidator fit", t0, t1)

best_rf_model = cv_rf_model.bestModel
best_rf_stage = best_rf_model.stages[-1]
best_num_trees = best_rf_stage.getNumTrees
print(f"(D.3) Best numTrees: {best_num_trees}")
print(f"(D.3) Best maxDepth: {best_rf_stage.getMaxDepth()}")
print(f"(D.3) Best maxBins: {best_rf_stage.getMaxBins()}")
print(f"(D.3) Best impurity: {best_rf_stage.getImpurity()}")

t0 = time.perf_counter()
pred_rf = best_rf_model.transform(test_data_d)
t1 = time.perf_counter()
print_duration("RandomForest transform", t0, t1)

# Use MulticlassMetrics for evaluation
t0 = time.perf_counter()
prediction_label_rdd_rf = pred_rf.select("prediction", "label").rdd.map(
    lambda row: (float(row.prediction), float(row.label))
)
metrics_rf = MulticlassMetrics(prediction_label_rdd_rf)
acc_rf = metrics_rf.accuracy
prec_rf = metrics_rf.weightedPrecision
rec_rf = metrics_rf.weightedRecall
t1 = time.perf_counter()
print_duration("RandomForest MulticlassMetrics", t0, t1)

print(f"\n(D.3) RandomForest test metrics (AgeCategory, MulticlassMetrics):")
labels_rf = sorted(
    prediction_label_rdd_rf.map(lambda x: x[1]).distinct().collect()
)
for label in labels_rf:
    print(f"  Label {int(label)}:")
    print(f"    Precision: {metrics_rf.precision(label):.4f}")
    print(f"    Recall: {metrics_rf.recall(label):.4f}")
print(f"  Accuracy: {acc_rf:.4f}")
print(f"  Weighted Precision: {prec_rf:.4f}")
print(f"  Weighted Recall: {rec_rf:.4f}")

# Save best RandomForest model
t0 = time.perf_counter()
best_rf_stage.write().overwrite().save(str(MODEL_PATH_RF))
t1 = time.perf_counter()
print_duration("Save best RandomForest model", t0, t1)
print(
    "(D.3) Best RandomForest model saved to: solutions/problem_1/best_random_forest_model"
)

# ----------------------------------------------------------------
# Step D.4: Model comparison summary
print(f"\n(D.4) Model Comparison Summary (AgeCategory target, full dataset):")
print(
    f"  GradientBoostedTrees (OvR)  - Accuracy: {acc_dt:.4f}, Precision: {prec_dt:.4f}, Recall: {rec_dt:.4f}"
)
print(
    f"  RandomForest               - Accuracy: {acc_rf:.4f}, Precision: {prec_rf:.4f}, Recall: {rec_rf:.4f} (best numTrees={best_num_trees})"
)
print("\n(D.5) Comparison: GBT (OvR) vs RandomForest")
print(f"  Metric           | GBT (OvR)       | RandomForest")
print(f"  -----------------|-----------------|---------------------")
print(f"  Accuracy         | {acc_dt:.4f}         | {acc_rf:.4f}")
print(f"  Weighted Precision | {prec_dt:.4f}         | {prec_rf:.4f}")
print(f"  Weighted Recall    | {rec_dt:.4f}         | {rec_rf:.4f}")

spark.stop()

**Running on the hpc**
```bash
cd solutions/problem_1/
source env/bin/activate
export SPARK_HOME=/home/$USER/Apps/spark-4.0.2-bin-hadoop3
export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-src.zip:$PYTHONPATH
python main.py
```




Procceed with the instructions according to the assingment instructions 




 