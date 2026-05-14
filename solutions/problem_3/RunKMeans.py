# Generated from: RunKMeans.ipynb
# Converted at: 2026-05-13T09:10:08.027Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

import warnings
warnings.filterwarnings('ignore')

from pyspark.mllib.clustering import KMeans
from pyspark.mllib.linalg import Vectors
import math, numpy as np

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("RunKMeans").getOrCreate()
sc = spark.sparkContext

rawData = sc.textFile("../Data/kddcup.corrected.csv").sample(False, 0.01) # again 1 percent of data, randomly sampled

# Split the lines by commas and get the last elements as labels
labelsAndCounts = rawData.map(lambda x: x.split(',')[-1]).countByValue().items()
for label, count in sorted(labelsAndCounts, key=lambda x: x[1], reverse=True):
    print(f"{label}: {count}")

# Extract both labels and vectors
labelsAndData = rawData.map(lambda line: line.split(',')).map(lambda arr: (arr[-1], Vectors.dense(arr[0:1] + arr[4:-1])))
data = labelsAndData.values().cache()

# Take the first 10 elements as a sample
for sample in labelsAndData.take(10):
    print(sample)

kmeans = KMeans()
model = kmeans.train(data, k=2)         # k = number of clusters
clusterCenters = model.clusterCenters   # cluster centers
for c in clusterCenters:
    print(c)

# Print the distribution of predicted labels per cluster
clusterLabelCount = labelsAndData.map(lambda x: (model.predict(x[1]), x[0])).countByValue()
for ((cluster, label), count) in sorted(clusterLabelCount.items()):
    print(f"{cluster}\t{label:18}{count:8}")

# Use this to Dump the Non-Normalized Data & Clusters

kmeans = KMeans()
model = kmeans.train(data, k=100)

# Map each vector to (cluster, vector) and concatenate clusters and vectors
sample = data.map(lambda vector: (model.predict(vector), vector.toArray())).map(lambda x: (x[0], ",".join(map(str, x[1]))))
#sample.sample(False, 0.01).saveAsTextFile("./kmeans-sample")

# Investigate the Average Distance to Closest Centroid 

def distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2)) # Euclidian distance

def distToCentroid(vector, model):
    cluster = model.predict(vector)
    centroid = model.clusterCenters[cluster]
    return distance(centroid, vector)

def clusteringScore(data, k):
    kmeans = KMeans.train(data, k)
    return data.map(lambda vector: distToCentroid(vector, kmeans)).mean()

results = []
for k in range(30, 61, 10):
    score = clusteringScore(data, k)
    results.append((k, score))

for r in results:
    print(r)

# Normalize the Data and Compare Clustering Scores 

def normalize(data):
    data_as_array = data.map(lambda x: x.toArray())
    num_cols = len(data_as_array.first())
    n = data_as_array.count()
    sums = data_as_array.reduce(lambda a, b: [x + y for x, y in zip(a, b)])
    sum_squares = data_as_array.aggregate([0.0] * num_cols,
                                          lambda a, b: [x + y * y for x, y in zip(a, b)],
                                          lambda a, b: [x + y for x, y in zip(a, b)])
    stdevs = [math.sqrt(n * sum_sq - sum_ * sum_) / n for sum_sq, sum_ in zip(sum_squares, sums)]
    means = [sum_ / n for sum_ in sums]

    def normalize_vector(vector):
        normalized_array = [(value - mean) / stdev if stdev > 0 else value - mean
                            for value, mean, stdev in zip(vector.toArray(), means, stdevs)]
        return Vectors.dense(normalized_array)

    return normalize_vector

normalized_data = data.map(normalize(data)).cache()

clustering_scores = [(k, clusteringScore(normalized_data, k)) for k in range(60, 121, 10)]
for score in clustering_scores:
    print(score)

normalized_data.unpersist()

# Switch to One-Hot Encoding of Categorical Attributes

def onehot(raw_data):
    splitData = raw_data.map(lambda x: x.split(','))
    protocols = splitData.map(lambda x: x[1]).distinct().zipWithIndex().collectAsMap()
    print("PROTOCOLS", protocols)
    services = splitData.map(lambda x: x[2]).distinct().zipWithIndex().collectAsMap()
    print("SERVICES", services)
    tcpStates = splitData.map(lambda x: x[3]).distinct().zipWithIndex().collectAsMap()
    print("TCPSTATES", tcpStates)

    def encode_line(line):
        buffer = line.split(',')
        protocol = buffer.pop(1)
        service = buffer.pop(1)
        tcpState = buffer.pop(1)
        label = buffer.pop(-1)
        vector = [float(x) for x in buffer]

        newProtocolFeatures = [0.0] * len(protocols)
        newProtocolFeatures[protocols[protocol]] = 1.0
        newServiceFeatures = [0.0] * len(services)
        newServiceFeatures[services[service]] = 1.0
        newTcpStateFeatures = [0.0] * len(tcpStates)
        newTcpStateFeatures[tcpStates[tcpState]] = 1.0

        vector[1:1] = newTcpStateFeatures
        vector[1:1] = newServiceFeatures
        vector[1:1] = newProtocolFeatures

        return label, Vectors.dense(vector)

    return encode_line

# Dump the Normalized and One-Hot Encoded Data

parseFunction = onehot(rawData)
labelsAndData = rawData.map(parseFunction)
normalizedLabelsAndData = labelsAndData.mapValues(normalize(labelsAndData.values())).cache()

kmeans = KMeans()
model = kmeans.train(normalizedLabelsAndData.values(), k=100)

sample = normalizedLabelsAndData.values().map(lambda vector: str(model.predict(vector)) + "," + ",".join(map(str, vector.toArray()))).sample(False, 0.01)
#sample.saveAsTextFile("./kmeans-sample-normalized")

normalizedLabelsAndData.unpersist()

# Switch to Entropy-Based Clustering Scores

def entropy(counts):
    values = [c for c in counts if c > 0]
    n = sum(values)
    return sum(-p * math.log(p) for v in values for p in [v / n])

def clusteringScoreByEntropy(normalizedData, k):
    normalizedData.cache()

    kmeans = KMeans()
    model = kmeans.train(normalizedData.values(), k)

    normalizedData.unpersist()

    labelsAndClusters = normalizedData.mapValues(lambda x: model.predict(x))
    clustersAndLabels = labelsAndClusters.map(lambda x: (x[1], x[0]))
    labelsInCluster = clustersAndLabels.groupByKey().values()
    labelCounts = labelsInCluster.map(lambda x: list(map(len, x))).map(lambda x: [sum(x)] + x)
    n = normalizedData.count()

    return labelCounts.map(lambda m: sum(m) * entropy(m)).sum() / n

parseFunction = onehot(rawData)
labelsAndData = rawData.map(parseFunction)

normalizeFunction = normalize(labelsAndData.values())
normalizedLabelsAndData = labelsAndData.mapValues(normalizeFunction).cache()

results = [(k, clusteringScoreByEntropy(normalizedLabelsAndData, k)) for k in range(120, 161, 10)]
for r in results:
    print(r)

# Show the Distribution of Labels for the Best Value of k = 150

model = KMeans.train(normalizedLabelsAndData.values(), k=150)

clusterLabelCount = labelsAndData.map(lambda x: (model.predict(normalizeFunction(x[1])), x[0])).countByValue()
for ((cluster, label), count) in sorted(clusterLabelCount.items()):
    print(f"{cluster:<1}\t{label:18}{count:8}")

normalizedLabelsAndData.unpersist()

# And Run the Final Anomaly Detection Strategy

def anomaly(data, normalizeFunction):
    normalizedData = data.map(normalizeFunction)
    normalizedData.cache()

    kmeans = KMeans()
    model = kmeans.train(normalizedData, k=150)

    normalizedData.unpersist()

    distances = normalizedData.map(lambda vector: distToCentroid(vector, model))
    threshold = distances.top(100)[-1]

    return lambda vector: distToCentroid(normalizeFunction(vector), model) > threshold

parseFunction = onehot(rawData)
originalAndData = rawData.map(lambda line: (line, parseFunction(line)[1]))
data = originalAndData.values()
normalizeFunction = normalize(data)
anomalyFunction = anomaly(data, normalizeFunction)
anomalies = originalAndData.filter(lambda x: anomalyFunction(x[1])).keys()
anomalies.take(10)

fractionOfAnomalies = anomalies.count() / rawData.count() * 100
print("Fraction of anomalies: {:.2f}%".format(fractionOfAnomalies))

normalizedLabelsAndData = labelsAndData.mapValues(normalize(labelsAndData.values())).cache()
normalizedLabelsAndData.cache()

kmeans = KMeans()
model = kmeans.train(normalizedLabelsAndData.map(lambda x: x[1]), k=15)

D = []
L = []
for d in normalizedLabelsAndData.map(lambda x: (x[0], np.array(x[1]), model.predict(x[1]))).collect():
    D.append(d[1])
    L.append(d[0])

normalizedLabelsAndData.unpersist()

D[:10]

L[:10]

import umap, umap.plot
mapper = umap.UMAP().fit(D)
umap.plot.points(mapper, labels=np.array(L))

from umap import UMAP

umap_3d = UMAP(n_components=3, init='random', random_state=0)
proj_3d = umap_3d.fit_transform(D)

import plotly.io as pio
pio.renderers.default = 'iframe'

import plotly.express as px

fig_3d = px.scatter_3d(
    proj_3d, x=0, y=1, z=2,
    color=L, width=1200, height=800
)
fig_3d.update_traces(marker_size=1)
fig_3d.show()