#!/usr/bin/env python3

import argparse
import math
import time
import re
import numpy as np

from scipy.sparse import csr_matrix

import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

from pyspark.sql import SparkSession
from pyspark.mllib.linalg import Vectors, Matrices
from pyspark.mllib.linalg.distributed import RowMatrix


# ============================================================
# Utility Functions
# ============================================================

def print_duration(name, start, end):
    print(f"({name}) Duration: {end - start:.3f} seconds")


# ============================================================
# Parse Wikipedia Documents
# ============================================================

def parse_header(line):
    try:
        s = line[line.index('id="') + 4:]
        article_id = s[:s.index('"')]

        s = s[s.index('url="') + 5:]
        url = s[:s.index('"')]

        s = s[s.index('title="') + 7:]
        title = s[:s.index('"')]

        return article_id, url, title

    except Exception:
        return "", "", ""


def parse_document(lines):

    docs = []

    title = ""
    content = ""

    for line in lines:

        try:

            if line.startswith("<doc "):

                title = parse_header(line)[2]
                content = ""

            elif line.startswith("</doc>"):

                if title and content:
                    docs.append((title, content))

            else:
                content += line + "\n"

        except Exception:
            content = ""

    return docs


def parse_flat_map(uri_text):

    uri, text = uri_text

    return parse_document(text.split("\n"))


# ============================================================
# NLP Pipelines
# ============================================================

lemmatizer = WordNetLemmatizer()


def plain_text_to_lemmas(title_text):

    title, text = title_text

    lemmas = []

    sentences = sent_tokenize(text)

    for sentence in sentences:

        tokens = word_tokenize(sentence)

        for token in tokens:

            lemma = lemmatizer.lemmatize(token.lower())

            if (
                len(lemma) > 2
                and lemma not in bStopWords.value
                and lemma.isalpha()
            ):
                lemmas.append(lemma)

    return (title, lemmas)


def plain_text_to_regex(title_text):

    title, text = title_text

    tokens = re.findall(r"[A-Za-z]+|\d+", text.lower())

    tokens = [
        token
        for token in tokens
        if len(token) > 2
        and token not in bStopWords.value
    ]

    return (title, tokens)


# ============================================================
# TF Computation
# ============================================================

def calculate_term_freqs(title_terms):

    title, terms = title_terms

    term_freqs = {}

    for term in terms:
        term_freqs[term] = term_freqs.get(term, 0) + 1

    return (title, term_freqs)


# ============================================================
# Query Functions
# ============================================================

def terms_to_query_vector(terms, id_terms, idfs):

    valid_terms = [
        term for term in terms
        if term in id_terms and term in idfs
    ]

    indices = [id_terms[term] for term in valid_terms]

    values = [idfs[term] for term in valid_terms]

    return csr_matrix(
        (values, (indices, [0] * len(indices))),
        shape=(len(id_terms), 1)
    )


def multiply_by_diagonal_row_matrix(mat, diag):

    s_arr = diag.toArray()

    return RowMatrix(
        mat.rows.map(
            lambda vec: Vectors.dense(
                np.multiply(vec.toArray(), s_arr)
            )
        )
    )


def top_docs_for_term_query(US, V, query):

    term_row_arr = np.dot(
        V.toArray().T,
        query.toarray()
    ).flatten()

    term_row_vec = Matrices.dense(
        len(term_row_arr),
        1,
        term_row_arr
    )

    doc_scores = US.multiply(term_row_vec)

    all_doc_weights = (
        doc_scores.rows
        .zipWithUniqueId()
        .map(lambda x: (x[0].toArray()[0], x[1]))
    )

    return sorted(
        all_doc_weights.collect(),
        key=lambda x: -x[0]
    )[:10]


# ============================================================
# Concept Inspection
# ============================================================

def top_terms_in_top_concepts(
    svd,
    term_ids,
    num_concepts=25,
    num_terms=25
):

    print("\n" + "=" * 60)
    print("TOP TERMS PER CONCEPT")
    print("=" * 60)

    v = svd.V

    arr = v.toArray().T

    for i in range(min(num_concepts, v.numCols)):

        print(f"\nConcept {i}")

        term_weights = [
            (arr[i][term_id], term_id)
            for term_id in range(v.numRows)
        ]

        sorted_terms = sorted(
            term_weights,
            key=lambda x: -x[0]
        )

        for score, idx in sorted_terms[:num_terms]:

            term = term_ids.get(idx, str(idx))

            print(f"{term:25s} {score:.6f}")


def top_docs_in_top_concepts(
    svd,
    doc_ids,
    num_concepts=25,
    num_docs=25
):

    print("\n" + "=" * 60)
    print("TOP DOCUMENTS PER CONCEPT")
    print("=" * 60)

    u = svd.U

    for i in range(min(num_concepts, svd.s.size)):

        print(f"\nConcept {i}")

        doc_weights = [
            (score, doc_id)
            for score, doc_id in (
                u.rows
                .map(lambda row: row.toArray()[i])
                .zipWithUniqueId()
                .collect()
            )
        ]

        sorted_docs = sorted(
            doc_weights,
            key=lambda x: -x[0]
        )

        for score, idx in sorted_docs[:num_docs]:

            doc = doc_ids.get(idx, str(idx))

            print(f"{doc[:80]:80s} {score:.6f}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Wikipedia LSA with Spark"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Wikipedia extracted directory"
    )

    parser.add_argument(
        "--stopwords",
        required=True,
        help="Path to stopwords.txt"
    )

    parser.add_argument(
        "--nltk_data",
        required=True,
        help="Path to nltk_data directory"
    )

    parser.add_argument(
        "--tokenizer",
        choices=["regex", "nlp"],
        default="regex"
    )

    parser.add_argument(
        "--numFreq",
        type=int,
        default=5000
    )

    parser.add_argument(
        "--k",
        type=int,
        default=25
    )

    parser.add_argument(
        "--sampleSize",
        type=float,
        default=1.0
    )

    parser.add_argument(
        "--query",
        type=str,
        default=None
    )

    args = parser.parse_args()

    # ========================================================
    # NLTK Resources
    # ========================================================

    nltk.data.path.append(args.nltk_data)

    # ========================================================
    # Spark Session
    # ========================================================

    t0 = time.perf_counter()

    spark = (
        SparkSession.builder
        .appName("RunLSA")
        .config("spark.driver.memory", "16g")
        .config("spark.driver.maxResultSize", "4g")
        .getOrCreate()
    )

    sc = spark.sparkContext

    t1 = time.perf_counter()

    print_duration("SparkSession creation", t0, t1)

    # ========================================================
    # Stopwords
    # ========================================================

    t0 = time.perf_counter()

    stopwords = set(
        sc.textFile(args.stopwords).collect()
    )

    bStopWords = sc.broadcast(stopwords)

    t1 = time.perf_counter()

    print_duration("Stopwords loading", t0, t1)

    print(f"Number of stopwords: {len(stopwords)}")

    # ========================================================
    # Load Wikipedia Files
    # ========================================================

    t0 = time.perf_counter()

    text_files = (
        sc.wholeTextFiles(args.input + "/*/*")
        .sample(False, args.sampleSize)
    )

    num_files = text_files.count()

    t1 = time.perf_counter()

    print_duration("Wikipedia loading", t0, t1)

    print(f"Number of Wiki files: {num_files}")

    # ========================================================
    # Parse Documents
    # ========================================================

    t0 = time.perf_counter()

    plain_text = text_files.flatMap(parse_flat_map)

    plain_text.cache()

    num_docs = plain_text.count()

    t1 = time.perf_counter()

    print_duration("Document parsing", t0, t1)

    print(f"Number of parsed docs: {num_docs}")

    # ========================================================
    # Tokenization Pipeline
    # ========================================================

    t0 = time.perf_counter()

    if args.tokenizer == "nlp":

        lemmatized = plain_text.mapPartitions(
            lambda x: map(plain_text_to_lemmas, x)
        )

    else:

        lemmatized = plain_text.mapPartitions(
            lambda x: map(plain_text_to_regex, x)
        )

    lemmatized.cache()

    lemmatized.count()

    t1 = time.perf_counter()

    print_duration("Tokenization pipeline", t0, t1)

    print(f"Tokenizer mode: {args.tokenizer}")

    # ========================================================
    # TF Computation
    # ========================================================

    t0 = time.perf_counter()

    doc_term_freqs = lemmatized.map(calculate_term_freqs)

    doc_term_freqs.cache()

    doc_term_freqs.count()

    t1 = time.perf_counter()

    print_duration("TF computation", t0, t1)

    # ========================================================
    # Document IDs
    # ========================================================

    doc_ids = (
        doc_term_freqs
        .map(lambda x: x[0])
        .zipWithUniqueId()
        .map(lambda x: (x[1], x[0]))
        .collectAsMap()
    )

    # ========================================================
    # DF / IDF
    # ========================================================

    t0 = time.perf_counter()

    doc_freqs = (
        doc_term_freqs
        .flatMap(lambda x: x[1].keys())
        .map(lambda x: (x, 1))
        .reduceByKey(lambda x, y: x + y)
    )

    ordering = lambda x: x[1]

    top_doc_freqs = doc_freqs.top(
        args.numFreq,
        key=ordering
    )

    idfs = {
        term: math.log(num_docs / count)
        for term, count in top_doc_freqs
    }

    id_terms = dict(
        zip(idfs.keys(), range(len(idfs)))
    )

    term_ids = {
        v: k for k, v in id_terms.items()
    }

    bIdfs = sc.broadcast(idfs)

    bIdTerms = sc.broadcast(id_terms)

    t1 = time.perf_counter()

    print_duration("DF / IDF computation", t0, t1)

    print(f"Vocabulary size: {len(id_terms)}")

    # ========================================================
    # TF-IDF Vectors
    # ========================================================

    t0 = time.perf_counter()

    row_vectors = (
        doc_term_freqs
        .map(lambda x: x[1])
        .map(
            lambda term_freqs:
            Vectors.sparse(
                len(bIdTerms.value),
                [
                    (
                        bIdTerms.value[term],
                        (
                            bIdfs.value[term]
                            * term_freqs[term]
                            / sum(term_freqs.values())
                        )
                    )
                    for term in term_freqs.keys()
                    if term in bIdTerms.value
                ]
            )
        )
    )

    row_vectors.cache()

    row_vectors.count()

    mat = RowMatrix(row_vectors)

    t1 = time.perf_counter()

    print_duration("TF-IDF vectorization", t0, t1)

    # ========================================================
    # SVD
    # ========================================================

    t0 = time.perf_counter()

    svd = mat.computeSVD(args.k, computeU=True)

    t1 = time.perf_counter()

    print_duration("SVD computation", t0, t1)

    print(f"SVD latent dimensions k={args.k}")

    # ========================================================
    # Top Concepts
    # ========================================================

    top_terms_in_top_concepts(
        svd,
        term_ids,
        num_concepts=25,
        num_terms=25
    )

    top_docs_in_top_concepts(
        svd,
        doc_ids,
        num_concepts=25,
        num_docs=25
    )

    # ========================================================
    # Query Engine
    # ========================================================

    if args.query:

        print("\n" + "=" * 60)
        print("SEARCH ENGINE")
        print("=" * 60)

        if args.tokenizer == "nlp":

            query_terms = [
                lemmatizer.lemmatize(t.lower())
                for t in word_tokenize(args.query)
                if (
                    t.lower() not in bStopWords.value
                    and len(t) > 2
                )
            ]

        else:

            query_terms = [
                t.lower()
                for t in re.findall(
                    r"[A-Za-z]+|\d+",
                    args.query
                )
                if (
                    t.lower() not in bStopWords.value
                    and len(t) > 2
                )
            ]

        print(f"\nQuery terms: {query_terms}")

        query_vec = terms_to_query_vector(
            query_terms,
            id_terms,
            idfs
        )

        US = multiply_by_diagonal_row_matrix(
            svd.U,
            svd.s
        )

        top_docs = top_docs_for_term_query(
            US,
            svd.V,
            query_vec
        )

        print("\nTop matching documents:\n")

        for score, doc_id in top_docs:

            title = doc_ids.get(doc_id, str(doc_id))

            print(f"{title:80s} {score:.6f}")

    # ========================================================
    # Matrix Dimensions
    # ========================================================

    print("\n" + "=" * 60)
    print("MATRIX DIMENSIONS")
    print("=" * 60)

    print(f"U rows: {svd.U.numRows()}")
    print(f"U cols: {svd.U.numCols()}")

    print(f"V rows: {svd.V.numRows}")
    print(f"V cols: {svd.V.numCols}")

    # ========================================================
    # Shutdown
    # ========================================================

    spark.stop()

    print("\nRunLSA completed successfully.")