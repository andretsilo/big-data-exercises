import os
import sys
import math
import time
import numpy as np
from scipy.sparse import csr_matrix
from datetime import datetime

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType
from pyspark.sql.functions import udf
from pyspark.mllib.linalg import Vectors, Matrices
from pyspark.mllib.linalg.distributed import RowMatrix

# Basic hyper-parameters
sampleSize = 1
numTerms   = 5000
k          = 25

script_dir     = os.path.dirname(os.path.abspath(__file__))
csv_path       = os.path.join(script_dir, "Data", "wiki_movie_plots_deduped.csv")
stopwords_path = os.path.join(script_dir, "Data", "stopwords.txt")


def plainTextToLemmas(text, stopwords):
    import nltk
    nltk.download('punkt_tab', quiet=True)
    nltk.download('wordnet', quiet=True)
    from nltk import sent_tokenize, word_tokenize
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for sentence in sent_tokenize(text):
        for token in word_tokenize(sentence):
            lemma = lemmatizer.lemmatize(token.lower())
            if len(lemma) > 2 and lemma not in stopwords and lemma.isalpha():
                lemmas.append(lemma)
    return lemmas


def calculateTermFreqs(terms):
    termFreqs = {}
    for term in terms:
        termFreqs[term] = termFreqs.get(term, 0) + 1
    return termFreqs


def topTermsInTopConcepts(svd, termIds, numConcepts, numTerms):
    v = svd.V
    arr = v.toArray().T
    result = []
    for i in range(numConcepts):
        termWeights = sorted([(arr[i][tid], tid) for tid in range(v.numRows)], key=lambda x: -x[0])
        result.append([(termIds.get(idx, str(idx)), score) for score, idx in termWeights[:numTerms]])
    return result


def topDocsInTopConcepts(svd, docIds, titleGenres, numConcepts, numDocs):
    result = []
    for i in range(numConcepts):
        docWeights = sorted(
            [(score, did) for score, did in svd.U.rows.map(lambda row: row.toArray()[i]).zipWithUniqueId().collect()],
            key=lambda x: -x[0]
        )
        top = [(docIds.get(idx, str(idx)), score) for score, idx in docWeights[:numDocs]]
        # attach top-5 genres from the returned docs
        genres = [titleGenres.get(title, "unknown") for title, _ in top[:5]]
        result.append((top, genres))
    return result


def termsToQueryVector(terms, idTerms, idfs):
    indices = [idTerms[t] for t in terms if t in idTerms]
    values  = [idfs[t]    for t in terms if t in idTerms]
    return csr_matrix((values, (indices, [0]*len(indices))), shape=(len(idTerms), 1))


def multiplyByDiagonalRowMatrix(mat, diag):
    s_arr = diag.toArray()
    return RowMatrix(mat.rows.map(lambda vec: Vectors.dense(np.multiply(vec.toArray(), s_arr))))


def topDocsForTermQuery(US, V, query):
    term_row_arr = np.dot(V.toArray().T, query.toarray()).flatten()
    term_row_vec = Matrices.dense(len(term_row_arr), 1, term_row_arr)
    doc_scores = US.multiply(term_row_vec)
    all_doc_weights = doc_scores.rows.zipWithUniqueId().map(lambda x: (x[0].toArray()[0], x[1]))
    return sorted(all_doc_weights.collect(), key=lambda x: -x[0])[:10]


def main():
    report = []
    report.append("=" * 70)
    report.append("PROBLEM 2 - LATENT SEMANTIC ANALYSIS REPORT")
    report.append(f"Run date/time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Sample size   : {sampleSize*100:.0f}%")
    report.append(f"numTerms      : {numTerms}")
    report.append(f"k (SVD dims)  : {k}")
    report.append("=" * 70)

    total_start = time.time()

    # [1/9] Spark session
    print("[1/9] Starting Spark session...")
    print(f"      Using Python: {sys.executable}")
    spark = SparkSession.builder.appName("RunLSA").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("WARN")
    print(f"      Spark version: {spark.version}")
    report.append(f"\nSpark version : {spark.version}")
    report.append(f"Python        : {sys.executable}")

    # [2/9] Read CSV into DataFrame
    print("[2/9] Reading CSV and building DataFrame...")
    print(f"      CSV path:       {csv_path}")
    print(f"      Stopwords path: {stopwords_path}")
    t = time.time()

    raw_df = spark.read.csv(csv_path, header=True, multiLine=True, escape='"')
    plainTextDF = raw_df.select("Title", "Genre", "Plot").na.drop()
    plainTextDF.cache()

    total_rows = plainTextDF.count()
    elapsed = time.time() - t
    print(f"      Total rows after dropping nulls: {total_rows}")
    print("      Schema:")
    plainTextDF.printSchema()
    print("      Sample rows:")
    plainTextDF.show(5, truncate=80)
    report.append(f"\n--- DataFrame ---")
    report.append(f"Total rows (after null drop) : {total_rows}")
    report.append(f"CSV load time                : {elapsed:.2f}s")

    # [3/9] Load stopwords and broadcast
    print("[4/9] Loading and broadcasting stopwords...")
    stopwords = set(sc.textFile(stopwords_path).collect())
    bStopWords = sc.broadcast(stopwords)
    print(f"      Stopwords loaded: {len(bStopWords.value)}")
    report.append(f"Stopwords loaded             : {len(bStopWords.value)}")

    # [4/9] Add features column (lemmatized tokens from Plot)
    print("[3/9] Adding 'features' column with lemmatized Plot tokens...")
    t = time.time()

    @udf(returnType=ArrayType(StringType()))
    def lemmatize_udf(text):
        if text is None:
            return []
        return plainTextToLemmas(text, bStopWords.value)

    plainTextDF = plainTextDF.withColumn("features", lemmatize_udf("Plot"))
    plainTextDF.cache()

    print("      Updated schema with features column:")
    plainTextDF.printSchema()
    print("      Sample with features:")
    plainTextDF.show(3, truncate=80)

    # [5/9] Sample and convert to RDD for LSA pipeline
    print(f"[5/9] Sampling {sampleSize*100:.0f}% of rows and converting to RDD...")
    sampledDF = plainTextDF.sample(False, sampleSize, seed=42)
    sampledDF.cache()

    # build title -> genre lookup from the sample
    titleGenres = {row["Title"]: row["Genre"] for row in sampledDF.select("Title", "Genre").collect()}

    plainText = (
        sampledDF
        .rdd
        .map(lambda row: (row["Title"], row["features"] or []))
    )
    plainText.cache()

    numDocs = plainText.count()
    lemma_elapsed = time.time() - t
    bNumDocs = sc.broadcast(numDocs)
    print(f"      Sampled docs: {numDocs}")
    report.append(f"Sampled docs                 : {numDocs}")
    report.append(f"Lemmatization time           : {lemma_elapsed:.2f}s")

    # [6/9] Term frequencies
    print("[6/9] Calculating term frequencies (TF)...")
    docTermFreqs = plainText.map(lambda x: (x[0], calculateTermFreqs(x[1])))
    docTermFreqs.cache()

    # [7/9] Document IDs and IDF
    print("[7/9] Computing document IDs and IDF scores...")
    t = time.time()
    docIds = docTermFreqs.map(lambda x: x[0]).zipWithUniqueId().map(lambda x: (x[1], x[0])).collectAsMap()
    print(f"      Unique documents indexed: {len(docIds)}")

    docFreqs = (
        docTermFreqs
        .flatMap(lambda x: x[1].keys())
        .map(lambda x: (x, 1))
        .reduceByKey(lambda x, y: x + y, numPartitions=24)
    )
    topDocFreqs = docFreqs.top(numTerms, key=lambda x: x[1])

    idfs     = {term: math.log(numDocs / count) for term, count in topDocFreqs}
    idTerms  = dict(zip(idfs.keys(), range(len(idfs))))
    termIds  = {v: k for k, v in idTerms.items()}

    idf_elapsed = time.time() - t
    print(f"      Vocabulary size (top terms): {len(idfs)}")
    print(f"      Top 5 terms by doc frequency: {[t for t, _ in topDocFreqs[:5]]}")
    report.append(f"Vocabulary size              : {len(idfs)}")
    report.append(f"Top 5 frequent terms         : {[t for t, _ in topDocFreqs[:5]]}")
    report.append(f"IDF computation time         : {idf_elapsed:.2f}s")

    bIdfs    = sc.broadcast(idfs)
    bIdTerms = sc.broadcast(idTerms)

    # [8/9] TF-IDF row vectors and SVD
    print("[8/9] Building TF-IDF row vectors and computing SVD...")
    print(f"      Matrix shape will be: {numDocs} docs x {len(idfs)} terms, reduced to k={k}")

    rowVectors = docTermFreqs.map(lambda x: x[1]).map(
        lambda termFreqs: Vectors.sparse(
            len(bIdTerms.value),
            [(bIdTerms.value[term], bIdfs.value[term] * termFreqs[term] / sum(termFreqs.values()))
             for term in termFreqs if term in bIdTerms.value]
        )
    )
    rowVectors.cache()

    mat = RowMatrix(rowVectors)
    print("      Running SVD (this may take a moment)...")
    t = time.time()
    svd = mat.computeSVD(k, computeU=True)
    svd_elapsed = time.time() - t
    print(f"      SVD complete. Singular values: {svd.s.toArray().round(4)}")
    report.append(f"\n--- SVD ---")
    report.append(f"Matrix shape                 : {numDocs} docs x {len(idfs)} terms")
    report.append(f"Reduced dimensions (k)       : {k}")
    report.append(f"SVD computation time         : {svd_elapsed:.2f}s")
    report.append(f"Singular values              : {svd.s.toArray().round(4).tolist()}")
    report.append(f"UxS matrix                   : {numDocs} rows x {k} cols")
    report.append(f"V matrix                     : {svd.V.numRows} rows x {svd.V.numCols} cols")

    # [9/9] Query concepts and keyword search
    print("[9/9] Querying latent concepts and running keyword search...")

    top_concept_terms = topTermsInTopConcepts(svd, termIds, k, 25)
    top_concept_docs  = topDocsInTopConcepts(svd, docIds, titleGenres, k, 25)

    report.append("\n--- Top-25 Terms and Docs per Latent Concept ---")
    print("\n--- Top terms and docs per latent concept ---")
    for i, (terms, (docs, genres)) in enumerate(zip(top_concept_terms, top_concept_docs)):
        print(f"Concept {i+1} terms: ", ", ".join(t for t, _ in terms))
        print(f"Concept {i+1} docs:  ", ", ".join(d for d, _ in docs))
        print(f"Concept {i+1} top-5 genres: ", ", ".join(genres))
        print()
        report.append(f"\nConcept {i+1}")
        report.append(f"  Terms  : {', '.join(t for t, _ in terms)}")
        report.append(f"  Docs   : {', '.join(d for d, _ in docs)}")
        report.append(f"  Genres : {', '.join(genres)}")

    US = multiplyByDiagonalRowMatrix(svd.U, svd.s)

    queries = [
        ["murder", "detective", "suspect"],
        ["love", "romance", "wedding"],
        ["war", "soldier", "battle"],
        ["monster", "creature", "horror"],
        ["space", "alien", "planet"],
        ["heist", "robbery", "gang"],
        ["king", "kingdom", "throne"],
        ["ghost", "haunted", "spirit"],
        ["drug", "dealer", "crime"],
        ["revenge", "betrayal", "enemy"],
    ]

    report.append("\n--- Keyword Queries ---")
    print("\n--- Keyword Queries ---")
    for query_terms in queries:
        queryVec = termsToQueryVector(query_terms, idTerms, idfs)
        results  = topDocsForTermQuery(US, svd.V, queryVec)
        print(f"\nQuery: {query_terms}")
        report.append(f"\nQuery: {query_terms}")
        for score, doc_id in results:
            title = docIds.get(doc_id, doc_id)
            genre = titleGenres.get(title, "unknown")
            line  = f"  score={score:.6f}  title='{title}'  genre='{genre}'"
            print(line)
            report.append(line)

    total_elapsed = time.time() - total_start
    report.append("\n" + "=" * 70)
    report.append(f"Total runtime : {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)")
    report.append("=" * 70)

    print("\n--- Matrix dimensions ---")
    print(f"  UxS : {US.numRows()} rows x {US.numCols()} cols")
    print(f"  V   : {svd.V.numRows} rows x {svd.V.numCols} cols")
    print(f"\nTotal runtime: {total_elapsed:.2f}s ({total_elapsed/60:.2f} min)")
    print("\n[Done]")

    # Write report to Problem_2.txt
    report_path = os.path.join(script_dir, "Problem_2.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"Report written to: {report_path}")

    spark.stop()


if __name__ == "__main__":
    main()
