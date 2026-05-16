import os
import sys
import random
import logging

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.mllib.recommendation import ALS, Rating
from pyspark.mllib.evaluation import BinaryClassificationMetrics

print(os.getcwd())
print(sys.executable)

logging.getLogger("py4j").setLevel(logging.ERROR)

spark = SparkSession.builder \
    .appName("Exercise2") \
    .getOrCreate()
sc = spark.sparkContext

rawArtistAlias = sc.textFile("/mnt/aiongpfs/users/atsilogiannis/recommender-system/Data/audioscrobbler/artist_alias.txt")
rawArtistData  = sc.textFile("/mnt/aiongpfs/users/atsilogiannis/recommender-system/Data/audioscrobbler/artist_data.txt")

rawUserArtistData = sc.textFile("/mnt/aiongpfs/users/atsilogiannis/recommender-system/Data/audioscrobbler/user_artist_data.txt")
rawUserArtistData = rawUserArtistData.sample(False, 0.3, seed=42)

artistAlias = (
    rawArtistAlias
    .map(lambda line: line.split('\t'))
    .filter(lambda tokens: tokens[0].strip() != "")
    .map(lambda tokens: (int(tokens[0]), int(tokens[1])))
    .collectAsMap()
)

def parse_artist_data(line):
    try:
        artist_id, artist_name = line.split('\t')
        return [(int(artist_id), artist_name.strip())]
    except ValueError:
        return []

artistByID = rawArtistData.flatMap(parse_artist_data)
bArtistAlias = sc.broadcast(artistAlias)

allData = (
    rawUserArtistData
    .map(lambda line: line.split(' '))
    .map(lambda tokens: (int(tokens[0]), bArtistAlias.value.get(int(tokens[1]), int(tokens[1])), int(tokens[2])))
    .map(lambda data: Rating(data[0], data[1], data[2]))
    .cache()
)

print("First entry:", allData.first())

# (a)(i) Filter to users with >= 100 distinct artists
qualifiedUsers = set(
    allData
    .map(lambda r: (r.user, r.product))
    .distinct()
    .map(lambda x: (x[0], 1))
    .reduceByKey(lambda a, b: a + b)
    .filter(lambda x: x[1] >= 100)
    .map(lambda x: x[0])
    .collect()
)

bQualifiedUsers = sc.broadcast(qualifiedUsers)
trainData100 = allData.filter(lambda r: r.user in bQualifiedUsers.value).cache()
allData.unpersist()

print("Qualified users:", len(qualifiedUsers))
print("trainData100 count:", trainData100.count())

# (a)(ii) Per-user 80/20 split
def split_user_ratings(ratings, seed=42):
    rng = random.Random(seed)
    shuffled = list(ratings)
    rng.shuffle(shuffled)
    split = max(1, int(len(shuffled) * 0.8))
    return shuffled[:split], shuffled[split:]

def emit_splits(x):
    train, test = split_user_ratings(x[1])
    return [('train', r) for r in train] + [('test', r) for r in test]

splits = trainData100.groupBy(lambda r: r.user).flatMap(emit_splits)

trainData = splits.filter(lambda x: x[0] == 'train').map(lambda x: x[1]).cache()
testData  = splits.filter(lambda x: x[0] == 'test').map(lambda x: x[1]).cache()

print('trainData count:', trainData.count())
print('testData count:', testData.count())

train_users = set(trainData.map(lambda r: r.user).distinct().collect())
test_users  = set(testData.map(lambda r: r.user).distinct().collect())
print("Same users in both splits:", train_users == test_users)

# Helper: compute AUC given a model, evaluation RDD, and ground truth RDD
def compute_auc(model, eval_rdd, actual_artists_rdd):
    eval_users = sc.broadcast(set(eval_rdd.map(lambda r: r.user).distinct().collect()))
    recs = model.recommendProductsForUsers(50).filter(lambda x: x[0] in eval_users.value)
    pal = (
        recs
        .join(actual_artists_rdd)
        .flatMap(lambda x: [(float(r.rating), 1.0 if r.product in x[1][1] else 0.0) for r in x[1][0]])
    )
    return BinaryClassificationMetrics(pal).areaUnderROC

# 5-fold CV using RDD-native splitting (avoids collecting to Python)
print('\nPreparing CV folds...')
K = 5
indexedTrain = trainData.zipWithIndex().map(lambda x: (x[1] % K, x[0])).cache()
folds = [
    (
        indexedTrain.filter(lambda x, i=i: x[0] != i).map(lambda x: x[1]),
        indexedTrain.filter(lambda x, i=i: x[0] == i).map(lambda x: x[1])
    )
    for i in range(K)
]

# Hyperparameter grid search
ranks   = [25, 50]
lambdas = [1.0, 0.1, 0.01]
alphas  = [1.0]

print('\nStarting hyperparameter grid search with 5-fold CV...')
evaluations = []

for rank in ranks:
    for lam in lambdas:
        for alpha in alphas:
            fold_aucs = []
            for train_fold, val_fold in folds:
                train_fold_c = train_fold.cache()
                val_fold_c   = val_fold.cache()
                val_actual = (
                    val_fold_c
                    .map(lambda r: (r.user, r.product))
                    .groupByKey()
                    .mapValues(set)
                )
                m = ALS.trainImplicit(train_fold_c, rank=rank, iterations=5, lambda_=lam, alpha=alpha)
                fold_aucs.append(compute_auc(m, val_fold_c, val_actual))
                train_fold_c.unpersist()
                val_fold_c.unpersist()
            avg_auc = sum(fold_aucs) / len(fold_aucs)
            evaluations.append(((rank, lam, alpha), avg_auc))
            print(f'  rank={rank}, lambda={lam}, alpha={alpha} -> CV AUC={avg_auc:.4f}')
            # Save progress after each combination
            with open('cv_results.txt', 'a') as f:
                f.write(f'rank={rank}, lambda={lam}, alpha={alpha}, CV AUC={avg_auc:.4f}\n')

evaluations.sort(key=lambda x: x[1], reverse=True)
indexedTrain.unpersist()
print('\nTop 5 hyperparameter combinations:')
for params, auc in evaluations[:5]:
    print(f'  rank={params[0]}, lambda={params[1]}, alpha={params[2]} -> AUC={auc:.4f}')

best_params, best_cv_auc = evaluations[0]
best_rank, best_lambda, best_alpha = best_params
print(f'\nBest params: rank={best_rank}, lambda={best_lambda}, alpha={best_alpha}, CV AUC={best_cv_auc:.4f}')

# Train final model on full trainData with best hyperparameters
best_model = ALS.trainImplicit(trainData, rank=best_rank, iterations=5, lambda_=best_lambda, alpha=best_alpha)
print('Best model trained.')

# Final evaluation on testData
actualArtistsForUser = (
    testData
    .map(lambda r: (r.user, r.product))
    .groupByKey()
    .mapValues(set)
)

testUsers  = testData.map(lambda r: r.user).distinct()
top50Recs  = best_model.recommendProductsForUsers(50)
bTestUsers = sc.broadcast(set(testUsers.collect()))
top50Recs  = top50Recs.filter(lambda x: x[0] in bTestUsers.value)

def make_predictions_and_labels(user_recs_actual):
    user, (recs, actual_set) = user_recs_actual
    return [(float(r.rating), 1.0 if r.product in actual_set else 0.0) for r in recs]

predictionsAndLabels = (
    top50Recs
    .join(actualArtistsForUser)
    .flatMap(make_predictions_and_labels)
)

metrics = BinaryClassificationMetrics(predictionsAndLabels)
auc_als = metrics.areaUnderROC
print(f'\nFinal evaluation (best model: rank={best_rank}, lambda={best_lambda}, alpha={best_alpha}):')
print(f'AUC (ALS model): {auc_als:.4f}')

pal_collected = predictionsAndLabels.collect()
tp = sum(1 for score, label in pal_collected if score > 0 and label == 1.0)
fp = sum(1 for score, label in pal_collected if score > 0 and label == 0.0)
fn = sum(1 for score, label in pal_collected if score <= 0 and label == 1.0)
tn = sum(1 for score, label in pal_collected if score <= 0 and label == 0.0)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
accuracy  = (tp + tn) / len(pal_collected) if pal_collected else 0.0
print(f'Precision: {precision:.4f}')
print(f'Recall:    {recall:.4f}')
print(f'Accuracy:  {accuracy:.4f}')

# Baseline: most popular artists
top50Popular  = [aid for aid, _ in trainData.map(lambda r: (r.product, r.rating)).reduceByKey(lambda a, b: a + b).sortBy(lambda x: x[1], ascending=False).take(50)]
bTop50Popular = sc.broadcast(top50Popular)

predictionsAndLabelsBaseline = (
    actualArtistsForUser
    .flatMap(lambda x: [(50.0 - i, 1.0 if artist in x[1] else 0.0) for i, artist in enumerate(bTop50Popular.value)])
)

auc_baseline = BinaryClassificationMetrics(predictionsAndLabelsBaseline).areaUnderROC
print(f'AUC (Most Popular baseline): {auc_baseline:.4f}')

# Task 3: Add a new user and get recommendations
# New user ID — must not exist in the dataset
new_user_id = 9999999

# Artist IDs chosen from audioscrobbler dataset (rap artists)
new_user_ratings = [
    Rating(new_user_id, 829,     1000),  # Nas
    Rating(new_user_id, 930,     1000),  # Eminem
    Rating(new_user_id, 1811,    1000),  # Dr. Dre
    Rating(new_user_id, 1001819, 1000),  # 2Pac
    Rating(new_user_id, 1007435, 1000),  # Big L
    Rating(new_user_id, 1004028, 1000),  # Notorious B.I.G.
    Rating(new_user_id, 250,     1000),  # Outkast
    Rating(new_user_id, 1004496, 1000),  # Ice Cube
]

# Combine new user ratings with full trainData and retrain
augmentedTrainData = trainData.union(sc.parallelize(new_user_ratings)).cache()
new_model = ALS.trainImplicit(augmentedTrainData, rank=best_rank, iterations=5, lambda_=best_lambda, alpha=best_alpha)

# Get top-25 recommendations for the new user
top25 = new_model.recommendProducts(new_user_id, 25)

# Look up artist names
artistMap = dict(artistByID.collect())
print(f'\nTop-25 recommended artists for new user {new_user_id}:')
for i, r in enumerate(top25, 1):
    print(f'  {i:2}. {artistMap.get(r.product, f"Unknown({r.product})")}')
