#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 16:40:38 2023

@author: maximestaleman
"""

#%% Packages

import pandas as pd
import numpy as np
import json
import re
import collections
import math
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import binarize
from tqdm import tqdm
import random
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, make_scorer, precision_score, recall_score
from scipy.spatial.distance import euclidean
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

#%% Data cleaning

json_file = open("TVs-all-merged.json")
data = json.load(json_file)

data_frame = pd.json_normalize(data).transpose()

dictionary_list = []
model_id = []
for key, value in data.items():
    for value in data[key]:
        dictionary_list.append(key)
        dictionary_list.append(value)
        model_id.append(key)

# list of only the product descriptions
productDescriptions = dictionary_list[1::2]

titles = []
for title in productDescriptions:
    value = title['title']
    titles.append(value)
    
def process_titles(titles):
    processed_titles = []
    brands = []
    # Define terms to be removed
    remove_terms = ["refurbished", "newegg.com", "best", "buy", "tv", "-", "diagonal", "diag", "wide", "widescreen", "open","box", ":", "/"]

    for title in titles:
        # Remove words that are close or similar to the defined terms
        title = title.lower()
        title = re.sub(r'\([^)]*\)', '', title)
        title = title.replace('"', 'inch')
        title = title.replace("-inch", 'inch')
        title = title.replace('inches', 'inch')
        title = title.replace('led-lcd', 'ledlcd')

        title_words = title.split()
        cleaned_title_words = [word for word in title_words if all(fuzz.ratio(word, term) < 80 for term in remove_terms)]
        brand = cleaned_title_words[0]
        cleaned_title = ' '.join(cleaned_title_words)
        processed_titles.append(cleaned_title)
        brands.append(brand)

    #all_words = ' '.join(processed_titles).split()
    #word_counts = Counter(all_words)

    # Get a set of words that appear more than once
    #common_words = {word for word, count in word_counts.items() if count > 1}

    # Remove words that appear only once in each title
    #processed_titles2 = [' '.join([word for word in title.split() if word in common_words]) for title in processed_titles]

    return processed_titles, brands

new_titles, brands = process_titles(titles)

def true_pairs(model_id):
    true_matching_pairs = []

    # Iterate through unique model_ids
    unique_model_ids = set(model_id)
    for model in unique_model_ids:
        # Find indices where model_id matches
        matching_indices = [i for i, m in enumerate(model_id) if m == model]
        # Create pairs from matching indices
        pairs = combinations(matching_indices, 2)
        true_matching_pairs.extend(pairs)
        
    return true_matching_pairs

def extract_features(tv_data):
    # Initialize lists to store extracted features
    screen_sizes = []
    resolutions = []

    # Extract features
    for model_id, listings in tv_data.items():
        for listing in listings:
            features = listing.get("featuresMap", {})
            # Extract screen size
            screen_size = features.get("Screen Size Class", "unknown")
            screen_sizes.append(screen_size)
            # Extract resolution
            resolution = features.get("Vertical Resolution", "unknown")
            resolutions.append(resolution)

    # Create a DataFrame
    df = pd.DataFrame({
        "ScreenSize": screen_sizes,
        "Resolution": resolutions
    })

    # Preprocess features
    # For categorical features, use one-hot encoding
    df = pd.get_dummies(df, columns=["ScreenSize", "Resolution"], drop_first=True)

    return df

# Extract and preprocess features
features_df = extract_features(data)
features_df.head()

true_matching_pairs = true_pairs(model_id)

#%% LSH

def tokens(titles):
    title = re.sub(r'[^\w\s]', '', titles)
    tokens = title.split()
    return tokens

tokenized_titles = [tokens(title) for title in new_titles]

flat_tokenized_titles = [' '.join(title) for title in tokenized_titles]
vectorizer = CountVectorizer(binary=True)
binary_vectors = vectorizer.fit_transform(flat_tokenized_titles).toarray()
 
prime_list = [1423, 1433, 1447, 1451, 1453]

def minhash(binaryVector, prime = 1433):
    np.random.seed(10)
    matrix = binaryVector
    numberRows, numberProducts = binaryVector.shape
    numHash = int(numberRows / 2)
    signature_matrix = np.full((numHash, numberProducts), np.inf)
    hash_functions = []
    # Set seed
    # generate the signature matrix using minhashing
    for row in tqdm(range(numberRows)):
        hash_row = []
        for i in range(numHash):
            int1 = random.randint(0, 2 ** 32 - 1)
            int2 = random.randint(0, 2 ** 32 - 1)
            hash_value = (int1 + int2 * (row + 1)) % prime
            hash_row.append(hash_value)
        hash_functions.append(hash_row)
        for column in range(numberProducts):
            if matrix[row][column] == 0:
                continue
            for i in range(numHash):
                value = hash_functions[row][i]
                if signature_matrix[i][column] > value:
                    signature_matrix[i][column] = value
    return signature_matrix


def localitySensitiveHashing(signatureMatrix, b):
    np.random.seed(10)
    n, d = signatureMatrix.shape
    r = int(n / b)
    threshold = math.pow((1 / b), (1 / r))
    
    buckets = collections.defaultdict(set)
    bands = np.array_split(signatureMatrix, b, axis=0)
    for band_index, band in enumerate(bands):
        for column_index in range(d):
        # Extract the column values in the current band
            column_values = band[:, column_index]
            bucket_id = tuple(column_values) + (band_index,)
            buckets[bucket_id].add(column_index)
    candidate_pairs = set()
    for bucket in buckets.values():
        if len(bucket) > 1:
            for pair in combinations(bucket, 2):
                candidate_pairs.add(pair)
    return candidate_pairs

def filter_pairs_by_brand(candidate_pairs, brands):
    filtered_pairs = []
    for pair in candidate_pairs:
        idx_1, idx_2 = pair
        brand_1 = brands[idx_1]
        brand_2 = brands[idx_2]
        if brand_1 == brand_2:
            filtered_pairs.append(pair)
    return filtered_pairs


def calculate_f1star(candidate_pairs, true_matching_pairs, model_id):
    true_labels = [1 if pair in true_matching_pairs else 0 for pair in candidate_pairs]

    # Calculate Precision, Recall, and F1-star
    tp = sum(true_labels)
    total_actual_duplicates = len(true_matching_pairs)
    total_predicted_duplicates = len(candidate_pairs)

    precision = tp / total_predicted_duplicates if total_predicted_duplicates > 0 else 0
    recall = tp / total_actual_duplicates if total_actual_duplicates > 0 else 0

    f1star = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1star

b = 220
binaryVectors = binary_vectors.transpose()
signature_matrix = minhash(binaryVectors)
candidate_pairs = localitySensitiveHashing(signature_matrix, b)
filtered_pairs = filter_pairs_by_brand(candidate_pairs, brands)
precision, recall, f1star = calculate_f1star(filtered_pairs, true_matching_pairs, model_id)
print("F1-star:", f1star)

#%% Plots

def calculate_metrics_for_b(lsh_function, true_matching_pairs, b, signature_matrix, binary_vectors, model_id, brands):
    
    candidate_pairs = lsh_function(signature_matrix, b)
    filtered_pairs = filter_pairs_by_brand(candidate_pairs, brands)
    features, labels = generate_labels(filtered_pairs, binary_vectors, signature_matrix, model_id)
    avg_f1, avg_precision, avg_recall = train_evaluate_random_forest(features, labels, num_bootstraps=5)
    
    precision, recall, f1_star = calculate_f1star(candidate_pairs, true_matching_pairs, model_id)
    fraction = fracComparisons(candidate_pairs)

    return precision, recall, f1_star, fraction, avg_f1, avg_precision, avg_recall


def fracComparisons(candidates):
    numCom = len(candidates)
    n = 1624
    totalCom = n**2
    fraction = numCom / totalCom
    return fraction

def plot_pair_completeness(metric_results, b_values):
    recalls = [result[1] for result in metric_results]  # Recall is the second element in the tuple
    fractions = [result[3] for result in metric_results]  # Fraction of comparisons is the fourth element
    avg_recall = [result[6] for result in metric_results]
    
    plt.figure()
    plt.plot(fractions, recalls, label='LSH')
    plt.plot(fractions, avg_recall, label='LSH + RF')
    plt.xlabel('Fractions of comparison')
    plt.ylabel('Recall')
    plt.title('Pair Completeness')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to generate the Pair Quality Graph
def plot_pair_quality(metric_results, b_values):
    precisions = [result[0] for result in metric_results]
    fractions = [result[3] for result in metric_results]
    avg_precision = [result[5] for result in metric_results]

    plt.figure()
    plt.plot(fractions, precisions, label='LSH')
    plt.plot(fractions, avg_precision, label='LSH + RF')
    plt.xlabel('Fractions of comparison')
    plt.ylabel('Precision')
    plt.title('Pair Quality')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to generate the F1 Star Measure Graph
def plot_f1_star(metric_results, b_values):
    f1_stars = [result[2] for result in metric_results]  # F1 star is the third element in the tuple
    fractions = [result[3] for result in metric_results]
    avg_f1 = [result[4] for result in metric_results]

    plt.figure()
    plt.plot(fractions, f1_stars, label='LSH')
    plt.plot(fractions, avg_f1, label='LSH + RF')
    plt.xlabel('Fractions of comparison')
    plt.ylabel('F1')
    plt.title('F1')
    plt.legend()
    plt.grid(True)
    plt.show()
   

b_values = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 130, 140, 150, 200, 220, 240, 280, 300]
metric_results = [calculate_metrics_for_b(localitySensitiveHashing, true_matching_pairs, b, signature_matrix, binary_vectors, model_id, brands) for b in b_values]


plot_pair_completeness(metric_results, b_values)
plot_pair_quality(metric_results, b_values)
plot_f1_star(metric_results, b_values)

#%% Random Forest

def jaccard_similarity(vector1, vector2):
    intersection = np.sum(np.logical_and(vector1, vector2))
    union = np.sum(np.logical_or(vector1, vector2))
    return intersection / union

def generate_labels(filtered_pairs, binary_vectors, signature_matrix, model_id):
    features = []
    labels = []

    for candidate_pair in filtered_pairs:
        idx_1, idx_2 = candidate_pair

        arr1 = binary_vectors[idx_1]
        arr2 = binary_vectors[idx_2]
        jaccard_similarity_score = jaccard_similarity(arr1, arr2)

        sig1 = signature_matrix[:, idx_1]
        sig2 = signature_matrix[:, idx_2]
        COS = np.vstack((sig1.transpose(), sig2.transpose()))
        cosine_similarity_matrix = cosine_similarity(COS)
        cosine_similarity_score = cosine_similarity_matrix[0, 1]
        
        euclidean_distance = euclidean(arr1, arr2)
        
        features.append([jaccard_similarity_score, cosine_similarity_score, euclidean_distance])

        labels.append(1 if model_id[idx_1] == model_id[idx_2] else 0)

    return features, labels

def train_evaluate_random_forest(features, labels, num_bootstraps=5, test_size=0.3):
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    for _ in range(num_bootstraps):
        # Create a bootstrap sample
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=np.random.randint(1, 1000))
        
        # Train the Random Forest Model
        random_forest_model = RandomForestClassifier(max_depth=30, max_features='sqrt', min_samples_split=2, n_estimators=100, random_state=42)
        random_forest_model.fit(X_train, y_train)

        # Make Predictions
        predictions = random_forest_model.predict(X_test)

        # Calculate metrics for this bootstrap
        f1 = f1_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)

        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)

    # Calculate the average scores across bootstraps
    avg_f1 = np.mean(f1_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)

    return avg_f1, avg_precision, avg_recall

# Use the function to evaluate the Random Forest with bootstrapping
num_bootstraps = 5  # You can adjust the number of bootstraps as needed
features, labels = generate_labels(filtered_pairs, binary_vectors, signature_matrix, model_id)
avg_f1, avg_precision, avg_recall = train_evaluate_random_forest(features, labels, num_bootstraps=num_bootstraps)
print("Average F1-measure across bootstraps:", avg_f1)

#%% Grid search

# Define the parameter grid to search
param_grid = {
    'n_estimators': [50, 100, 200],        # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],       # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],      # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],        # Minimum number of samples required to be a leaf node
    'max_features': ['auto', 'sqrt'],     # Number of features to consider when looking for the best split
    'bootstrap': [True, False],           # Whether to use bootstrapping
    'random_state': [42],                # Random seed for reproducibility
}

# Create a random forest classifier
rf_classifier = RandomForestClassifier()

# Create GridSearchCV with the classifier and parameter grid
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring={'f1_class1': make_scorer(f1_score, pos_label=1), 
                                    'precision_class1': make_scorer(precision_score, pos_label=1),
                                    'recall_class1': make_scorer(recall_score, pos_label=1)},
                           refit='f1_class1', n_jobs=-1)
X_trainG, X_testG, y_trainG, y_testG = train_test_split(features, labels, test_size=0.3, random_state=42) 
# Fit the grid search to your data
grid_search.fit(X_trainG, y_trainG)

# Print the best hyperparameters found
print("Best Hyperparameters:", grid_search.best_params_)

# Get the best model with the best hyperparameters
best_rf_model = grid_search.best_estimator_

# Make predictions with the best model
predictions = best_rf_model.predict(X_testG)

# Evaluate the model
accuracy = accuracy_score(y_testG, predictions)
print(f"Accuracy: {accuracy}")

print("Classification Report:")
print(classification_report(y_testG, predictions))