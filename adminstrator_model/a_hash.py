import numpy as np
import pandas as pd
import hashlib
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from scipy.optimize import lsq_linear
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Constants
NUM_BLOOMBITS = 1438
lambda_lasso = 0.01
num_trials = 10
threshold = 0.000001
prob_p, prob_q, prob_f = 0.5, 0.75, 0.5
query_size = 10
views_per_url = 1000
candidate_size = 100

# Load dataset
urls = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")
urls_mal_all = urls[urls['label'] == 0]['URL'].values
urls_norm_all = urls[urls['label'] == 1]['URL'].values

# RAPPOR core functions
def hash_domain(url, num_bloombits, num_hashes):
    bits = np.zeros(num_bloombits, dtype=bool)
    for i in range(num_hashes):
        salted_input = f"{i}:{url}".encode()
        h = hashlib.sha256(salted_input).hexdigest()
        idx = int(h, 16) % num_bloombits
        bits[idx] = True
    return bits

def create_candidate_set(urls, num_bloombits, num_hashes):
    return np.array([hash_domain(url, num_bloombits, num_hashes) for url in urls])

def prr_view(url, num_bloombits, num_hashes, f):
    bits = hash_domain(url, num_bloombits, num_hashes)
    keep_mask = np.random.rand(num_bloombits) < 1 - f
    random_bits = np.random.rand(num_bloombits) < 0.5
    return np.where(keep_mask, bits, random_bits)

def irr_view(prr, p, q):
    rand = np.random.rand(len(prr))
    return np.where(prr, rand < q, rand < p)

# Sweep over different numbers of hashes
hash_counts = [2, 5, 10, 15, 20]
results = []

for num_hashes in hash_counts:
    precisions, recalls, f1s, accuracies, in_candidates = [], [], [], [], []

    # Recompute epsilon_1
    q_star = 0.5 * prob_f * (prob_p + prob_q) + (1 - prob_f) * prob_q
    p_star = 0.5 * prob_f * (prob_p + prob_q) + (1 - prob_f) * prob_p
    epsilon_1 = num_hashes * np.log((q_star * (1 - p_star)) / (p_star * (1 - q_star)))

    # Candidate set
    candidate_urls = np.random.choice(urls_mal_all, candidate_size, replace=False)
    candidate_set = create_candidate_set(candidate_urls, NUM_BLOOMBITS, num_hashes)

    for trial in range(num_trials):
        phishing_url = np.random.choice(candidate_urls, 1).tolist()
        normal_urls = np.random.choice(urls_norm_all, query_size - 1, replace=False).tolist()
        query_set = normal_urls + phishing_url

        prr_views = np.array([prr_view(url, NUM_BLOOMBITS, num_hashes, prob_f) for url in query_set])

        irr_matrix = []
        for _ in range(views_per_url):
            for prr in prr_views:
                irr_matrix.append(irr_view(prr, prob_p, prob_q))
        irr_matrix = np.array(irr_matrix)

        y = irr_matrix.mean(axis=0)
        alpha = lambda_lasso / candidate_set.shape[0]
        model = make_pipeline(
            StandardScaler(),
            Lasso(alpha=alpha, fit_intercept=True, positive=False, tol=1e-4, max_iter=2000)
        )

        try:
            model.fit(candidate_set.T, y)
            lasso_coef = model.named_steps['lasso'].coef_
            reconstructed_freq = np.maximum(lasso_coef, 0)
        except:
            res = lsq_linear(candidate_set.T, y, bounds=(0, np.inf))
            reconstructed_freq = res.x

        detected_indices = np.where(reconstructed_freq > threshold)[0]
        urls_detected = candidate_urls[detected_indices].tolist()

        tp = 1 if phishing_url[0] in urls_detected else 0
        fp = sum(1 for url in urls_detected if url != phishing_url[0])
        fn = 1 - tp

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        accuracy = tp  # 1 if detected, 0 if not
        in_candidate = sum(1 for url in urls_detected if url in candidate_urls)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        accuracies.append(accuracy)
        in_candidates.append(in_candidate)

    results.append((
        num_hashes, epsilon_1,
        np.mean(precisions), np.mean(recalls), np.mean(f1s),
        np.mean(accuracies), np.mean(in_candidates)
    ))

# Results DataFrame
results_df = pd.DataFrame(results, columns=[
    'Num_Hashes', 'Epsilon_1', 'Precision', 'Recall', 'F1', 'Accuracy', 'In_Candidate'
])
print(results_df)

# Normalize In_Candidate for fair comparison
results_df['In_Candidate_Norm'] = results_df['In_Candidate'] / candidate_size

# Normalize In_Candidate
results_df['In_Candidate_Norm'] = results_df['In_Candidate'] / candidate_size

# --- Combined Plot: Detection Metrics + In Candidate ---
plt.figure(figsize=(9, 5))
plt.plot(results_df["Num_Hashes"], results_df["F1"], marker='o', label='F1-score')
plt.plot(results_df["Num_Hashes"], results_df["Precision"], marker='x', linestyle='--', label='Precision')
plt.plot(results_df["Num_Hashes"], results_df["Recall"], marker='^', linestyle=':', label='Recall')
plt.plot(results_df["Num_Hashes"], results_df["Accuracy"], marker='d', linestyle='-.', label='Detection Accuracy')
plt.plot(results_df["Num_Hashes"], results_df["In_Candidate_Norm"], marker='v', linestyle='-', label='In Candidate (Normalized)', color='green')

plt.title("Detection Metrics vs Number of Hash Functions")
plt.xlabel("Number of Hash Functions")
plt.ylabel("Score (0–1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("rappor_detection_metrics_vs_hashes.png", dpi=300) 
plt.show()


# --- Plot 2: Privacy Loss ε₁ ---
plt.figure(figsize=(8, 4))
plt.plot(results_df["Num_Hashes"], results_df["Epsilon_1"], marker='s', linestyle='-', color='gray')
plt.title("Privacy Loss (ε₁) vs Number of Hash Functions")
plt.xlabel("Number of Hash Functions")
plt.ylabel("ε₁ (Instantaneous Differential Privacy)")
plt.grid(True)
plt.tight_layout()
plt.savefig("ε₁_vs_Number_of_Hash_Functions.png", dpi=300) 
plt.show()



