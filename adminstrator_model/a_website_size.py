import numpy as np
import pandas as pd
import hashlib
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from scipy.optimize import lsq_linear
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# --- Constants ---
NUM_BLOOMBITS = 1438
NUM_HASHES = 10
lambda_lasso = 0.01
num_trials = 10
threshold = 0.000001

# Fixed privacy parameters
prob_p, prob_q, prob_f = 0.5, 0.75, 0.5
# --- Compute ε₁ once for this privacy setting ---
q_star = 0.5 * prob_f * (prob_p + prob_q) + (1 - prob_f) * prob_q
p_star = 0.5 * prob_f * (prob_p + prob_q) + (1 - prob_f) * prob_p
epsilon_1 = NUM_HASHES * np.log((q_star * (1 - p_star)) / (p_star * (1 - q_star)))
print(f"Instantaneous privacy guarantee (ε₁): {epsilon_1:.4f}")


# --- Vary this: total query set size (1 phishing + N-1 normal)
query_sizes = [5, 10, 20, 50, 100]

# --- Load data
urls = pd.read_csv('PhiUSIIL_Phishing_URL_Dataset.csv')
urls_mal_all = urls[urls['label'] == 0]['URL'].values
urls_norm_all = urls[urls['label'] == 1]['URL'].values

# --- Hash functions
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

# --- Prepare candidate set
urls_mal_sample = np.random.choice(urls_mal_all, 100, replace=False)
candidate_set = create_candidate_set(urls_mal_sample, NUM_BLOOMBITS, NUM_HASHES)

# --- Run Experiment
results = []

for query_size in query_sizes:
    precisions, recalls, f1s = [], [], []

    for trial in range(num_trials):
        phishing_url = np.random.choice(urls_mal_sample, 1).tolist()
        num_normal = query_size - 1
        normal_urls = np.random.choice(urls_norm_all, num_normal, replace=False).tolist()
        query_set = normal_urls + phishing_url

        # --- PRR
        prr_views = np.array([prr_view(url, NUM_BLOOMBITS, NUM_HASHES, prob_f) for url in query_set])

        # --- IRR
        irr_matrix = []
        for _ in range(1000):
            for prr in prr_views:
                irr_matrix.append(irr_view(prr, prob_p, prob_q))
        irr_matrix = np.array(irr_matrix)

        # --- LASSO
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

        # --- Detection
        detected_indices = np.where(reconstructed_freq > threshold)[0]
        urls_detected = urls_mal_sample[detected_indices].tolist()

        # --- Metric
        tp = 1 if phishing_url[0] in urls_detected else 0
        fp = sum(1 for url in urls_detected if url != phishing_url[0])
        fn = 1 - tp

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    results.append((query_size, np.mean(precisions), np.mean(recalls), np.mean(f1s)))

# --- Output
# Add extra comparison columns
results_df = pd.DataFrame(results, columns=['Query_Size', 'Precision', 'Recall', 'F1'])
results_df['Malicious_Count'] = 1
results_df['Normal_Count'] = results_df['Query_Size'] - 1
results_df['Malicious_Ratio'] = results_df['Malicious_Count'] / results_df['Query_Size']
results_df['Normal_Ratio'] = results_df['Normal_Count'] / results_df['Query_Size']

# Reorder for clarity
results_df = results_df[[
    'Query_Size',
    'Malicious_Count',
    'Normal_Count',
    'Malicious_Ratio',
    'Normal_Ratio',
    'Precision',
    'Recall',
    'F1'
]]

print("\n=== Comparison Table: Query Composition vs Detection Metrics ===")
print(results_df.to_string(index=False))

# --- Plot
plt.figure(figsize=(8, 5))
plt.plot(results_df["Query_Size"], results_df["F1"], marker='o', label='F1-score')
plt.plot(results_df["Query_Size"], results_df["Precision"], marker='x', linestyle='--', label='Precision')
plt.plot(results_df["Query_Size"], results_df["Recall"], marker='^', linestyle=':', label='Recall')
plt.xlabel("Total Query Set Size (URLs)")
plt.ylabel("Metric Score")
plt.title("Impact of Query Set Size on Detection Metrics")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Number_of_websites.png", dpi=300) 
plt.show()
     