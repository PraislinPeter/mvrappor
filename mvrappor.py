import numpy as np
import pandas as pd
import hashlib
import re
from sklearn.linear_model import Lasso
from scipy.optimize import lsq_linear
from joblib import Parallel, delayed

# --- Parameters ---
num_bloombits = 2048
num_hashes = 5
prob_p = 0.5
prob_q = 0.75
prob_f = 0.5
max_url_length = 100
lambda_lasso = 0.0001
eps = np.finfo(float).eps
view_range = range(50, 501, 50)

# --- Load and trim dataset (first 1000 rows) ---
try:
    data = pd.read_csv('PhiUSIIL_Phishing_URL_Dataset.csv', header=None)
    urls = data.iloc[1:, 1].astype(str).values 
    labels = data.iloc[1:, -1].astype(float).values 
    N = min(1000, len(urls))
    websites = urls[:N]
    labels = labels[:N]
except FileNotFoundError:
    print("Error: 'PhiUSIIL_Phishing_URL_Dataset.csv' not found.")
    exit()

def bloom_hash(input_string, salt, num_bits):
    key = f"{input_string}_{salt}".encode('utf-8')
    md = hashlib.sha256()
    md.update(key)
    hash_bytes = md.digest()
    return int.from_bytes(hash_bytes[:4], 'big') % num_bits

def pad_string(s, maxlen):
    return str(s)[:maxlen].ljust(maxlen)

def normalize_urls(urls):
    urls = np.array([str(url).lower() for url in urls])
    urls = np.array([re.sub(r'^https?://', '', url) for url in urls])
    urls = np.array([re.sub(r'/$', '', url) for url in urls])
    return urls

# --- Step 1: PRR Encoding (once) ---
print("Generating PRR matrix...")
prr_matrix = np.zeros((N, num_bloombits), dtype=int)
for i in range(N):
    s = pad_string(websites[i], max_url_length)
    bf = np.zeros(num_bloombits, dtype=int)
    for h in range(1, num_hashes + 1):
        idx = bloom_hash(s, f"{h}_cohort", num_bloombits)
        bf[idx] = 1
    f_mask = np.random.rand(num_bloombits) < prob_f
    uniform = np.random.rand(num_bloombits) < 0.5
    prr = (bf & ~f_mask) | (uniform & f_mask)
    prr_matrix[i] = prr

# --- Step 2: Design Matrix (once) ---
print("Creating design matrix...")
candidates = np.unique(websites)
M = len(candidates)
A = np.zeros((M, num_bloombits), dtype=int)
for i in range(M):
    s = pad_string(candidates[i], max_url_length)
    for h in range(1, num_hashes + 1):
        idx = bloom_hash(s, f"{h}_cohort", num_bloombits)
        A[i, idx] = 1

# --- Experiment Function ---
def run_experiment(num_views):
    irr_all = []
    for _ in range(num_views):
        p_bits = np.random.rand(N, num_bloombits) < prob_p
        q_bits = np.random.rand(N, num_bloombits) < prob_q
        irr = (p_bits & ~prr_matrix) | (q_bits & prr_matrix)
        irr_all.append(irr)
    irr_matrix = np.vstack(irr_all)

    y = np.mean(irr_matrix, axis=0).reshape(-1, 1)
    try:
        model = Lasso(alpha=lambda_lasso, fit_intercept=False, positive=True, max_iter=2000)
        model.fit(A.T, y.ravel())
        freq = model.coef_
        freq[freq < 1e-9] = 0
        method = "LASSO"
        if np.sum(freq > 0) == 0:
            raise Exception()
    except:
        res = lsq_linear(A.T, y.ravel(), bounds=(0, np.inf), method='trf')
        freq = res.x
        method = "lsqnonneg"

    threshold = 0.003
    detected_indices = np.where(freq > threshold)[0]
    recon_urls = candidates[detected_indices]

    recon_clean = normalize_urls(recon_urls)
    true_malicious = normalize_urls(websites[labels == 1])
    recon_labels = np.isin(recon_clean, true_malicious)

    tp = np.sum(recon_labels)
    fp = len(recon_labels) - tp
    fn = np.sum(~np.isin(true_malicious, recon_clean))

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    print(f"\nViews: {num_views}")
    print(f"# Detected: {len(recon_labels)}")
    print(f"# True phishing: {len(true_malicious)}")
    print(f"# Matched (True Positives): {tp}")
    print(f"Precision: {precision:.2f} | Recall: {recall:.2f} | F1-score: {f1:.2f} | Method: {method}")

    print("\nReconstructed Websites:")
    for url, is_phishing in zip(recon_clean, recon_labels):
        label = "phishing ✅" if is_phishing else "benign ❌"
        print(f"- {url} [{label}]")

    return (num_views, precision, recall, f1, method)


# --- Run All Experiments in Parallel ---
print("\nRunning experiments in parallel...")
results = Parallel(n_jobs=-1, backend='loky')(delayed(run_experiment)(v) for v in view_range)

# --- Results Summary ---
summary_df = pd.DataFrame(results, columns=["Views", "Precision", "Recall", "F1", "Method"])
print("\n--- Summary of Results ---")
print(summary_df.to_string(index=False))
best_idx = summary_df['F1'].idxmax()
best_row = summary_df.loc[best_idx]

# --- Find Best View Counts for Each Metric ---
best_f1_idx = summary_df['F1'].idxmax()
best_precision_idx = summary_df['Precision'].idxmax()
best_recall_idx = summary_df['Recall'].idxmax()

best_f1 = summary_df.loc[best_f1_idx]
best_precision = summary_df.loc[best_precision_idx]
best_recall = summary_df.loc[best_recall_idx]

print("\n🏆 Best Results by Metric:")
print(f"Best F1-score     : {best_f1['F1']:.4f} at {int(best_f1['Views'])} views")
print(f"Best Precision    : {best_precision['Precision']:.4f} at {int(best_precision['Views'])} views")
print(f"Best Recall       : {best_recall['Recall']:.4f} at {int(best_recall['Views'])} views")


import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(10, 6))

plt.plot(summary_df['Views'], summary_df['Precision'], label='Precision', marker='o', markersize=6, linewidth=1.5)
plt.plot(summary_df['Views'], summary_df['Recall'], label='Recall', marker='s', markersize=6, linewidth=1.5)
plt.plot(summary_df['Views'], summary_df['F1'], label='F1-score', marker='^', markersize=6, linewidth=1.5)

plt.title('Phishing Detection Metrics vs Number of Views (Zoomed In)')
plt.xlabel('Number of Views')
plt.ylabel('Score')

plt.ylim(0.65, 0.9)
plt.yticks(np.linspace(0.65, 0.9, 10))  # ← 10 evenly spaced ticks
plt.xticks(summary_df['Views'])         # ← Show all view counts on x-axis

plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("phishing_metrics_zoomed.png", dpi=300)
plt.show()





