import numpy as np
import pandas as pd
import hashlib
from sklearn.linear_model import Lasso
from scipy.optimize import lsq_linear
import matplotlib.pyplot as plt

# --- Parameters ---
num_bloombits = 2048
num_hashes = 5
prob_p = 0.5
prob_q = 0.75
prob_f = 0.5
num_views = 15
max_url_length = 100
lambda_lasso = 0.0001

# --- Load and trim dataset (first 1000 rows) ---
data = pd.read_csv('PhiUSIIL_Phishing_URL_Dataset.csv', header=None)
urls = data.iloc[1:, 1].astype(str).values
labels = data.iloc[1:, -1].astype(float).values
N = min(1000, len(urls))
websites = urls[:N]
labels = labels[:N]

# --- Helper Functions ---
def bloom_hash(input_string, salt, num_bits):
    key = f"{input_string}_{salt}".encode('utf-8')
    hash_bytes = hashlib.sha256(key).digest()
    index = int.from_bytes(hash_bytes[:4], 'big') % num_bits
    return index

def pad_string(s, maxlen):
    s = str(s)
    return (s[:maxlen] + ' ' * maxlen)[:maxlen]

def normalize_urls(urls):
    urls = [u.lower() for u in urls]
    urls = [u.replace("http://", "").replace("https://", "") for u in urls]
    urls = [u.rstrip('/') for u in urls]
    return urls

# --- Step 1: PRR Encoding ---
prr_matrix = np.zeros((N, num_bloombits), dtype=int)
for i in range(N):
    s = pad_string(websites[i], max_url_length)
    bf = np.zeros(num_bloombits, dtype=int)
    for h in range(1, num_hashes + 1):
        salt = f"{h}_cohort"
        idx = bloom_hash(s, salt, num_bloombits)
        bf[idx] = 1
    f_mask = np.random.rand(num_bloombits) < prob_f
    uniform = np.random.rand(num_bloombits) < 0.5
    prr = (bf & ~f_mask) | (uniform & f_mask)
    prr_matrix[i, :] = prr

# --- Step 2: IRR Views ---
irr_matrix = []
for v in range(num_views):
    print(v + 1)
    for i in range(N):
        prr = prr_matrix[i, :]
        p_bits = np.random.rand(num_bloombits) < prob_p
        q_bits = np.random.rand(num_bloombits) < prob_q
        irr = (p_bits & ~prr) | (q_bits & prr)
        irr_matrix.append(irr)
irr_matrix = np.array(irr_matrix)

# --- Step 3: Design Matrix for Candidates ---
candidates, idx_unique = np.unique(websites, return_index=True)
M = len(candidates)
A = np.zeros((M, num_bloombits), dtype=int)
for i in range(M):
    s = pad_string(candidates[i], max_url_length)
    for h in range(1, num_hashes + 1):
        salt = f"{h}_cohort"
        idx = bloom_hash(s, salt, num_bloombits)
        A[i, idx] = 1

# --- Step 4: Frequency Reconstruction ---
y = irr_matrix.mean(axis=0)

try:
    lasso = Lasso(alpha=lambda_lasso, fit_intercept=False, max_iter=10000)
    lasso.fit(A.T, y)
    reconstructed_freq = np.maximum(lasso.coef_, 0)
    print(f"Used LASSO: {np.sum(reconstructed_freq > 0)} non-zero coefficients")
    # Normalize candidate URLs for comparison
    recon_clean_all = normalize_urls(candidates)
    true_malicious = normalize_urls(websites[labels == 0])

    # --- Debug print to inspect frequency separation ---
    print("\n--- Candidate Frequency Estimates ---")
    with open("freq_output.txt", "w") as f:
        f.write("\n--- Candidate Frequency Estimates ---\n")
        for i, freq in enumerate(reconstructed_freq):
            label = 1 if recon_clean_all[i] in true_malicious else 0
            f.write(f"{candidates[i]:60} | freq={freq:.4f} | label={label}\n")

    
    # --- Evaluate multiple thresholds ---
    thresholds = np.linspace(0.001, 0.1, 20)  # Try thresholds from 0.001 to 0.02
    metrics = []

    recon_clean_all = normalize_urls(candidates)
    true_malicious = normalize_urls(websites[labels == 0])

    for threshold in thresholds:
        detected = np.where(reconstructed_freq > threshold)[0]
        recon_websites = [recon_clean_all[i] for i in detected]

        recon_labels = [1 if r in true_malicious else 0 for r in recon_websites]
        tp = sum(recon_labels)
        fp = len(recon_labels) - tp
        fn = sum(1 for t in true_malicious if t not in recon_websites)
        precision = tp / (tp + fp + np.finfo(float).eps)
        recall = tp / (tp + fn + np.finfo(float).eps)
        f1 = 2 * precision * recall / (precision + recall + np.finfo(float).eps)

        metrics.append((threshold, len(recon_websites), tp, precision, recall, f1))

    # --- Display results ---
    print(f"{'Threshold':>10} | {'Detected':>9} | {'Matched':>7} | {'Precision':>9} | {'Recall':>6} | {'F1-score':>8}")
    print("-" * 60)
    for t, detected, matched, p, r, f1 in metrics:
        print(f"{t:10.4f} | {detected:9} | {matched:7} | {p:9.2f} | {r:6.2f} | {f1:8.2f}")

    # --- Plot Precision, Recall, F1 vs Threshold ---
    thresholds_plot = [m[0] for m in metrics]
    precision_plot = [m[3] for m in metrics]
    recall_plot = [m[4] for m in metrics]
    f1_plot = [m[5] for m in metrics]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds_plot, precision_plot, label='Precision', marker='o')
    plt.plot(thresholds_plot, recall_plot, label='Recall', marker='s')
    plt.plot(thresholds_plot, f1_plot, label='F1-score', marker='^')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1 vs. Threshold')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if np.sum(reconstructed_freq > 0) == 0:
        raise Exception("Fallback to lsqnonneg")
except:
    result = lsq_linear(A.T, y, bounds=(0, np.inf))
    reconstructed_freq = result.x
    print(f"Used lsqnonneg: {np.sum(reconstructed_freq > 0)} non-zero coefficients")

# --- Step 5: Detection with Normalized Matching ---
threshold = 0.003
detected = np.where(reconstructed_freq > threshold)[0]
recon_websites = candidates[detected]

recon_clean = normalize_urls(recon_websites)
true_malicious = normalize_urls(websites[labels == 0])
recon_labels = [1 if r in true_malicious else 0 for r in recon_clean]

print('\nReconstructed Websites (label = 1 = phishing):')
for site, label in zip(recon_websites, recon_labels):
    print(f"{site}\tlabel={label}")

# --- Metrics ---
tp = sum(recon_labels)
fp = len(recon_labels) - tp
fn = sum(1 for t in true_malicious if t not in recon_clean)
precision = tp / (tp + fp + np.finfo(float).eps)
recall = tp / (tp + fn + np.finfo(float).eps)
f1 = 2 * precision * recall / (precision + recall + np.finfo(float).eps)
print(f'\n# Detected: {len(recon_websites)}')
print(f'# True phishing: {len(true_malicious)}')
print(f'# Matched: {tp}')
print(f'Precision: {precision:.2f} | Recall: {recall:.2f} | F1-score: {f1:.2f}')

# --- Top Results ---
top_idx = np.argsort(-reconstructed_freq)
print('\nTop Candidates by Estimated Frequency:')
for i in range(min(10, len(top_idx))):
    print(f"{candidates[top_idx[i]]}: {reconstructed_freq[top_idx[i]]:.4f}")

