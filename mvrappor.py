import numpy as np
import pandas as pd
import hashlib
from sklearn.linear_model import Lasso
from scipy.optimize import lsq_linear

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
labels = data.iloc[1:, -1].astype(int).values
N = min(1000, len(urls))
websites = urls[:N]
labels = labels[:N]

# --- Helper functions ---
def bloom_hash(input_string, salt, num_bits):
    key = (input_string + "_" + salt).encode('utf-8')
    hash_digest = hashlib.sha256(key).digest()
    hash_val = int.from_bytes(hash_digest[:4], 'big')
    return hash_val % num_bits

def pad_string(s, maxlen):
    s = s[:maxlen].ljust(maxlen)
    return s

def normalize_urls(urls):
    urls = [u.lower().replace("https://", "").replace("http://", "").rstrip("/") for u in urls]
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
    prr = np.where(f_mask, uniform, bf)
    prr_matrix[i] = prr

# --- Step 2: IRR Views ---
irr_matrix = []
for v in range(num_views):
    print(f"View {v+1}")
    for i in range(N):
        prr = prr_matrix[i]
        p_bits = np.random.rand(num_bloombits) < prob_p
        q_bits = np.random.rand(num_bloombits) < prob_q
        irr = np.where(prr == 1, q_bits, p_bits)
        irr_matrix.append(irr)
irr_matrix = np.array(irr_matrix)

# --- Step 3: Design Matrix for Candidates ---
candidates = np.unique(websites)
M = len(candidates)
A = np.zeros((M, num_bloombits), dtype=int)
for i, s in enumerate(candidates):
    s = pad_string(s, max_url_length)
    for h in range(1, num_hashes + 1):
        salt = f"{h}_cohort"
        idx = bloom_hash(s, salt, num_bloombits)
        A[i, idx] = 1

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Compute y as usual
y = irr_matrix.mean(axis=0)

# Compute alpha correctly
n_samples = A.shape[0]
alpha = lambda_lasso / n_samples

# Setup Lasso pipeline with standardization
model = make_pipeline(
    StandardScaler(),
    Lasso(
        alpha=alpha,
        fit_intercept=True,
        positive=False,
        tol=1e-4,
        max_iter=2000
    )
)

try:
    model.fit(A.T, y)
    # Extract Lasso from pipeline to get coefficients
    lasso_coef = model.named_steps['lasso'].coef_
    reconstructed_freq = np.maximum(lasso_coef, 0)
    nonzero = np.sum(reconstructed_freq > 0)
    print(f"Used LASSO: {nonzero} non-zero coefficients")
    if nonzero == 0:
        raise Exception("Fallback to lsqnonneg")
except:
    res = lsq_linear(A.T, y, bounds=(0, np.inf))
    reconstructed_freq = res.x
    print(f"Used lsqnonneg: {np.sum(reconstructed_freq > 0)} non-zero coefficients")


# --- Step 5: Detection with Normalized Matching ---
threshold = 0.003
detected = np.where(reconstructed_freq > threshold)[0]
recon_websites = candidates[detected]

recon_clean = normalize_urls(recon_websites)
true_malicious = normalize_urls(websites[labels == 0])
recon_labels = [1 if r in true_malicious else 0 for r in recon_clean]

print("\nReconstructed Websites (label = 1 = phishing):")
for i, r in enumerate(recon_websites):
    print(f"{r}\tlabel={recon_labels[i]}")

# --- Metrics ---
tp = sum(recon_labels)
fp = len(recon_labels) - tp
fn = sum([1 for url in true_malicious if url not in recon_clean])
precision = tp / (tp + fp + 1e-10)
recall = tp / (tp + fn + 1e-10)
f1 = 2 * precision * recall / (precision + recall + 1e-10)
print(f"\n# Detected: {len(recon_websites)}\n# True phishing: {len(true_malicious)}\n# Matched: {tp}")
print(f"Precision: {precision:.2f} | Recall: {recall:.2f} | F1-score: {f1:.2f}")

# --- Top Results ---
top_idx = np.argsort(reconstructed_freq)[::-1]
print("\nTop Candidates by Estimated Frequency:")
for i in range(min(10, len(top_idx))):
    print(f"{candidates[top_idx[i]]}: {reconstructed_freq[top_idx[i]]:.4f}")