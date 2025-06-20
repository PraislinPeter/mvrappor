import numpy as np
import pandas as pd
import hashlib
import re
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
lambda_lasso = 0.0001  # Alpha in sklearn's Lasso

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
    print("Please ensure the dataset file is in the same directory as the script.")
    exit() 

# --- Helper Functions ---
def bloom_hash(input_string, salt, num_bits):
    key = f"{input_string}_{salt}".encode('utf-8')
    md = hashlib.sha256()
    md.update(key)
    hash_bytes = md.digest()
    # Take the first 4 bytes for consistent integer conversion with MATLAB's typecast to uint32
    hash_num = int.from_bytes(hash_bytes[:4], 'big')
    index = (hash_num % num_bits)
    return index

def pad_string(s, maxlen):
    s = str(s)
    if len(s) > maxlen:
        return s[:maxlen]
    else:
        return s + ' ' * (maxlen - len(s))

def normalize_urls(urls):
    urls = np.array([str(url).lower() for url in urls])
    urls = np.array([re.sub(r'^https?://', '', url) for url in urls])
    urls = np.array([re.sub(r'/$', '', url) for url in urls])
    return urls

print(f"Loaded {N} URLs.")

# --- Step 1: PRR Encoding ---
print("Step 1: PRR Encoding...")
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

print("PRR Encoding complete.")

# --- Step 2: IRR Views ---
print("Step 2: Generating IRR Views...")
irr_list = [] 
for v in range(num_views):
    for i in range(N):
        prr = prr_matrix[i, :]
        p_bits = np.random.rand(num_bloombits) < prob_p
        q_bits = np.random.rand(num_bloombits) < prob_q
        irr = (p_bits & ~prr) | (q_bits & prr)
        irr_list.append(irr)
irr_matrix = np.array(irr_list)
print("IRR Views complete.")

# --- Step 3: Design Matrix for Candidates ---
print("Step 3: Creating Design Matrix for Candidates...")
candidates = np.unique(websites)
M = len(candidates)
A = np.zeros((M, num_bloombits), dtype=int)
for i in range(M):
    s = pad_string(candidates[i], max_url_length)
    for h in range(1, num_hashes + 1):
        salt = f"{h}_cohort"
        idx = bloom_hash(s, salt, num_bloombits)
        A[i, idx] = 1

print("Design Matrix complete.")

# --- Step 4: Frequency Reconstruction ---
print("Step 4: Performing Frequency Reconstruction...")
y = np.mean(irr_matrix, axis=0).reshape(-1, 1) 
reconstructed_freq = None
try:
    
    lasso_model = Lasso(alpha=lambda_lasso, fit_intercept=False, positive=True, max_iter=2000) 
    lasso_model.fit(A.T, y.ravel())
    reconstructed_freq = lasso_model.coef_
    reconstructed_freq[reconstructed_freq < 1e-9] = 0

    print(f'Used LASSO: {np.sum(reconstructed_freq > 0)} non-zero coefficients')
    if np.sum(reconstructed_freq > 0) == 0:
        raise Exception('Fallback to lsqnonneg')
except Exception as e:
    print(f"LASSO failed or had no non-zero coefficients ({e}). Falling back to lsqnonneg.")
    res = lsq_linear(A.T, y.ravel(), bounds=(0, np.inf), method='trf')
    reconstructed_freq = res.x
    print(f'Used lsqnonneg: {np.sum(reconstructed_freq > 0)} non-zero coefficients')

print("Frequency Reconstruction complete.")

# --- Step 5: Detection with Normalized Matching ---
print("\nStep 5: Detecting Phishing URLs...")
threshold = 0.003
detected_indices = np.where(reconstructed_freq > threshold)[0]
recon_websites = candidates[detected_indices]

# Normalize both sets for string comparison
recon_clean = normalize_urls(recon_websites)
true_malicious = normalize_urls(websites[labels == 1])

# Determine if reconstructed websites are actually malicious
recon_labels = np.isin(recon_clean, true_malicious)

print('\nReconstructed Websites (label = 1 = phishing):')
for i in range(len(recon_websites)):
    print(f'{recon_websites[i]}\tlabel={int(recon_labels[i])}')

# --- Metrics ---
tp = np.sum(recon_labels)
fp = len(recon_labels) - tp
fn = np.sum(~np.isin(true_malicious, recon_clean)) # Malicious not found in reconstructed

# Add a small epsilon to avoid division by zero
eps = np.finfo(float).eps

precision = tp / (tp + fp + eps)
recall = tp / (tp + fn + eps)
f1 = 2 * precision * recall / (precision + recall + eps)

print(f'\n# Detected: {len(recon_websites)}')
print(f'# True phishing: {len(true_malicious)}')
print(f'# Matched (True Positives): {tp}')
print(f'Precision: {precision:.2f} | Recall: {recall:.2f} | F1-score: {f1:.2f}')

# --- Top Results ---
# Sort by reconstructed frequency in descending order
sorted_indices = np.argsort(reconstructed_freq)[::-1]
print('\nTop Candidates by Estimated Frequency:')
for i in range(min(10, len(sorted_indices))):
    idx = sorted_indices[i]
    print(f'{candidates[idx]}: {reconstructed_freq[idx]:.4f}')