import numpy as np
import pandas as pd
import hashlib
from sklearn.linear_model import Lasso
from scipy.optimize import lsq_linear
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# --- Parameters ---
num_bloombits = 2048
num_hashes = 5
num_views = 1
max_url_length = 100
lambda_lasso = 0.0001
threshold = 0.00003
num_trials = 10

# --- Load Dataset ---
data = pd.read_csv('PhiUSIIL_Phishing_URL_Dataset.csv', header=None)
urls = data.iloc[1:, 1].astype(str).values
labels = data.iloc[1:, -1].astype(int).values
N = min(1000, len(urls))
websites = urls[:N]
labels = labels[:N]

# --- Utility Functions ---
def bloom_hash(input_string, salt, num_bits):
    key = (input_string + "_" + salt).encode('utf-8')
    hash_digest = hashlib.sha256(key).digest()
    hash_val = int.from_bytes(hash_digest[:4], 'big')
    return hash_val % num_bits

def pad_string(s, maxlen):
    return s[:maxlen].ljust(maxlen)

def normalize_urls(urls):
    return [u.lower().replace("https://", "").replace("http://", "").rstrip("/") for u in urls]

# --- Parameter Sweep Setup ---
epsilon_results = []

privacy_params = [
    (0.4, 0.6, 0.8),
    (0.45, 0.65, 0.7),
    (0.5, 0.75, 0.5),
    (0.55, 0.8, 0.4),
    (0.6, 0.9, 0.3),
    (0.7, 0.95, 0.2),
]

for prob_p, prob_q, prob_f in privacy_params:
    epsilons = []
    precisions, recalls, f1s = [], [], []

    # Compute ε₁
    q_star = 0.5 * prob_f * (prob_p + prob_q) + (1 - prob_f) * prob_q
    p_star = 0.5 * prob_f * (prob_p + prob_q) + (1 - prob_f) * prob_p
    epsilon_1 = num_hashes * np.log((q_star * (1 - p_star)) / (p_star * (1 - q_star)))

    for trial in range(num_trials):
        # --- PRR Encoding ---
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

        # --- IRR Views ---
        irr_matrix = []
        for v in range(num_views):
            for i in range(N):
                prr = prr_matrix[i]
                p_bits = np.random.rand(num_bloombits) < prob_p
                q_bits = np.random.rand(num_bloombits) < prob_q
                irr = np.where(prr == 1, q_bits, p_bits)
                irr_matrix.append(irr)
        irr_matrix = np.array(irr_matrix)

        # --- Candidate Design Matrix ---
        candidates = np.unique(websites)
        M = len(candidates)
        A = np.zeros((M, num_bloombits), dtype=int)
        for i, s in enumerate(candidates):
            s = pad_string(s, max_url_length)
            for h in range(1, num_hashes + 1):
                salt = f"{h}_cohort"
                idx = bloom_hash(s, salt, num_bloombits)
                A[i, idx] = 1

        y = irr_matrix.mean(axis=0)
        alpha = lambda_lasso / A.shape[0]

        model = make_pipeline(
            StandardScaler(),
            Lasso(alpha=alpha, fit_intercept=True, positive=False, tol=1e-4, max_iter=2000)
        )

        try:
            model.fit(A.T, y)
            lasso_coef = model.named_steps['lasso'].coef_
            reconstructed_freq = np.maximum(lasso_coef, 0)
        except:
            res = lsq_linear(A.T, y, bounds=(0, np.inf))
            reconstructed_freq = res.x

        # --- Detection ---
        detected = np.where(reconstructed_freq > threshold)[0]
        recon_websites = candidates[detected]
        recon_clean = normalize_urls(recon_websites)
        true_malicious = normalize_urls(websites[labels == 0])
        recon_labels = [1 if r in true_malicious else 0 for r in recon_clean]

        tp = sum(recon_labels)
        fp = len(recon_labels) - tp
        fn = sum([1 for url in true_malicious if url not in recon_clean])
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        epsilons.append(epsilon_1)

    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1s)
    epsilon_results.append((np.mean(epsilons), avg_precision, avg_recall, avg_f1))

# --- Results ---
results_df = pd.DataFrame(epsilon_results, columns=['Epsilon_1', 'Avg_Precision', 'Avg_Recall', 'Avg_F1'])
print("\n=== Averaged Results over Trials ===")
print(results_df)

# Optional: Save to CSV
results_df.to_csv('epsilon_sweep_results.csv', index=False)
