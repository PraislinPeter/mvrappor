import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hashlib
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from scipy.optimize import lsq_linear

# --- Configuration ---
NUM_BLOOMBITS = 1438
NUM_HASHES = 10
NUM_VIEWS = 1
NUM_TRIALS = 3
LAMBDA_LASSO = 0.01
THRESHOLD = 0.000001
PRINT_PER_TRIAL_STATS = True  # Set to True for detailed trial output

# --- Privacy Settings (p, q, f) ---
privacy_params = [
    (0.4, 0.6, 0.8),
    (0.45, 0.65, 0.7),
    (0.5, 0.75, 0.5),
    (0.55, 0.8, 0.4),
    (0.6, 0.9, 0.3),
    (0.7, 0.95, 0.2),
]

# --- Load Dataset ---
urls = pd.read_csv("PhiUSIIL_Phishing_URL_Dataset.csv")
urls['URL'] = urls['URL'].str.lower().str.replace("https://", "").str.replace("http://", "").str.rstrip("/")

# --- Hash Functions ---
def hash_domain(url, num_bloombits, num_hashes):
    bits = np.zeros(num_bloombits, dtype=bool)
    for i in range(num_hashes):
        salted_input = f"{i}:{url}".encode()
        hash_digest = hashlib.sha256(salted_input).hexdigest()
        index = int(hash_digest, 16) % num_bloombits
        bits[index] = True
    return bits

def create_candidate_set(urls, num_bloombits, num_hashes):
    return np.array([hash_domain(url, num_bloombits, num_hashes) for url in urls])

def prr_view(url, num_bloombits, num_hashes, f):
    bits = hash_domain(url, num_bloombits, num_hashes)
    keep_mask = np.random.rand(num_bloombits) < (1 - f)
    random_bits = np.random.rand(num_bloombits) < 0.5
    return np.where(keep_mask, bits, random_bits)

def irr_view(prr, p, q):
    rand = np.random.rand(len(prr))
    return np.where(prr, rand < q, rand < p)

# --- Evaluation ---
results = []

for prob_p, prob_q, prob_f in privacy_params:
    precisions, recalls, f1s, accuracies, candidate_sizes = [], [], [], [], []
    phishing_detection_flags = []

    q_star = 0.5 * prob_f * (prob_p + prob_q) + (1 - prob_f) * prob_q
    p_star = 0.5 * prob_f * (prob_p + prob_q) + (1 - prob_f) * prob_p
    epsilon_1 = NUM_HASHES * np.log((q_star * (1 - p_star)) / (p_star * (1 - q_star)))

    for trial in range(NUM_TRIALS):
        urls_mal = urls[urls['label'] == 0].sample(100, random_state=None)['URL'].values
        candidate_set = create_candidate_set(urls_mal, NUM_BLOOMBITS, NUM_HASHES)

        normal = urls[urls['label'] == 1].sample(10, random_state=None)['URL'].tolist()
        phishing = np.random.choice(urls_mal, 1, replace=False).tolist()
        query_set = normal + phishing

        prr_views = [prr_view(url, NUM_BLOOMBITS, NUM_HASHES, prob_f) for url in query_set]

        irr_matrix = []
        for _ in range(NUM_VIEWS):
            for prr in prr_views:
                irr_matrix.append(irr_view(prr, prob_p, prob_q))
        irr_matrix = np.array(irr_matrix)

        y = irr_matrix.mean(axis=0)
        alpha = LAMBDA_LASSO / candidate_set.shape[0]

        model = make_pipeline(
            StandardScaler(),
            Lasso(alpha=alpha, fit_intercept=True, positive=False, max_iter=2000)
        )

        try:
            model.fit(candidate_set.T, y)
            coef = model.named_steps['lasso'].coef_
            reconstructed_freq = np.maximum(coef, 0)
        except:
            res = lsq_linear(candidate_set.T, y, bounds=(0, np.inf))
            reconstructed_freq = res.x

        detection = np.where(reconstructed_freq > THRESHOLD)[0]
        urls_detected = urls_mal[detection]

        # Evaluate
        phishing_detected = phishing[0] in urls_detected
        phishing_detection_flags.append(int(phishing_detected))

        y_true = [1 if url in phishing else 0 for url in urls_mal]
        y_pred = [1 if url in urls_detected else 0 for url in urls_mal]

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = sum(y_true[i] == y_pred[i] for i in range(len(y_true))) / len(y_true)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        accuracies.append(accuracy)
        candidate_sizes.append(len(candidate_set))

        # Per-trial debug info
        if PRINT_PER_TRIAL_STATS:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            false_positives = [url for url in urls_detected if url not in phishing]
            in_candidate = sum(1 for url in urls_detected if url in urls_mal)

            print(f"\nTrial {trial+1}")
            print(f"  Phishing Detected: {phishing_detected}")
            print(f"  Detected URLs: {len(urls_detected)}")
            print(f"  In Candidate Set: {in_candidate} / {len(urls_mal)}")
            print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
            print(f"  Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Accuracy={accuracy:.4f}")

    # Aggregate results for this (p, q, f) set
    results.append({
        "Epsilon": round(epsilon_1, 4),
        "Avg_Precision": round(np.mean(precisions), 4),
        "Avg_Recall": round(np.mean(recalls), 4),
        "Avg_F1": round(np.mean(f1s), 4),
        "Avg_Accuracy": round(np.mean(accuracies), 4),
        "Phishing_Detection_Rate": round(np.mean(phishing_detection_flags), 2),
        "Candidate_Set_Size": int(np.mean(candidate_sizes))
    })

# --- Results Table ---
results_df = pd.DataFrame(results)
print("\n=== Summary Results by Epsilon ===")
print(results_df)

# Optional: Save results
results_df.to_csv("epsilon_summary_metrics.csv", index=False)

# --- Plot ---
plt.figure(figsize=(10, 6))
plt.plot(results_df["Epsilon"], results_df["Avg_Accuracy"], label='Accuracy', marker='o')
plt.plot(results_df["Epsilon"], results_df["Avg_Precision"], label='Precision', marker='s')
plt.plot(results_df["Epsilon"], results_df["Avg_Recall"], label='Recall', marker='^')
plt.plot(results_df["Epsilon"], results_df["Avg_F1"], label='F1 Score', marker='d')
plt.plot(results_df["Epsilon"], results_df["Phishing_Detection_Rate"], label='Phishing Detection Rate', marker='x', linestyle='--')
plt.xlabel("Epsilon")
plt.ylabel("Metric Score")
plt.title("Detection Metrics vs Privacy Budget (ε)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("epsilon_vs_metrics.png", dpi=300)
plt.show()
