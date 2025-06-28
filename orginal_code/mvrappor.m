clear all; clc;

% --- Parameters ---
num_bloombits = 6144;
num_hashes = 1;
prob_p = 0.3;
prob_q = 0.95;
prob_f = 0.4;
num_views = 55;
max_url_length = 100;
lambda_lasso = 1e-5;

% --- Load and trim dataset (first 1000 rows) ---
data = readtable('PhiUSIIL_Phishing_URL_Dataset.csv', 'ReadVariableNames', false);
urls = string(data{2:end, 2});
labels = double(data{2:end, end});
N = min(1000, length(urls));
websites = urls(1:N);
labels = labels(1:N);

% --- Step 1: PRR Encoding (preallocated) ---
prr_matrix = false(N, num_bloombits);
for i = 1:N
    s = pad_string(websites(i), max_url_length);
    bf = false(1, num_bloombits);
    for h = 1:num_hashes
        salt = strcat(num2str(h), '_cohort');
        idx = bloom_hash(s, salt, num_bloombits);
        bf(idx) = 1;
    end
    f_mask = rand(1, num_bloombits) < prob_f;
    uniform = rand(1, num_bloombits) < 0.5;
    prr = (bf & ~f_mask) | (uniform & f_mask);
    prr_matrix(i, :) = prr;
end

% --- Step 2: IRR Views (preallocated, fast) ---
irr_matrix = false(N * num_views, num_bloombits);
for v = 1:num_views
    % Vectorized bit generation
    p_bits = rand(N, num_bloombits) < prob_p;
    q_bits = rand(N, num_bloombits) < prob_q;
    prr_view = prr_matrix;
    irr = (p_bits & ~prr_view) | (q_bits & prr_view);
    irr_matrix((v-1)*N + 1:v*N, :) = irr;
end

% --- Step 3: Design Matrix for Candidates (preallocated) ---
candidates = unique(websites);
M = length(candidates);
A = false(M, num_bloombits);
for i = 1:M
    s = pad_string(candidates(i), max_url_length);
    for h = 1:num_hashes
        salt = strcat(num2str(h), '_cohort');
        idx = bloom_hash(s, salt, num_bloombits);
        A(i, idx) = 1;
    end
end

% --- Step 4: Frequency Reconstruction ---
y = mean(irr_matrix, 1)';
try
    [B, ~] = lasso(double(A'), y, 'Lambda', lambda_lasso);
    reconstructed_freq = max(B(:, 1), 0);
    fprintf('Used LASSO: %d non-zero coefficients\n', sum(reconstructed_freq > 0));
    if sum(reconstructed_freq > 0) == 0
        error('Fallback to lsqnonneg');
    end
catch
    reconstructed_freq = lsqnonneg(double(A'), y);
    fprintf('Used lsqnonneg: %d non-zero coefficients\n', sum(reconstructed_freq > 0));
end

% --- Step 5: Detection with Normalized Matching ---
threshold = 5e-07;
detected = find(reconstructed_freq > threshold);
recon_websites = candidates(detected);

% Normalize both sets for string comparison
recon_clean = normalize_urls(recon_websites);
true_malicious = normalize_urls(websites(labels == 0));
recon_labels = ismember(recon_clean, true_malicious);

fprintf('\nReconstructed Websites (label = 1 = phishing):\n');
for i = 1:length(recon_websites)
    fprintf('%s\tlabel=%d\n', recon_websites(i), recon_labels(i));
end

% --- Metrics ---
tp = sum(recon_labels);
fp = length(recon_labels) - tp;
fn = sum(~ismember(true_malicious, recon_clean));
precision = tp / (tp + fp + eps);
recall = tp / (tp + fn + eps);
f1 = 2 * precision * recall / (precision + recall + eps);
fprintf('\n# Detected: %d\n# True phishing: %d\n# Matched: %d\n', ...
        length(recon_websites), length(true_malicious), tp);
fprintf('Precision: %.2f | Recall: %.2f | F1-score: %.2f\n', precision, recall, f1);

% --- Top Results ---
[~, idx] = sort(reconstructed_freq, 'descend');
fprintf('\nTop Candidates by Estimated Frequency:\n');
for i = 1:min(10, length(idx))
    fprintf('%s: %.4f\n', candidates(idx(i)), reconstructed_freq(idx(i)));
end

% --- Helpers ---
function index = bloom_hash(input_string, salt, num_bits)
    key = [char(input_string), '_', salt];
    md = java.security.MessageDigest.getInstance('SHA-256');
    hash = md.digest(uint8(key));
    hash_num = typecast(hash(1:4), 'uint32');
    index = mod(double(hash_num), num_bits) + 1;
end

function padded = pad_string(s, maxlen)
    s = char(s);
    padded = s(1:min(end, maxlen));
    padded = [padded, repmat(' ', 1, maxlen - length(padded))];
end

function out = normalize_urls(urls)
    urls = lower(urls);
    urls = regexprep(urls, '^https?://', '');
    urls = regexprep(urls, '/$', '');
    out = urls;
end
