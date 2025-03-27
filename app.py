import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, render_template, send_file
import tldextract
import re
import pickle
import pandas as pd
import requests
from urllib.parse import urlparse
import io
import csv

# Model architecture (unchanged)
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(InvertedResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.activation1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.activation2 = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.activation2(out)
        return out

class ResidualPipeline(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_channels=128):
        super(ResidualPipeline, self).__init__()
        self.conv_block1 = ConvBlock(input_dim, hidden_channels)
        self.inverted_residual1 = InvertedResidualBlock(hidden_channels, hidden_channels)
        self.inverted_residual2 = InvertedResidualBlock(hidden_channels, hidden_channels)
        self.conv_block2 = ConvBlock(hidden_channels, hidden_channels)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.conv_block1(x)
        x = self.inverted_residual1(x)
        x = self.inverted_residual2(x)
        x = self.conv_block2(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x

# Flask app setup
app = Flask(__name__)

# Load scaler and label encoder
with open('/Users/gnaneshwarkandula/Downloads/ias/draft/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('/Users/gnaneshwarkandula/Downloads/ias/draft/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
    print("Label encoder classes:", label_encoder.classes_)

# Load the model
model_path = 'outputs/phishing_detection_model.pt'
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()
print("Model loaded:", model)

# Optimized session for HTTP requests
session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0 (compatible; FeatureExtractor/1.0)'})

# Feature extraction function (unchanged)
def extract_url_features(url):
    features = {}
    parsed = urlparse(url)
    hostname = parsed.netloc

    # Basic URL features
    features['f1_url_length'] = len(url)
    features['f2_hostname_length'] = len(hostname)
    features['f3_has_ip'] = 1 if re.match(r'^\d{1,3}(?:\.\d{1,3}){3}$', hostname) else 0

    # Special character counts
    special_chars = '. - @ ? & | = _ ~ % / * : , ; $'.split()
    for i, char in enumerate(special_chars, 4):
        features[f'f{i}_{char}_count'] = url.count(char)

    # Additional character-based features
    features['f19_space_or_%20'] = url.count('%20') + url.count(' ')
    features['f20_www_count'] = url.lower().count("www")
    features['f21_dotcom_count'] = url.lower().count(".com")
    features['f22_http_count'] = 0
    features['f23_double_slash_count'] = url.count("//")
    features['f24_https'] = 1 if parsed.scheme.lower() == 'https' else 0

    # Digit ratios
    digits_url = sum(c.isdigit() for c in url)
    features['f25_digit_ratio_url'] = digits_url / len(url) if len(url) > 0 else 0
    digits_host = sum(c.isdigit() for c in hostname)
    features['f26_digit_ratio_hostname'] = digits_host / len(hostname) if len(hostname) > 0 else 0

    # Domain-specific features
    features['f27_punycode'] = 1 if "xn--" in hostname else 0
    features['f28_port'] = 1 if parsed.port else 0
    ext = tldextract.extract(url)
    tld, subdomain = ext.suffix, ext.subdomain
    features['f29_tld_in_path'] = 1 if tld and tld in parsed.path else 0
    features['f30_tld_in_subdomain'] = 1 if tld and tld in subdomain else 0
    features['f31_abnormal_subdomain'] = 1 if subdomain and subdomain != "www" and re.match(r'^w+\d*$', subdomain) else 0
    features['f32_subdomain_count'] = len(subdomain.split('.')) if subdomain else 0
    features['f33_prefix_suffix'] = 1 if '-' in ext.domain else 0
    vowels = sum(1 for c in ext.domain.lower() if c in 'aeiou')
    features['f34_random_domain'] = 1 if ext.domain and (vowels / len(ext.domain)) < 0.3 else 0

    # URL type features
    shortening_services = {'bit.ly', 'tinyurl.com', 'goo.gl', 'ow.ly', 't.co', 'is.gd', 'buff.ly', 'adf.ly'}
    features['f35_shortening_service'] = 1 if hostname.lower() in shortening_services else 0
    features['f36_suspicious_extension'] = 1 if any(parsed.path.lower().endswith(ext) for ext in ['.exe', '.js', '.txt']) else 0

    # Redirection features
    try:
        resp = session.get(url, timeout=5, allow_redirects=True)
        history = resp.history
        features['f37_redirection_count'] = len(history)
        features['f38_external_redirections'] = sum(1 for r in history if urlparse(r.url).netloc.lower() != hostname.lower())
    except:
        features['f37_redirection_count'] = -1
        features['f38_external_redirections'] = -1

    # Word-based features
    words_url = re.findall(r'[A-Za-z0-9]+', url) or [""]
    features['f39_word_count_url'] = len(words_url)
    max_repeat = max((url.count(char) for char in set(url)), default=0)
    features['f40_max_char_repeat'] = max_repeat
    word_lengths = [len(w) for w in words_url]
    features['f41_shortest_word_length_url'] = min(word_lengths, default=0)
    features['f42_longest_word_length_url'] = max(word_lengths, default=0)

    words_host = re.findall(r'[A-Za-z0-9]+', hostname) or [""]
    features['f43_word_count_hostname'] = len(words_host)
    features['f44_longest_word_length_hostname'] = max((len(w) for w in words_host), default=0)

    words_path = re.findall(r'[A-Za-z0-9]+', parsed.path) or [""]
    features['f45_word_count_path'] = len(words_path)
    features['f46_longest_word_length_path'] = max((len(w) for w in words_path), default=0)

    features['f47_avg_word_length_url'] = sum(len(w) for w in words_url) / len(words_url) if words_url else 0
    features['f48_avg_word_length_hostname'] = sum(len(w) for w in words_host) / len(words_host) if words_host else 0
    features['f49_avg_word_length_path'] = sum(len(w) for w in words_path) / len(words_path) if words_path else 0

    # Phishing hints and brand detection
    sensitive_words = {"login", "signin", "verify", "account", "update", "secure", "confirm", "bank", "paypal", "ebay", "admin", "security", "password"}
    features['f50_phish_hints'] = sum(word in url.lower() for word in sensitive_words)
    brands = {
        "google":  ["goog", "g00gle"],
        "facebook": [ "fb", "faceb00k"],
        "amazon": ["amzn", "amaz0n"],
        "paypal": [ "paypl", "p4ypal"],
        "apple": ["appl", "app1e"],
        "microsoft": [ "msft", "micr0soft"],
        "ebay": ["eb4y", "e-bay"]
    }
    features['f51_brand_in_domain'] = 0
    features['f52_brand_in_subdomain'] = 0
    features['f53_brand_in_path'] = 0
    domain_lower, subdomain_lower, path_lower = ext.domain.lower(), subdomain.lower(), parsed.path.lower()
    for brand_variants in brands.values():
        if any(variant in domain_lower for variant in brand_variants):
            features['f51_brand_in_domain'] = 1
        if any(variant in subdomain_lower for variant in brand_variants):
            features['f52_brand_in_subdomain'] = 1
        if any(variant in path_lower for variant in brand_variants):
            features['f53_brand_in_path'] = 1

    # TLD and statistical features
    features['f54_suspicious_tld'] = 1 if tld.lower() in {"tk", "ml", "ga", "cf", "gq"} else 0
    features['f55_statistical_report'] = 0
    features['f56_statistical_report'] = 0

    return features

# Prepare features for the model
def prepare_features(url):
    features_dict = extract_url_features(url)
    feature_keys = [
        'f1_url_length', 'f2_hostname_length', 'f3_has_ip', 'f4_._count', 'f5_-_count', 'f6_@_count',
        'f7_?_count', 'f8_&_count', 'f9_|_count', 'f10_=_count', 'f11_~_count', 'f12_%_count',
        'f13_/_count', 'f14_*_count', 'f15_:_count', 'f16_,_count', 'f17_;_count', 'f18_$_count',
        'f19_space_or_%20', 'f20_www_count', 'f21_dotcom_count', 'f22_http_count', 'f23_double_slash_count',
        'f24_https', 'f25_digit_ratio_url', 'f26_digit_ratio_hostname', 'f27_punycode', 'f28_port',
        'f29_tld_in_path', 'f30_tld_in_subdomain', 'f31_abnormal_subdomain', 'f32_subdomain_count',
        'f33_prefix_suffix', 'f34_random_domain', 'f35_shortening_service', 'f36_suspicious_extension',
        'f37_redirection_count', 'f38_external_redirections', 'f39_word_count_url', 'f40_max_char_repeat',
        'f41_shortest_word_length_url', 'f42_longest_word_length_url', 'f43_word_count_hostname',
        'f44_longest_word_length_hostname', 'f45_word_count_path', 'f46_longest_word_length_path',
        'f47_avg_word_length_url', 'f48_avg_word_length_hostname', 'f49_avg_word_length_path',
        'f50_phish_hints', 'f51_brand_in_domain', 'f52_brand_in_subdomain', 'f53_brand_in_path',
        'f54_suspicious_tld', 'f55_statistical_report', 'f56_statistical_report'
    ]
    feature_values = [features_dict.get(key, 0) for key in feature_keys]
    print(f"Features for {url}:", feature_values)
    features_scaled = scaler.transform([feature_values])
    print(f"Features for {url}:", features_scaled)
    return torch.tensor(features_scaled, dtype=torch.float32)

# Global variable to store batch results
batch_results = {}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url']
    features = prepare_features(url)
    with torch.no_grad():
        output = model(features)
        probabilities = F.softmax(output, dim=1)
        print(f"Probabilities for {url}: {probabilities.numpy()}")
        _, predicted = torch.max(probabilities, 1)
        result = label_encoder.inverse_transform(predicted.numpy())[0]
    return render_template('index.html', url=url, result=result)

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    global batch_results
    file = request.files['file']
    if file.filename.endswith('.txt'):
        urls = file.read().decode('utf-8').splitlines()
    elif file.filename.endswith('.csv'):
        df = pd.read_csv(file)
        urls = df['url'].tolist()  # Assumes CSV has a 'url' column
    else:
        return "Unsupported file format. Please upload a .txt or .csv file.", 400

    batch_results = {}
    for url in urls:
        if url.strip():
            try:
                features = prepare_features(url.strip())
                with torch.no_grad():
                    output = model(features)
                    probabilities = F.softmax(output, dim=1)
                    print(f"Probabilities for {url}: {probabilities.numpy()}")
                    _, predicted = torch.max(probabilities, 1)
                    result = label_encoder.inverse_transform(predicted.numpy())[0]
                    batch_results[url] = result
            except Exception as e:
                batch_results[url] = f"Error: {str(e)}"
    
    return render_template('index.html', batch_results=batch_results)

@app.route('/download_results')
def download_results():
    global batch_results
    if not batch_results:
        return "No batch results available to download.", 400

    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['URL', 'Status'])
    for url, result in batch_results.items():
        writer.writerow([url, result])

    # Prepare file for download
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='phishing_detection_results.csv'
    )

if __name__ == '__main__':
    app.run(debug=True)