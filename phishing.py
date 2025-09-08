# phishing_detector_advanced.py
# pip install scikit-learn pandas tldextract python-whois xgboost

import re
import tldextract
import pandas as pd
import whois
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# --- Feature extraction ---
suspicious_words = [
    'verify', 'account', 'password', 'login', 'click', 'urgent',
    'update', 'confirm', 'bank', 'payment', 'security', 'suspend'
]

def extract_urls(text):
    return re.findall(r'(https?://[^\s]+)', text)

def domain_is_ip(url):
    m = re.match(r'https?://([^/:]+)', url)
    if not m: return False
    host = m.group(1)
    return re.match(r'^\d{1,3}(\.\d{1,3}){3}$', host) is not None

def domain_age_days(domain):
    try:
        w = whois.whois(domain)
        if isinstance(w.creation_date, list):
            creation_date = w.creation_date[0]
        else:
            creation_date = w.creation_date
        if creation_date:
            return (datetime.now() - creation_date).days
    except:
        return -1
    return -1

def extract_features_from_email(subject, body, from_addr):
    text = (subject or "") + " " + (body or "")
    urls = extract_urls(text)
    num_urls = len(urls)
    num_ip_urls = sum(1 for u in urls if domain_is_ip(u))
    url_lengths = [len(u) for u in urls] if urls else [0]
    avg_url_length = sum(url_lengths) / len(url_lengths)
    suspicious_word_count = sum(1 for w in suspicious_words if w in text.lower())
    num_exclamations = text.count("!")
    num_uppercase = sum(1 for w in text.split() if w.isupper())
    
    # Domain age feature
    age = -1
    if urls:
        ext = tldextract.extract(urls[0])
        domain = f"{ext.domain}.{ext.suffix}"
        age = domain_age_days(domain)
    
    return [
        num_urls,
        num_ip_urls,
        avg_url_length,
        suspicious_word_count,
        num_exclamations,
        num_uppercase,
        age
    ]

# --- Synthetic dataset (demo) ---
data = [
    ("Verify your account", "Click http://198.51.100.5/login", "support@bank.com", 1),
    ("Invoice April", "Attached invoice. Thanks", "accounts@trustedco.com", 0),
    ("URGENT: confirm payment", "Click https://secure-pay.xyz/confirm now", "billing@payments.com", 1),
    ("Meeting notes", "Here are notes https://company.share/doc", "colleague@company.com", 0),
    ("Password reset", "Reset: http://malicious.tk/reset", "no-reply@service.com", 1),
    ("Your order shipped", "Track at https://shop.example.com/track", "orders@shop.example.com", 0)
]

features = []
labels = []
texts = []

for subj, body, frm, label in data:
    f = extract_features_from_email(subj, body, frm)
    features.append(f)
    labels.append(label)
    texts.append((subj or "") + " " + (body or ""))

X_num = pd.DataFrame(features, columns=[
    'num_urls','num_ip_urls','avg_url_length','suspicious_word_count',
    'num_exclamations','num_uppercase','domain_age_days'
])

# --- Text features (TF-IDF) ---
vectorizer = TfidfVectorizer(max_features=50)
X_text = vectorizer.fit_transform(texts).toarray()
X = pd.concat([X_num, pd.DataFrame(X_text)], axis=1)
y = pd.Series(labels)

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

# --- Confusion matrix ---
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Legit","Phish"], yticklabels=["Legit","Phish"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# --- Save model ---
joblib.dump((clf, vectorizer), "phishing_detector.pkl")
print("✅ Model saved as phishing_detector.pkl")
