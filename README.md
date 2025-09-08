# Phishing Email Detection System 🚀

This project is a **machine learning-based phishing email detection system**.  
It analyzes the content of an email (subject, body, sender, and URLs) and predicts whether it is **Phishing (1)** or **Legitimate (0)**.

---

## 🔍 Features

- **Rule-based Features**
  - Number of URLs in the email
  - Number of IP-based URLs
  - Average URL length
  - Suspicious keywords count (e.g., *verify, password, urgent*)
  - Count of exclamation marks `!`
  - Count of ALL CAPS words (e.g., *URGENT*)
  - Domain age in days (via WHOIS lookup)

- **ML-based Features**
  - TF-IDF Vectorization of email subject and body (text mining)

- **Hybrid Model**
  - Combines rule-based features with text-based features for better accuracy.

---

## 🛠️ Tech Stack

- **Python 3.x**
- Libraries:
  - `scikit-learn`
  - `pandas`
  - `tldextract`
  - `python-whois`
  - `xgboost`
  - `seaborn`, `matplotlib`
  - `joblib`

---
