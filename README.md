
# Twitter Sentiment Analysis using Machine Learning

This project performs **Sentiment Analysis on Twitter data** using **Natural Language Processing (NLP)** and **Machine Learning**.  
It classifies tweets as **Positive** or **Negative** using the **Sentiment140 dataset** and a **Logistic Regression** model.

---

## 📌 Project Overview

- Dataset: **Sentiment140**
- Total Tweets: **1.6 Million**
- Task: Binary Classification  
  - `0` → Negative Tweet  
  - `1` → Positive Tweet
- Feature Extraction: **TF-IDF**
- Model Used: **Logistic Regression**
- Accuracy Achieved: **~77.6%**

---

## 🛠️ Libraries Used

- `numpy` – numerical operations
- `pandas` – data handling
- `re` – text cleaning using regular expressions
- `nltk` – stopwords and stemming
- `scikit-learn` – machine learning models and evaluation
- `pickle` – saving and loading trained model
- `kaggle` – dataset download

---

## 📂 Dataset Details

- **Name:** Sentiment140
- **Source:** Kaggle
- **Labels:**
  - `0` → Negative sentiment
  - `4` → Positive sentiment (converted to `1`)

---

## 📥 Dataset Download using Kaggle API

### Step 1: Install Kaggle
```
pip install kaggle
```

### Step 2: Upload Kaggle API Token
- Go to Kaggle → Account → Create API Token
- Upload `kaggle.json` in Google Colab

```
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Step 3: Download Dataset
```
kaggle datasets download -d kazanova/sentiment140
```

### Step 4: Extract Dataset
```python
from zipfile import ZipFile

with ZipFile('sentiment140.zip', 'r') as zip:
    zip.extractall()
```

---

## 🔄 Project Workflow (Step by Step)

### 1️⃣ Import Required Libraries
All necessary Python, NLP, and ML libraries are imported.

---

### 2️⃣ Load the Dataset
The CSV file is loaded into a Pandas DataFrame and column names are assigned.

```python
column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
```

---

### 3️⃣ Data Exploration
- Checked number of rows and columns
- Verified there are no missing values
- Checked sentiment label distribution

---

### 4️⃣ Label Conversion
Original labels:
- `0` → Negative
- `4` → Positive

Converted to:
```
4 → 1
```

---

### 5️⃣ Text Preprocessing (Stemming)

Each tweet is processed as follows:
- Remove special characters
- Convert to lowercase
- Remove stopwords
- Apply stemming using Porter Stemmer

```python
def stemming(content):
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower().split()
    content = [stemmer.stem(word) for word in content if word not in stopwords.words('english')]
    return ' '.join(content)
```

---

### 6️⃣ Feature and Label Separation

```python
X = tdata['stemmed_content'].values
Y = tdata['target'].values
```

---

### 7️⃣ Train-Test Split

- 80% Training Data
- 20% Testing Data

```python
train_test_split(X, Y, stratify=Y, test_size=0.2)
```

---

### 8️⃣ Text Vectorization using TF-IDF

```python
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
```

---

### 9️⃣ Model Training (Logistic Regression)

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)
```

---

## 💾 Saving the Trained Model

```python
import pickle
pickle.dump(model, open("trained_model.pkl", "wb"))
```

---

## 🔁 Loading and Using Saved Model

```python
loaded_model = pickle.load(open("trained_model.pkl", "rb"))
```

---

## 📊 Final Results

| Metric | Score |
|------|------|
| Training Accuracy | ~79.8% |
| Testing Accuracy | ~77.6% |

---

⭐ If you like this project, give the repository a star!
