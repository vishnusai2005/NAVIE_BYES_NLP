# 📊 Naive Bayes Text Classifier — From Bayes' Theorem to Real-World NLP Pipelines

**A from-first-principles implementation of Multinomial Naive Bayes for text classification, evaluated across three progressively harder problems — toy spam detection, a benchmark 20-class news topic classifier, and social-media sentiment — with a fully transparent, self-audited engineering trail.**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Naive%20Bayes-F7931E?logo=scikitlearn&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-Text%20Preprocessing-green)
![Pandas](https://img.shields.io/badge/pandas-Data%20Handling-150458?logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Audited%20%26%20Documented-blueviolet)

---

## 🧭 Table of Contents

1. [Why This Project](#-why-this-project)
2. [What's Inside the Notebook](#-whats-inside-the-notebook)
3. [Datasets Used](#-datasets-used)
4. [Results & Visualizations](#-results--visualizations)
5. [Honest Engineering Audit](#-honest-engineering-audit)
6. [What I'd Do Differently](#-what-id-do-differently)
7. [Tech Stack](#-tech-stack)
8. [How to Run](#-how-to-run)
9. [About the Author](#-about-the-author)

---

## 🎯 Why This Project

Naive Bayes is the classical baseline behind spam filters, content moderation systems, and topic routers — it's fast, interpretable, and still competitive on high-dimensional sparse text data. This project builds that baseline from the ground up: starting with the raw probability math, moving through feature engineering (Bag-of-Words vs. TF-IDF, n-grams, lemmatization), and finishing with a genuine multi-class benchmark (20 Newsgroups) instead of stopping at a toy example.

Just as importantly, this README documents the notebook **as it actually ran** — including a dataset that failed to load, a preprocessing bug that quietly broke tokenization, and a mislabeled plot — rather than presenting a polished narrative that hides them. Catching and explaining these issues is, in my view, as valuable a signal to a reviewer as the accuracy numbers themselves.

---

## 🧩 What's Inside the Notebook

The notebook is organized as five progressively more realistic experiments:

| # | Section | Purpose |
|---|---------|---------|
| 1 | **Naive Bayes by hand** | Manually computes `P(spam \| words)` vs. `P(ham \| words)` from illustrative word-probability tables to build intuition for the conditional-independence assumption behind the model. |
| 2 | **Toy email dataset** | A 6-message spam/ham set used to walk through tokenization → lemmatization → `TfidfVectorizer` (uni+bigrams) → `MultinomialNB`, end to end. |
| 3 | **SMS spam classification** | Attempts to download the UCI SMS Spam Collection and compares `CountVectorizer` vs. `TfidfVectorizer`, with and without a custom cleaning step. *(See audit note below — this section did not run on the intended dataset.)* |
| 4 | **20 Newsgroups (real benchmark)** | The flagship experiment: **18,846 real documents across 20 topic categories**, comparing three feature pipelines — raw CountVectorizer, raw TF-IDF, and preprocessed TF-IDF (lowercasing, digit-stripping, lemmatization). |
| 5 | **Twitter sentiment (mini-experiment)** | A 30-tweet, hand-labeled, 3-class (positive/negative/neutral) sentiment set — including emojis and hashtags — used to stress-test Naive Bayes on small, noisy, informal text. |

---

## 📚 Datasets Used

| Dataset | Source | Scale | Status in this notebook |
|---|---|---|---|
| Toy email set | Hand-written | 6 messages | ✅ Ran as intended (illustrative only) |
| UCI SMS Spam Collection | `archive.ics.uci.edu` | ~5,574 SMS messages | ⚠️ Download step failed — see audit |
| 20 Newsgroups | `sklearn.datasets.fetch_20newsgroups` | **18,846 documents, 20 classes** | ✅ Ran successfully — real benchmark |
| Hand-labeled tweets | Custom | 30 tweets, 3 classes | ✅ Ran as intended (small-sample demo) |

---

## 📈 Results & Visualizations

### Manual probability walkthrough
The hand-computed example (`p_spam = 0.3`, illustrative word-probability tables) correctly reproduces the direction of a real spam classifier's reasoning:
```
Probability of SMS being spam: 0.0000285120
Probability of SMS being ham:  0.0000000044
Prediction: spam
```
This is a worked example with hand-picked numbers, not a value fitted from data — it's included to demonstrate the math, not as a performance result.

### 20 Newsgroups — CountVectorizer vs. TF-IDF vs. Preprocessed TF-IDF
![20 Newsgroups ablation](assets/20newsgroups_ablation_heatmaps.png)

This is the project's real, large-scale test: three identical `MultinomialNB` pipelines differing only in their feature representation, evaluated on the same 18,846-document, 20-class corpus (class sizes ranging from 628 to 999 documents — a mild, natural imbalance). **Note:** the axis labels on this specific figure should be read with the caveat described in the audit section below — the five class *names* shown and the five sets of *values* plotted do not reliably correspond to each other due to an indexing bug in the plotting helper. The comparison of feature-engineering strategies is still valid; the specific per-class breakdown in this exact figure is not.

### Twitter sentiment — a genuine limitation, shown rather than hidden
![Tweet sentiment confusion matrix](assets/tweets_sentiment_confusion_matrix.png)

On 6 held-out tweets from a 30-tweet hand-labeled set, the model scored **17% accuracy** — barely better than chance, and worse than the "predict the majority class" baseline. This isn't a flattering number, but it's an honest and expected one: 24 training examples across 3 classes, with emojis and hashtags as most of the signal, is far below what Naive Bayes needs to generalize. It's included here as evidence of understanding *why* a model failed, not just running it.

---

## 🔍 Honest Engineering Audit

Following an audit style I apply consistently across my portfolio, here are the real issues found while tracing through this notebook's execution — with root causes, not just symptoms.

1. **The SMS Spam Collection never actually loaded.** The download cell (`!wget ... && !unzip ...`) hit an interactive "replace file? [y/n]" prompt mid-execution and raised an `OSError` before `pd.read_csv(...)` ran. Every subsequent "SMS spam classifier" comparison — CountVectorizer vs. TF-IDF, the 3-panel ablation heatmap — silently kept using the leftover 6-row toy DataFrame from Section 2 instead of the real ~5,574-message corpus. This is confirmed directly in the notebook's own output: `df.shape` returns `(6, 3)` right after the failed download, and every "accuracy" reported in that section (100%, 50%) is computed on a 1–2-row test split — not a meaningful evaluation.

2. **A silent whitespace bug broke tokenization in both `clean_text()` functions** (SMS and 20 Newsgroups versions). The final line returns `''.join(tokens)` instead of `' '.join(tokens)`, concatenating every cleaned word into one run-on string. This is visible directly in the notebook's own printed output — `"Free money now!!!"` becomes `"freemoney"`, `"Hi Bob, how about a game tomorrow?"` becomes `"hibobgametomorrow"`. Since `CountVectorizer`/`TfidfVectorizer` split on word boundaries, this collapses each document into effectively a single giant token, which is the most likely explanation for why the "preprocessed" pipeline underperforms the raw-text baselines in both ablation studies.

3. **A no-op regex in the 20 Newsgroups cleaner.** `re.sub(r'[^\W\S]', '', text)` was almost certainly meant to strip punctuation (`[^\w\s]`), but `[^\W\S]` matches *zero* characters by construction (a character cannot simultaneously fail both the "non-word" and "non-whitespace" tests), so this line is dead code — verified independently with a regex test. Punctuation is never actually removed in this step.

4. **The 20 Newsgroups confusion-matrix figure has a label/index mismatch.** The plotting helper filters the class list down to the top-5-by-frequency classes *before* computing index positions (`idx = [i for i, l in enumerate(labs)]`, where `labs` is already the filtered list) — so `idx` always evaluates to `[0, 1, 2, 3, 4]`. Those indices are then used to slice the *original* 20×20 confusion matrix, which pulls values for whichever five classes sort first alphabetically overall (`alt.atheism`, `comp.graphics`, `comp.os.ms-windows.misc`, `comp.sys.ibm.pc.hardware`, `comp.sys.mac.hardware`), while the chart's axis ticks are labeled with the intended top-5-by-frequency classes (`rec.motorcycles`, `rec.sport.baseball`, `rec.sport.hockey`, `sci.crypt`, `soc.religion.christian`). I verified this by reproducing the exact indexing logic on a synthetic 20-class label set — the figure's labels and values genuinely belong to two different sets of classes.

5. **A loose URL-stripping regex.** `re.sub(r'http\S+|www\s+|https\S+', '', text)` — the `https\S+` branch is redundant (`http\S+` already matches any string starting with "http", including "https"), and `www\s+` requires a whitespace character immediately after "www" rather than matching the link itself, so bare `www.`-prefixed URLs (with no `http://`) are not actually removed.

6. **Naming**: the project filename, and one chart title, consistently spell it "Naive **Byes**." Cosmetic, but worth a pass before publishing.

None of these change the intent or the underlying understanding of Naive Bayes demonstrated in the notebook — but they do change which specific numbers in it should be trusted, which is exactly why they're documented here rather than smoothed over.

---

## 🚀 What I'd Do Differently

- Make the dataset download non-interactive (`unzip -o` or check `os.path.exists()` first) and add an assertion on `df.shape` immediately after loading, so a failed download fails loudly instead of silently falling through to stale data.
- Fix the `join()` calls (`' '.join(tokens)`) and re-run both ablations to get a real read on whether preprocessing helps or hurts.
- Fix the punctuation regex and the confusion-matrix indexing bug, and re-generate the 20 Newsgroups figure with correct labels.
- Grow the tweet sentiment set well beyond 30 examples before drawing any conclusions from it.
- Wrap the repeated "vectorize → fit → predict → evaluate" pattern into a single reusable function instead of copy-pasting it per experiment.
- Add `requirements.txt`, a fixed `random_state` audit, and a short `predict(text)` helper so the trained model is actually usable outside the notebook.

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| ML | scikit-learn (`MultinomialNB`, `CountVectorizer`, `TfidfVectorizer`, `train_test_split`) |
| NLP | NLTK (tokenization, `WordNetLemmatizer`, stopwords) |
| Data handling | pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Environment | Jupyter Notebook |

---

## ▶️ How to Run

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install numpy pandas scikit-learn nltk matplotlib seaborn

python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4'); nltk.download('stopwords')"

jupyter notebook NLP_NAIVE_BAYES_CLASSIFIER.ipynb
```

> **Note:** the SMS Spam Collection download cell uses an interactive `unzip` prompt that can hang in non-interactive environments (see audit item 1). Replace it with `unzip -o smsspamcollection.zip` before running end to end.

---

## 👤 About the Author

**Vishnusai Vydhyam**
Final-year B.Tech CSE (AI & ML) student, Mohan Babu University, Tirupati
Actively seeking Machine Learning, Data Science, and AI Engineer internship/fresher roles.

- GitHub: [@vishnusai2005](https://github.com/vishnusai2005)
- LinkedIn: [vishnusai-vydhyam](https://www.linkedin.com/in/vishnusai-vydhyam)
- Hugging Face: [v2005](https://huggingface.co/v2005)
- X (Twitter): [@VishnusaiSaii](https://twitter.com/VishnusaiSaii)

*This README follows an honest-audit format applied consistently across my project portfolio — documenting what worked, what didn't, and why, alongside the results.*
