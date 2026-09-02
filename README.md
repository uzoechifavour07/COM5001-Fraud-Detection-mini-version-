# Ethereum Fraud Detection

A machine learning classifier that identifies fraudulent Ethereum wallet addresses from on-chain transaction behaviour alone — no identity data, no off-chain signals.

Blockchain addresses are pseudonymous, so traditional identity-based fraud checks don't apply. What is available is behaviour: how often an address transacts, how much value moves through it, how many distinct counterparties it deals with. This project tests whether those patterns are enough to separate wallets linked to Ponzi schemes, phishing and rug pulls from legitimate ones.

**Research question:** can machine learning models trained on on-chain transaction features accurately detect fraudulent Ethereum wallet addresses, and which behavioural features are most predictive?

## Dataset

[Ethereum Fraud Detection Dataset](https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset) (Kaggle, 2021).

| | |
|---|---|
| Records | 9,841 wallet addresses |
| Features | 45 |
| Target | `FLAG` — 1 fraudulent, 0 legitimate |
| Class balance | 7,662 legitimate / 2,179 fraudulent (22.1% fraud) |
| Format | CSV |

Features are derived from on-chain transaction metadata. The ones that matter most conceptually:

- `Avg min between sent tnx` — timing regularity, which separates automated contract activity from human behaviour
- `total Ether sent` / `total Ether received` — cumulative value flow
- `Unique Sent To Addresses` / `Unique Received From Addresses` — counterparty diversity, a strong signal for phishing and distribution patterns
- `Number of Created Contracts` — programmatic activity
- `ERC20 total Ether sent` — token-level volume

At 22% fraud the classes are imbalanced but not severely so. It still rules out accuracy as a headline metric — a model predicting "legitimate" for every address scores 77.9% while catching nothing — so evaluation is on precision, recall and F1 with the confusion matrix inspected directly.

## Approach

1. **Exploration** — class balance, feature distributions, missing values and constant columns checked before any modelling.
2. **Preprocessing** — [WHAT YOU ACTUALLY DID: dropped columns with high missingness, imputed, scaled numeric features?]
3. **Handling imbalance** — [class weighting / SMOTE / nothing — say which and why]
4. **Model** — [MODEL NAME] with a stratified [80/20] train/test split so both sets keep the same fraud ratio.
5. **Evaluation** — precision, recall, F1 and ROC-AUC, plus feature importances to see which behaviours drive predictions.

## Results

| Metric | Score |
|---|---|
| Precision | [X] |
| Recall | [X] |
| F1 | [X] |
| ROC-AUC | [X] |

**Most predictive features:** [TOP 3-5 FROM YOUR FEATURE IMPORTANCE OUTPUT]

[Two sentences interpreting this. Recall is the number that matters — a fraudulent wallet that slips through costs an exchange far more than a legitimate wallet flagged for manual review. Say what the model catches and what it misses.]

## Running it

Requires Python 3.8+.

```bash
git clone https://github.com/uzoechifavour07/ethereum-fraud-detection.git
cd ethereum-fraud-detection
pip install -r requirements.txt
```

Download the dataset from the Kaggle link above and place the CSV in the project folder, then:

```bash
[python train.py  — or name the notebook if that's what you have]
```

## What I learned

- Why accuracy is misleading on imbalanced data, and how the precision/recall trade-off changes depending on which error is more expensive
- That behavioural features alone carry real predictive signal, without needing identity data
- Why the train/test split has to be stratified when the positive class is the minority
- [Something you got wrong first — this is the line interviewers ask about]

## Limitations

- Labels come from historical fraud reports; fraud patterns adapt, so live performance would degrade without retraining
- On-chain behaviour only — combining it with off-chain signals (exchange KYC, known-scam databases) would likely improve detection
- Address-level, not transaction-level: it flags wallets, not individual transfers
- [Anything else you know is weak]

## Possible improvements

- Compare gradient boosting (XGBoost / LightGBM) against a logistic regression baseline
- Tune the decision threshold explicitly rather than defaulting to 0.5
- Test on the Elliptic Bitcoin dataset to see whether the approach generalises across chains
- Serve the model behind a small API so an address can be scored on demand

## Author

Favour Uzoechi — BSc Computer Science with AI, Buckinghamshire New University
