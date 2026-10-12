# Credit Card Fraud Detection — 

Live Demo link : https://ccfrauddetection-1v0t379bt-cc-fraud-detection.vercel.app/

This is the same overall project flow as the supplied Claude project, adapted to the readable new dataset:
- use `is_fraud` as the target;
- use `amt`, `category`, and timestamp-derived `hour` and `day_of_week`;
- compare Logistic Regression, Linear SVM, Random Forest, and XGBoost;
- apply SMOTE only to the training portion;
- report precision, recall, F1 and PR-AUC;
- use the same 0.2 / 0.8 review band;
- deploy the Random Forest to Vercel with an HTML frontend.

The Kaggle dataset is simulated transaction data, not real bank records.

## 1. Download the dataset
Download the new dataset from https://www.kaggle.com/datasets/kartik2112/fraud-detection.
Upload `fraudTrain.csv` and `fraudTest.csv` into the root of a Google Colab session. The training script expects those filenames and the columns `trans_date_trans_time`, `category`, `amt`, and `is_fraud`.

## 2. Train in Google Colab
1. Open a new notebook at https://colab.research.google.com/.
2. Run:
   ```python
   !pip install scikit-learn==1.6.1 imbalanced-learn xgboost pandas numpy joblib
   ```
3. Upload both CSVs with the Colab Files panel (left sidebar), so they appear at `/content/fraudTrain.csv` and `/content/fraudTest.csv`.
4. Upload/run `train.py`, or paste its contents into a cell.
5. The code compares the four classifiers, evaluates the Random Forest, and saves `output/fraud_model.joblib` and `output/sample_transactions.csv`.
6. Download those two output files from Colab.

Important: the dataset is large and SMOTE can require substantial RAM. If Colab runs out of memory, stop and tell me the exact error rather than silently changing the method.

## 3. Put the files into the project
Create this structure:
```text
project/
├── api/predict.py
├── model/fraud_model.joblib       # copy the model downloaded from Colab here
├── public/index.html
├── public/sample_transactions.csv # copy the sample CSV downloaded from Colab here
├── train.py
├── requirements.txt
├── pyproject.toml
└── vercel.json
```
Do not upload the full dataset CSVs to GitHub.

## 4. Test and deploy
Install Node.js, then:
```bash
npm install -g vercel
vercel login
vercel
```
Follow the CLI prompts. Once the preview deployment works:
```bash
vercel --prod
```

## 5. Test the API
Open `/api/predict` on the deployed domain. It should return `status: ok`, the two thresholds, and the expected features. Then test a single transaction and a CSV upload in the web page.

## Important notes
- Do not reuse the old ULB `fraud_model.joblib`; it expects `Time`, `Amount`, and `V1`–`V28`.
- The new model expects `amt`, `category`, `hour`, and `day_of_week`.
- The UI uses a small category dropdown; categories unseen during training are accepted by the one-hot encoder.
- The review queue/feedback log from the original frontend was not included in this adapted frontend because the current requested UI is the simple input-and-result flow. The same model review thresholds remain.
- Do not claim performance numbers until the new training script has actually run and produced its metrics.
