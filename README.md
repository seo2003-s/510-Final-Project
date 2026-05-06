# 510-Final-Project- Identifying Key Predictors of Cardiovascular Diseases

## Project Introduction
The goal of this project is to analyze the datasets related to cardiovascular health including heart disease, heart failure, and general cardiovascular disease. Although the datasets focus on slightly different conditions, they contain many similar health indicators such as age, blood pressure, cholesterol levels, gender, and heart rate. Using these, I found the most significant predictors associated with these conditions.
 
## Data Sources 

| # | Dataset | Source URL | Access Type | Fields Used | Format | Python Access |
|---|---------|-----------|-------------|-------------|--------|---------------|
| 1 | Framingham Heart Disease Dataset | [Kaggle](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset/data) | API | age, sex, totChol, sysBP, TenYearCHD | CSV | Yes (kagglehub) |
| 2 | Heart Failure Prediction Dataset | [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data) | API | Age, Sex, Cholesterol, RestingBP, HeartDisease | CSV | Yes (kagglehub) |
| 3 | Cardiovascular Disease Dataset | [Mendeley](https://data.mendeley.com/datasets/dzz48mvjht/) | Web Page | age, gender, serumcholestrol, restingBP, target | CSV | No (manual download) |

## Analysis 
I first loaded the three datasets and standardized the variables to a common name which were age, sex, cholesterol, resting blood pressure, and target. From there I cleaned the data and created histograms, box plots, scatter plots, and found additional summary statistics for the variables. I then checked if assumptions are met before modelling, including independence, VIF values and logit plots. A logistic regression model was then made with the likelihood ratio test, Wald test and the Odds-Ratio test to find the most significant predictor. 

## Results
Combined dataset: 5,883 rows after cleaning across three sources
VIF values: All near 1.0, indicating no multicollinearity among features
Model AUC: 0.75, showing the model is strong
Accuracy: 0.51 at threshold 0.3 (prioritizes recall over precision)
Recall: 0.86, the model correctly identifies 86% of heart disease cases
Durbin-Watson: 1.67, suggesting no significant autocorrelation in residuals
Strongest predictors: Resting blood pressure (odds ratio 2.1) were the strongest predictors of heart disease, followed by cholesterol (1.81) and sex(1.62)
Statistical significance: All four features were statistically significant (p < 0.001)

## 1. Setup
```bash
git clone https://github.com/seo2003-s/510-Final-Project.git
cd 510-Final-Project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

## 3. Run main.py
```bash
python main.py --cardio-csv /path/to/Cardiovascular_Disease_Dataset.csv
```
The Framingham and Heart Failure datasets are downloaded through kagglehub. 

## 4. Run Tests
```bash
python -m pytest tests.py -v
```
