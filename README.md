

---

# Customer Churn Prediction

This project focuses on predicting customer churn for a telecommunications company using Python. The project involves data exploration, preprocessing, feature engineering, model training, and evaluation.

## Project Overview

Customer churn prediction is crucial for telecom companies to retain customers by identifying the factors that contribute to customer attrition. This project aims to build a machine learning model that predicts whether a customer will churn based on various features such as customer demographics, services availed, and account information.

## Dataset

The dataset used in this project is the [Telco Customer Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn), which contains 7,043 rows and 21 columns. The key features include:

- **CustomerID**: Unique identifier for each customer.
- **Gender**: Gender of the customer.
- **SeniorCitizen**: Indicates if the customer is a senior citizen (1) or not (0).
- **Partner**: Whether the customer has a partner.
- **Dependents**: Whether the customer has dependents.
- **Tenure**: Number of months the customer has stayed with the company.
- **PhoneService**: Whether the customer has a phone service.
- **MultipleLines**: Whether the customer has multiple lines.
- **InternetService**: Type of internet service (DSL, Fiber optic, No).
- **OnlineSecurity**: Whether the customer has online security.
- **OnlineBackup**: Whether the customer has online backup.
- **DeviceProtection**: Whether the customer has device protection.
- **TechSupport**: Whether the customer has tech support.
- **StreamingTV**: Whether the customer has streaming TV.
- **StreamingMovies**: Whether the customer has streaming movies.
- **Contract**: Type of contract (Month-to-month, One year, Two year).
- **PaperlessBilling**: Whether the customer has paperless billing.
- **PaymentMethod**: Payment method used by the customer.
- **MonthlyCharges**: The amount charged to the customer monthly.
- **TotalCharges**: The total amount charged to the customer.
- **Churn**: Whether the customer has churned (Yes or No).

## Data Exploration

1. **Loading the Data**: The dataset is loaded using `pandas.read_csv()` and the first few rows are inspected using `data.head()`.

2. **Descriptive Statistics**: Key statistics of the dataset are explored using `data.describe()` to understand the distribution of numerical variables.

3. **Missing Values**: Missing values are checked using `data.isnull().sum()`.

4. **Data Types**: Data types of each column are inspected to ensure appropriate types for further processing.

## Data Preprocessing

1. **Handling Missing Values**: Missing values in the `TotalCharges` column are handled by converting them to numerical format and filling them with appropriate values.

2. **Converting Categorical Variables**: Categorical variables such as `Yes`/`No` responses are converted to binary format (1/0) for model compatibility.

3. **Encoding**: The dataset's categorical variables are encoded using techniques such as one-hot encoding to convert them into a format suitable for machine learning models.

4. **Feature Scaling**: Continuous variables like `MonthlyCharges` and `TotalCharges` are scaled using standardization techniques.

## Feature Selection

1. **Feature Selection**: Relevant features are selected using correlation analysis and domain expertise to reduce dimensionality and improve model performance.

## Model Training and Evaluation

1. **Splitting the Data**: The dataset is split into training and testing sets using an 80-20 ratio.

2. **Model Selection**: Several machine learning models are tested, including Logistic Regression and Decision Trees.

3. **Training the Model**: The models are trained using the training dataset.

4. **Model Evaluation**: The performance of each model is evaluated using metrics such as accuracy, precision, recall, F1-score, and ROC-AUC curve.

## Results

The best-performing model was Logistic Regression, achieving an accuracy of 81%, precision of 83%, recall of 92%. The confusion matrix and other relevant evaluation metrics are provided to give a detailed overview of model performance.

## Conclusion

The project successfully developed a model that can predict customer churn with significant accuracy. The most important features contributing to churn were identified, providing actionable insights for the telecommunications company to improve customer retention strategies.

---

