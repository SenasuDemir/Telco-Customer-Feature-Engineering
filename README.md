# Telco Customer Churn Prediction using CatBoost Classifier

This project aims to develop a machine learning model to predict customer churn in a telecommunications company. The model is built using the **Telco Customer Churn** dataset, which contains information on customers' demographics, account details, and service subscriptions. The primary objective is to identify customers who are likely to leave the company, enabling the company to take preventive measures to retain them.

## Dataset

The dataset used for this project is the **Telco Customer Churn** dataset, containing **7,043 observations** and **21 variables**. It represents data from a fictional telecommunications company that provides home phone and internet services in California during the third quarter.

### Variables

- **CustomerID**: Unique customer ID
- **Gender**: Customer gender
- **SeniorCitizen**: Indicator of whether the customer is a senior citizen (1, 0)
- **Partner**: Whether the customer has a partner (Yes, No)
- **Dependents**: Whether the customer has dependents (Yes, No)
- **Tenure**: Number of months the customer has stayed with the company
- **PhoneService**: Whether the customer has phone service (Yes, No)
- **MultipleLines**: Whether the customer has multiple lines (Yes, No, No phone service)
- **InternetService**: The customer's internet service provider (DSL, Fiber optic, No)
- **OnlineSecurity**: Whether the customer has online security (Yes, No, No internet service)
- **OnlineBackup**: Whether the customer has online backup (Yes, No, No internet service)
- **DeviceProtection**: Whether the customer has device protection (Yes, No, No internet service)
- **TechSupport**: Whether the customer has tech support (Yes, No, No internet service)
- **StreamingTV**: Whether the customer streams TV (Yes, No, No internet service)
- **StreamingMovies**: Whether the customer streams movies (Yes, No, No internet service)
- **Contract**: The customer’s contract term (Month-to-month, One year, Two years)
- **PaperlessBilling**: Whether the customer uses paperless billing (Yes, No)
- **PaymentMethod**: The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
- **MonthlyCharges**: The monthly charges for the customer
- **TotalCharges**: The total amount charged to the customer
- **Churn**: Whether the customer has churned (Yes, No)

## Feature Engineering

Several feature engineering steps were applied to improve the model's performance:

1. **Handling missing values**: Specifically, missing values in the `TotalCharges` column were handled.
2. **Encoding categorical variables**: Categorical variables such as `Gender`, `Partner`, and `Contract` were converted into numerical formats using encoding techniques suitable for the machine learning model.
3. **Scaling numeric features**: Continuous variables such as `MonthlyCharges` and `TotalCharges` were scaled to ensure uniformity.
4. **Creating interaction features**: Interaction between specific variables such as `Tenure`, `Contract`, and `PaymentMethod` was used to derive new insights.
5. **Handling outliers**: Outliers in certain columns, especially numeric ones, were dealt with appropriately.

## Model: CatBoost Classifier

The **CatBoostClassifier** from the CatBoost library was used to build the prediction model. This model was chosen due to its efficient handling of categorical variables and its ability to manage non-linear relationships in the data.

### Performance Metrics

The model was evaluated using several metrics:

| Metric        | **Value (Engineered)** | **Value (Baseline)** |
|---------------|------------------------|----------------------|
| **Accuracy**  | 0.79                   | 0.78                 |
| **Recall**    | 0.65                   | 0.633                |
| **Precision** | 0.50                   | 0.49                 |
| **F1 Score**  | 0.56                   | 0.55                 |
| **AUC**       | 0.74                   | 0.73                 |

