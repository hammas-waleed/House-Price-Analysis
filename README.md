# HOUSE PRICE ANALYSIS
A beginner-friendly Python project for analyzing and preprocessing a housing dataset using Pandas, Matplotlib, Seaborn, and Scikit-learn.

## TECHNOLOGIES USED
Python
Pandas
Matplotlib
Seaborn
Scikit-learn

## PROJECT OVERVIEW
This project covers the basic Data Analysis, Exploratory Data Analysis (EDA), and Machine Learning preprocessing workflow.

Main tasks performed:
* Loaded and explored the housing dataset
* Checked dataset shape, columns, data types, and information
* Checked missing values and duplicate records
* Performed basic statistical analysis
* Found the most expensive and cheapest houses
* Compared average price by number of bedrooms
* Created new features
* Performed univariate analysis
* Performed bivariate analysis
* Performed multivariate analysis
* Created correlation analysis and a heatmap
* Visualized relationships between area and price
* Detected outliers using boxplots and the IQR method
* Performed binning on house areas
* Encoded categorical variables
* Performed train-test split
* Applied feature scaling using StandardScaler

## DATASET
The dataset contains information about houses including:
* Price
* Area
* Bedrooms
* Bathrooms
* Parking
* Main road access
* Guest room
* Basement
* Hot water heating
* Air conditioning
* Furnishing status

The target variable for the future machine learning stage is:
* text
* price

## FEATURE ENGINEERING
The project creates additional features including:

```text
price_per_sqft
total_rooms
area_category
```

`price_per_sqft` is used for analysis but is not included in the machine learning features because it is calculated using the target variable `price`.

## DATA PREPROCESSING

Categorical variables are converted into numerical values so that they can be used during machine learning preprocessing.

The dataset is then divided into:

```text
80% Training Data
20% Testing Data
```

Feature scaling is performed using:

```text
StandardScaler
```

The scaler is fitted only on the training data and then applied to the test data.

## MACHINE LEARNING STATUS

This project currently stops at the preprocessing and scaling stage.

Machine learning algorithms such as Linear Regression will be added after studying the relevant machine learning concepts.

## HOW TO RUN

Install the required libraries:

```bash
pip install -r requirements.txt
```

Make sure `Housing.csv` is in the same folder as the Python file.

Run the project:

```bash
python house_price_analyzer.py
```
## PURPOSE

This project was created as part of my Machine Learning and AI Engineering learning journey to practice data analysis, exploratory data analysis, feature engineering, data preprocessing, train-test splitting, and feature scaling.

## AUTHOR

Hammas Waleed
