# Data-Cleaning-and-Preprocessing
#Titanic Dataset:
This document outlines the complete data cleaning and preprocessing steps applied to the Titanic dataset to prepare it for machine learning. The raw dataset contained 891 passenger records with 12 features, including mixed data types and missing values. Through systematic processing, we've transformed it into a clean, analysis-ready dataset suitable for predictive modeling.

#Missing Values
The initial analysis revealed significant missing data, particularly in the Cabin column (687/891 missing) and Age column (177 missing). We addressed this by: (1) completely removing the Cabin column due to excessive missingness, (2) imputing missing Age values with the median age of 28 years, and (3) filling the 2 missing Embarked values with the most frequent port ('S'). This approach preserved 100% of the remaining data while eliminating all null values.

#Feature Engineering
Categorical variables were converted to numerical formats to make them machine-readable. The Sex feature was label encoded (Female=0, Male=1), while the Embarked feature (with 3 categories) was split into two binary dummy variables (Embarked_Q and Embarked_S) using one-hot encoding, with Embarked_C serving as the reference category. This transformation maintained all categorical information without introducing ordinal relationships where none existed.

#Outlier Treatment
We identified and removed extreme values in the Fare feature using the Interquartile Range (IQR) method. Any fares exceeding £65 (Q3 + 1.5*IQR) were considered outliers and removed, affecting 16 records (1.8% of data). This conservative approach maintained data integrity while reducing potential skewness in model training.

#Results
The final cleaned dataset contains 875 complete passenger records with 10 optimized features. All variables are now numerical, properly scaled, and free of missing values - making it immediately suitable for machine learning applications. The preprocessing pipeline has successfully transformed the raw data while preserving its predictive potential for survival analysis.
