# Data Analysis and Classification using Naïve Bayes and Neural Network

## Introduction
This paper explores the analysis and classification of data using two popular machine learning algorithms: the Naïve Bayes (NB) classifier and Neural Network. The focus is on binary classification tasks using a tabular dataset of quantitative and categorical variables. Rigorous data cleaning techniques are employed to ensure the reliability and integrity of the dataset. The exploration phase involves visualizing the dataset's features through histograms and conducting statistical tests to assess feature distributions. Outliers are addressed through cleaning techniques to enhance the dataset's quality and reduce the potential for biased or erroneous results.

## Steps
### 2.1 Importing data
The data is imported from Kaggle using the Pandas library's `read_csv()` method. Pandas is chosen for its powerful data manipulation capabilities, such as filtering, sorting, and aggregation.

### 2.2 Data cleaning
Data cleaning involves removing NaN values, duplicate rows, and outliers. The interquartile method (IQR) is used to identify outliers. Any rows containing NaN values are removed using the `dropna()` method.

### 2.3 Transforming Categorical Data into Numerical Data
Categorical data, specifically the "name" column, is transformed into numerical values. Orange is represented as 0, and Grapefruit is represented as 1.

### 2.4 Calculating descriptive statistics
Descriptive statistics, such as mean, median, variance, and standard deviation, are calculated for each column using Pandas. The statistics are stored in a dictionary for easy access and analysis.

### 2.5 Standardization using the Z-score method
The Z-score transformation is applied to standardize the features. The mean and standard deviation calculated in the previous step are used to compute the Z-scores for each value in the dataset.

### 3. Hypothesis Testing for Normality
#### 3.1 Shapiro-Wilk test
The Shapiro-Wilk test is conducted to determine if the features follow a normal distribution. The test calculates the test statistic based on the observed distribution and compares it with the expected normal distribution. If the p-value is less than 0.05, the null hypothesis of normality is rejected.

#### 3.2 QQ plots
QQ plots are used to visually assess the similarity between the observed data and theoretical normal distribution. Quantiles of the observed data are compared to the quantiles of the expected normal distribution. Any deviations or departures from the expected distribution shape are visually indicated in the QQ plot.

### 4. Conditional distribution
Conditional probability plots are created to analyze the relationship between variables while conditioning a third variable. These plots provide insights into how the probability of an event or outcome varies based on different values or categories of the conditioning variable.

### 5. Naïve Bayes classifier
#### 5.1 Theory and formula
The Naïve Bayes (NB) classifier is a widely used machine learning algorithm based on Bayes' theorem. It assumes conditional independence among the features given the class. The classifier calculates the probability of each class given the features and selects the class with the highest probability as the predicted class.

#### 5.2 Software packages
The implementation of the Naïve Bayes classifier utilizes various functions and attributes from the NumPy library. NumPy provides efficient array operations and mathematical functions necessary for the classifier's computations.

## Simple NN using Pytorch
* Neural networks learn patterns and relationships from input data through training.
* Data transformation involves converting data to tensors and performing label encoding and one-hot encoding.
* Weight and bias tensors are initialized and their gradients are tracked during training.
* Forward propagation processes input data through the network's layers, including the hidden layer and activation function.
* The softmax activation function is commonly used for classification problems.
* The output layer produces the final predictions based on the activation values.
* Prediction error, measured by the Binary Cross-Entropy loss function, quantifies the model's performance.
* Backpropagation computes gradients of the parameters with respect to the loss and updates the parameters accordingly.
* Gradient Descent is a commonly used optimization algorithm in training neural networks.
* The optimization loop continues until convergence or the specified stopping criteria are met.
* Loss values are plotted over time to visualize the optimization progress.
* The achieved accuracy of the implemented neural network was mentioned, indicating its performance compared to other models.

## Conclusion
This paper demonstrates the exploration, cleaning, and classification of data using the Naïve Bayes classifier and Neural Network algorithms. The dataset's integrity is ensured through rigorous data cleaning and statistical analysis. The Naïve Bayes classifier is implemented and benchmarked.
