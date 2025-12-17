# kNN(k-Nearest Neighbor)
<img width="700" height="300" alt="image" src="https://github.com/user-attachments/assets/62754886-f981-4c58-aaaf-685a9cf149dc" />


# 1. Classification Problem 

We consider a supervised classification dataset:

$$
\mathcal{D} = {(x_i, y_i)}_{i=1}^{n}
$$

where:

* $x_i \in \mathbb{R}^p$ is a feature vector with $p$ predictors
* $y_i \in {1, 2, \dots, C}$ is the class label

The objective is to learn a decision function:

$$
f: \mathbb{R}^p \rightarrow {1, 2, \dots, C}
$$

that minimizes the **expected classification error**:

$$
\mathbb{E}[\mathbb{I}(f(x) \neq y)]
$$


# 2. K-Nearest Neighbors (KNN):

## 2.1 Nature of the Algorithm

KNN is characterized as:

* **Non-parametric**: no explicit model parameters are learned
* **Instance-based**: training samples are stored in memory
* **Lazy learning**: no optimization during training; computation occurs at inference



## 2.2 Distance Function

Classification decisions are based on a distance metric:

$$
d(x, x') : \mathbb{R}^p \times \mathbb{R}^p \rightarrow \mathbb{R}
$$

Common distance measures include:

**Euclidean distance**:

$$
d(x, x') = \sqrt{\sum_{j=1}^{p} (x_j - x'_j)^2}
$$

**Manhattan distance**:

$$
d(x, x') = \sum_{j=1}^{p} |x_j - x'_j|
$$


## 2.3 Decision Rule

For a new observation $x$:

1. Compute distances to all training samples
2. Identify the set $N_k(x)$ of the $k$ nearest neighbors
3. Predict the class using majority voting:

$$
\hat{y}(x) = \arg\max_{c} \sum_{i \in N_k(x)} \mathbb{I}(y_i = c)
$$



# 3. Statistical Properties of KNN

## 3.1 Bias–Variance Tradeoff

The hyperparameter $k$ governs model complexity:

* Small $k$:

  * Low bias
  * High variance
* Large $k$:

  * High bias
  * Low variance

Under suitable conditions, KNN converges to the **Bayes optimal classifier** when:

$$
k \rightarrow \infty, \quad \frac{k}{n} \rightarrow 0
$$



## 3.2 Curse of Dimensionality

As dimensionality $p$ increases:

$$
\frac{\max d(x_i, x_j) - \min d(x_i, x_j)}{\min d(x_i, x_j)} \rightarrow 0
$$

Implications:

* Distance values lose contrast
* Nearest and farthest neighbors become indistinguishable
* Classification performance deteriorates



# 4. Cross-Validation 

## 4.1 Purpose

Cross-validation estimates the **generalization error**:

$$
\mathbb{E}_{(x,y) \sim P}[\mathbb{I}(f(x) \neq y)]
$$



## 4.2 $K$-Fold Cross-Validation

The dataset is partitioned into $K$ disjoint folds:

$$
\mathcal{D} = \bigcup_{j=1}^{K} \mathcal{D}_j
$$

For each fold $j$:

* Train on $\mathcal{D} \setminus \mathcal{D}_j$
* Validate on $\mathcal{D}_j$

The estimated risk is:

$$
\hat{R}*{CV} = \frac{1}{K} \sum*{j=1}^{K} R_j
$$



# 5. Model Selection

Hyperparameters (e.g., $k$) are selected by minimizing the cross-validated risk:

$$
k^* = \arg\min_k \hat{R}_{CV}(k)
$$

This balances:

* Overfitting
* Underfitting



# 6. Confusion Matrix

For binary classification:

|                    | Actual Positive | Actual Negative |
| ------------------ | --------------- | --------------- |
| Predicted Positive | TP              | FP              |
| Predicted Negative | FN              | TN              |

These four quantities form the basis of **all evaluation metrics**.



# 7. Evaluation Metrics (Mathematical Definitions)

## 7.1 Accuracy

$$
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
$$

Measures overall correctness.



## 7.2 Sensitivity (Recall)

$$
Sensitivity = \frac{TP}{TP + FN}
$$

Measures the ability to detect positive cases.



## 7.3 Specificity

$$
Specificity = \frac{TN}{TN + FP}
$$

Measures the ability to correctly reject negatives.



## 7.4 Precision (Positive Predictive Value)

$$
Precision = \frac{TP}{TP + FP}
$$

Reliability of positive predictions.



## 7.5 Negative Predictive Value

$$
NPV = \frac{TN}{TN + FN}
$$

Reliability of negative predictions.



## 7.6 Balanced Accuracy

$$
Balanced\ Accuracy = \frac{Sensitivity + Specificity}{2}
$$



# 8. No Information Rate (NIR)

The No Information Rate is defined as:

$$
NIR = \max_{c} P(y = c)
$$

It represents the accuracy of a naïve majority-class classifier.



# 9. Hypothesis Testing for Accuracy

Null hypothesis:

$$
H_0: Accuracy \le NIR
$$

Alternative hypothesis:

$$
H_1: Accuracy > NIR
$$

Typically evaluated using a **binomial test**.



# 10. Cohen’s Kappa (Chance-Corrected Agreement)

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

where:

* $p_o$ is observed agreement
* $p_e$ is expected agreement by chance



# 11. Confidence Interval for Accuracy

Accuracy is modeled as a binomial proportion:

$$
\hat{p} \sim \text{Binomial}(n, p)
$$

Confidence intervals quantify uncertainty due to finite sampling.


# 12. McNemar’s Test (Error Symmetry)

Used to test whether two error types occur at equal rates:

$$
\chi^2 = \frac{(|b - c| - 1)^2}{b + c}
$$




# kNN(k-Nearest Neighbor)
```
#-------------------------------------------
set.seed(7)
fit.knn <- train(Symbols~., 
                 data=train, 
                 method="knn", 
                 metric=metric, 
                 trControl=control)
fit.knn

# check important variables
varImp(fit.knn)
p1 <- plot(varImp(fit.knn), top = 30, main="kNN")
p2 <- plot(fit.knn, main="kNN")
plot_grid(p1, p2)
png("feature_importance_knn.png", width = 800, height = 600)
plot(varImp(fit.knn), top = 30, main = "Feature Importance - Random Forest")
dev.off()

# make predictions using trained model on new/test
pred.knn <- predict(fit.knn, newdata = test)

# Model Evaluation
cm_knn <- confusionMatrix(pred.knn, test$Symbols, positive = "Cancer")
print(cm_knn)
c1 <- cm_knn$table
model_accuracies$knn <- cm_knn$overall["Accuracy"]

# Plot and save confusion matrix
plot_save_cm(c1, "knn")

# Plot and save learning curve
plot_save_learning_curve(fit.knn, "knn")
```
<img width="575" height="288" alt="fit_knn" src="https://github.com/user-attachments/assets/6026272e-4db2-48d6-b564-e6bcd2b84499" />
<img width="575" height="288" alt="knn_evaluation" src="https://github.com/user-attachments/assets/47c177b1-2935-438c-9ad4-8989b72576ac" />


# 1. Mathematical Foundation of K-Nearest Neighbors (KNN)

## 1.1 Problem Setup

You are solving a **binary classification problem**:

$$
\mathcal{D} = {(x_i, y_i)}_{i=1}^{n}
$$

Where:

* $n = 12$ samples
* $x_i \in \mathbb{R}^{139}$ (139-dimensional feature vector)
* $y_i \in {0,1}$

  * $1 =$ Cancer
  * $0 =$ Normal

Goal:

$$
f: \mathbb{R}^{139} \rightarrow {0,1}
$$



## 1.2 Distance Computation

KNN relies on a **distance metric**.

Most common (caret default):

### Euclidean Distance

$$
d(x_i, x_j) = \sqrt{\sum_{k=1}^{139} (x_{ik} - x_{jk})^2}
$$

 **Problem in high dimensions**:

$$
\lim_{p \to \infty} \frac{\max(d) - \min(d)}{\min(d)} \to 0
$$

➡ Distances become indistinguishable
➡ Nearest neighbor ≈ farthest neighbor
➡ Classification becomes unstable



## 2. Curse of Dimensionality (Mathematical Insight)

For uniformly distributed data in $[0,1]^p$:

Expected distance:

$$
\mathbb{E}[d] \propto \sqrt{p}
$$

Variance:

$$
\mathrm{Var}(d) \to 0 \quad \text{as} \quad p \to \infty
$$

With $p = 139$, distances lose discrimination power.

 **This is the core theoretical limitation of model**.



# 3. Cross-Validation (CV) Mathematics

## 3.1 8-Fold Cross-Validation

Dataset is partitioned:

$$
\mathcal{D} = \bigcup_{k=1}^{8} \mathcal{D}_k
$$

For each fold:

* Train on $\mathcal{D} \setminus \mathcal{D}_k$
* Validate on $\mathcal{D}_k$

Each training set size:

$$
n_{train} \approx 10\text{–}11
$$

Validation set size:

$$
n_{val} \approx 1\text{–}2
$$



## 3.2 Accuracy Estimation

Fold accuracy:

$$
\mathrm{Acc}_k = \frac{1}{|\mathcal{D}*k|} \sum*{i \in \mathcal{D}_k} \mathbb{I}(\hat{y}_i = y_i)
$$

Overall CV accuracy:

$$
\mathrm{Acc}*{CV} = \frac{1}{8} \sum*{k=1}^{8} \mathrm{Acc}_k
$$

 With $|\mathcal{D}_k| \le 2$, variance is extremely high.



# 4. Hyperparameter ($k$) Optimization

## 4.1 Decision Rule

Given test point $x$:

$$
\hat{y}(x) =
\begin{cases}
1 & \text{if } \sum_{i \in N_k(x)} y_i \ge \frac{k}{2} \
0 & \text{otherwise}
\end{cases}
$$

Where:

* $N_k(x)$ = set of $k$ nearest neighbors



## 4.2 Bias–Variance Tradeoff

| $k$   | Bias | Variance |
| ----- | ---- | -------- |
| Small | Low  | High     |
| Large | High | Low      |

Your results:

| $k$ | Accuracy | Kappa   |
| --- | -------- | ------- |
| 5   | 0.5625   | 0.1667  |
| 7   | 0.3125   | −0.1429 |
| 9   | 0.5000   | 0.0000  |

### Interpretation

* $k=7$ → oversmoothing
* $k=9$ → majority-class dominance
* $k=5$ → optimal compromise



# 5. Confusion Matrix (Formal Definition)

|                    | Actual Positive | Actual Negative |
| ------------------ | --------------- | --------------- |
| Predicted Positive | TP              | FP              |
| Predicted Negative | FN              | TN              |

Your values:

* TP = 3
* FN = 1
* FP = 0
* TN = 4
* Total $n = 8$



# 6. Metric-by-Metric Mathematical Explanation

## 6.1 Accuracy

$$
\mathrm{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

$$
= \frac{3 + 4}{8} = 0.875
$$



## 6.2 No Information Rate (NIR)

$$
\mathrm{NIR} = \max(P(y=1), P(y=0))
$$

$$
P(y=1) = P(y=0) = 0.5 \Rightarrow \mathrm{NIR} = 0.5
$$



## 6.3 Hypothesis Test: Accuracy > NIR

Null hypothesis:

$$
H_0: \mathrm{Accuracy} \le \mathrm{NIR}
$$

Using binomial test:

$$
P(X \ge 7 \mid n=8, p=0.5) = 0.035
$$

 Reject $H_0$ at $\alpha = 0.05$



## 6.4 Cohen’s Kappa

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

Where:

* $p_o$ = observed agreement
* $p_e$ = expected agreement by chance

$$
p_o = \frac{7}{8} = 0.875
$$

$$
p_e = (P_{pred+} \cdot P_{actual+}) + (P_{pred-} \cdot P_{actual-})
$$

$$
= (0.375 \cdot 0.5) + (0.625 \cdot 0.5) = 0.5
$$

$$
\kappa = \frac{0.875 - 0.5}{1 - 0.5} = 0.75
$$



## 6.5 Sensitivity (Recall)

$$
\mathrm{Sensitivity} = \frac{TP}{TP + FN} = \frac{3}{4} = 0.75
$$



## 6.6 Specificity

$$
\mathrm{Specificity} = \frac{TN}{TN + FP} = \frac{4}{4} = 1.0
$$



## 6.7 Precision (PPV)

$$
\mathrm{Precision} = \frac{TP}{TP + FP} = 1.0
$$



## 6.8 Negative Predictive Value (NPV)

$$
\mathrm{NPV} = \frac{TN}{TN + FN} = \frac{4}{5} = 0.8
$$



## 6.9 Balanced Accuracy

$$
\mathrm{Balanced\ Accuracy} = \frac{\mathrm{Sensitivity} + \mathrm{Specificity}}{2}
$$

$$
= \frac{0.75 + 1.0}{2} = 0.875
$$



# 7. Confidence Interval for Accuracy

Using binomial proportion CI:

$$
\mathrm{CI} = \hat{p} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
$$

$$
= 0.875 \pm 1.96 \sqrt{\frac{0.875 \cdot 0.125}{8}}
$$

 Wide interval due to small $n$



# 8. McNemar’s Test (Error Symmetry)

$$
\chi^2 = \frac{(|b - c| - 1)^2}{b + c}
$$

Where:

* $b =$ FP = 0
* $c =$ FN = 1

$$
p = 1.0
$$

 Test is underpowered for small samples



# 9. Model Appears “Good” but Is Not Reliable

### Mathematical Reasons

* High variance estimator
* Overfitting due to local memorization
* Distance collapse in high dimensions
* Sampling bias in test set



# 10.  Mathematical Conclusion

> The KNN classifier approximates the Bayes decision boundary locally; however, under extreme dimensionality and limited sample size, the distance metric loses discriminative power, resulting in high variance estimates and optimistic performance metrics.

