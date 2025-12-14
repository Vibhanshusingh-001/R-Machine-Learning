

##  Setting Up Cross-Validation Control

```r
control <- trainControl(method="cv", number=8)
metric <- "Accuracy"
```

**Explanation:**

1. `trainControl()` is a function from the `caret` package that specifies **how model training and evaluation will be done**.
2. `method="cv"` specifies **cross-validation**, which is a resampling method to estimate model performance.
3. `number=8` means we are doing **8-fold cross-validation**:

   * The data is split into 8 parts (folds).
   * Each fold is used as a **validation set** once, while the remaining 7 folds are used for **training**.
   * This helps avoid overfitting and gives a more robust performance estimate.
4. `metric <- "Accuracy"` sets the **performance metric** for model evaluation. Accuracy measures the proportion of correctly classified samples.


---

##  Training the kNN Model

```r
set.seed(7)
fit.knn <- train(Symbols~., 
                 data=train, 
                 method="knn", 
                 metric=metric, 
                 trControl=control)
fit.knn
```

**Explanation:**

1. `set.seed(7)` ensures **reproducibility**—the same random splits are used each time.
2. `train()` is the main `caret` function for training models.

   * `Symbols ~ .` → Predict `Symbols` using all other columns.
   * `data=train` → Use the training dataset.
   * `method="knn"` → Train a **k-Nearest Neighbors** classifier.
   * `metric=metric` → Optimize **accuracy**.
   * `trControl=control` → Use the 8-fold cross-validation defined earlier.
3. `fit.knn` now contains:

   * The **trained kNN model**.
   * Cross-validated performance metrics.
   * Optimal `k` value chosen automatically.

---

## Feature Importance (Optional for kNN)

```r
varImp(fit.knn)
p1 <- plot(varImp(fit.knn), top = 30, main="kNN")
p2 <- plot(fit.knn, main="kNN")
plot_grid(p1, p2)
png("feature_importance_knn.png", width = 800, height = 600)
plot(varImp(fit.knn), top = 30, main = "Feature Importance - Random Forest")
dev.off()
```

**Explanation:**

1. `varImp()` computes the **importance of each feature** in the model.

   * For kNN, it is usually based on **how much each feature contributes to classification distance**.
2. `plot()` visualizes top features.
3. `plot_grid()` combines multiple plots into a single figure.
4. `png()` saves the figure.

> **Note:** The title mentions "Random Forest"—this seems like a copy-paste error. It should refer to kNN.

---

## Step 8: Making Predictions on Test Data

```r
pred.knn <- predict(fit.knn, newdata = test)
```

* `predict()` uses the **trained kNN model** to classify the test dataset.
* `pred.knn` contains the **predicted class labels** for all test samples.

---

##  Evaluating Model Performance

```r
cm_knn <- confusionMatrix(pred.knn, test$Symbols, positive = "Cancer")
print(cm_knn)
c1 <- cm_knn$table
model_accuracies$knn <- cm_knn$overall["Accuracy"]
```

**Explanation:**

1. `confusionMatrix()` calculates:

   * **True positives, false positives, true negatives, false negatives**
   * **Accuracy, sensitivity, specificity**, etc.
   * `positive = "Cancer"` sets which class is considered "positive".
2. `cm_knn$table` extracts the raw confusion matrix.
3. `model_accuracies$knn` stores **accuracy** for later comparison.




