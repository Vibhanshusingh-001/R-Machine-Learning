
# A Short Introduction to the caret Package

The caret package (short for Classification And REgression Training) contains functions to streamline the model training process for complex regression and classification problems. The package utilizes a number of R packages but tries not to load them all at package start-up (by removing formal package dependencies, the package startup time can be greatly decreased). The package “suggests” field includes 31 packages. caret loads packages as needed and assumes that they are installed. If a modeling package is missing, there is a prompt to install it.

## Install caret using

```r
install.packages("caret", dependencies = c("Depends", "Suggests"))
````

to ensure that all the needed packages are installed.

The main help pages for the package are at
[https://topepo.github.io/caret/](https://topepo.github.io/caret/)
Here, there are extended examples and a large amount of information that previously found in the package vignettes.

caret has several functions that attempt to streamline the model building and evaluation process, as well as feature selection and other techniques.

One of the primary tools in the package is the `train` function which can be used to:

* evaluate, using resampling, the effect of model tuning parameters on performance
* choose the ``optimal’’ model across these parameters
* estimate model performance from a training set

A formal algorithm description can be found in Section 5.1 of the caret manual.

There are options for customizing almost every step of this process (e.g. resampling technique, choosing the optimal parameters etc). To demonstrate this function, the Sonar data from the mlbench package will be used.

The Sonar data consist of 208 data points collected on 60 predictors. The goal is to predict the two classes M for metal cylinder or R for rock).

First, we split the data into two groups: a training set and a test set. To do this, the `createDataPartition` function is used:

```r
library(caret)
library(mlbench)
data(Sonar)

set.seed(107)
inTrain <- createDataPartition(
  y = Sonar$Class,
  ## the outcome data are needed
  p = .75,
  ## The percentage of data in the
  ## training set
  list = FALSE
)
## The format of the results

## The output is a set of integers for the rows of Sonar
## that belong in the training set.
str(inTrain)
#>  int [1:157, 1] 1 2 3 4 5 7 10 11 12 13 ...
#>  - attr(*, "dimnames")=List of 2
#>   ..$ : NULL
#>   ..$ : chr "Resample1"
```

By default, `createDataPartition` does a stratified random split of the data. To partition the data:

```r
training <- Sonar[ inTrain,]
testing  <- Sonar[-inTrain,]

nrow(training)
#> [1] 157
nrow(testing)
#> [1] 51
```

To tune a model using the algorithm above, the `train` function can be used. More details on this function can be found at
[https://topepo.github.io/caret/model-training-and-tuning.html](https://topepo.github.io/caret/model-training-and-tuning.html).
Here, a partial least squares discriminant analysis (PLSDA) model will be tuned over the number of PLS components that should be retained. The most basic syntax to do this is:

```r
plsFit <- train(
  Class ~ .,
  data = training,
  method = "pls",
  ## Center and scale the predictors for the training
  ## set and all future samples.
  preProc = c("center", "scale")
)
```

However, we would probably like to customize it in a few ways:

* expand the set of PLS models that the function evaluates. By default, the function will tune over three values of each tuning parameter.
* the type of resampling used. The simple bootstrap is used by default. We will have the function use three repeats of 10-fold cross-validation.
* the methods for measuring performance. If unspecified, overall accuracy and the Kappa statistic are computed. For regression models, root mean squared error and R2 are computed. Here, the function will be altered to estimate the area under the ROC curve, the sensitivity and specificity.

To change the candidate values of the tuning parameter, either of the `tuneLength` or `tuneGrid` arguments can be used. The train function can generate a candidate set of parameter values and the `tuneLength` argument controls how many are evaluated. In the case of PLS, the function uses a sequence of integers from 1 to `tuneLength`. If we want to evaluate all integers between 1 and 15, setting `tuneLength = 15` would achieve this. The `tuneGrid` argument is used when specific values are desired. A data frame is used where each row is a tuning parameter setting and each column is a tuning parameter. An example is used below to illustrate this.

```r
plsFit <- train(
  Class ~ .,
  data = training,
  method = "pls",
  preProc = c("center", "scale"),
  ## added:
  tuneLength = 15
)
```

To modify the resampling method, a `trainControl` function is used. The option `method` controls the type of resampling and defaults to `"boot"`. Another method, `"repeatedcv"`, is used to specify repeated K-fold cross-validation (and the argument `repeats` controls the number of repetitions). K is controlled by the `number` argument and defaults to 10. The new syntax is then:

```r
ctrl <- trainControl(method = "repeatedcv", repeats = 3)

plsFit <- train(
  Class ~ .,
  data = training,
  method = "pls",
  preProc = c("center", "scale"),
  tuneLength = 15,
  ## added:
  trControl = ctrl
)
```

Finally, to choose different measures of performance, additional arguments are given to `trainControl`. The `summaryFunction` argument is used to pass in a function that takes the observed and predicted values and estimate some measure of performance. Two such functions are already included in the package: `defaultSummary` and `twoClassSummary`. The latter will compute measures specific to two-class problems, such as the area under the ROC curve, the sensitivity and specificity. Since the ROC curve is based on the predicted class probabilities (which are not computed automatically), another option is required. The `classProbs = TRUE` option is used to include these calculations.

Lastly, the function will pick the tuning parameters associated with the best results. Since we are using custom performance measures, the criterion that should be optimized must also be specified. In the call to `train`, we can use `metric = "ROC"` to do this.

```r
ctrl <- trainControl(
  method = "repeatedcv", 
  repeats = 3,
  classProbs = TRUE, 
  summaryFunction = twoClassSummary
)

set.seed(123)
plsFit <- train(
  Class ~ .,
  data = training,
  method = "pls",
  preProc = c("center", "scale"),
  tuneLength = 15,
  trControl = ctrl,
  metric = "ROC"
)
plsFit
```

```r
#> Partial Least Squares 
#> 
#> 157 samples
#>  60 predictor
#>   2 classes: 'M', 'R' 
#> 
#> Pre-processing: centered (60), scaled (60) 
#> Resampling: Cross-Validated (10 fold, repeated 3 times) 
#> Summary of sample sizes: 141, 141, 142, 142, 141, 142, ... 
#> Resampling results across tuning parameters:
#> 
#>   ncomp  ROC    Sens   Spec 
#>    1     0.805  0.726  0.690
#>    2     0.848  0.750  0.801
#>    3     0.849  0.764  0.748
#>    4     0.836  0.765  0.736
#>    5     0.812  0.748  0.755
#>    6     0.789  0.724  0.699
#>    7     0.794  0.744  0.689
#>    8     0.801  0.739  0.698
#>    9     0.793  0.758  0.677
#>   10     0.790  0.741  0.690
#>   11     0.787  0.742  0.710
#>   12     0.777  0.737  0.715
#>   13     0.772  0.738  0.700
#>   14     0.768  0.718  0.690
#>   15     0.768  0.715  0.690
#> 
#> ROC was used to select the optimal model using
#>  the largest value.
#> The final value used for the model was ncomp = 3.
```

In this output the grid of results are the average resampled estimates of performance. The note at the bottom tells the user that 3 PLS components were found to be optimal. Based on this value, a final PLS model is fit to the whole data set using this specification and this is the model that is used to predict future samples.

The package has several functions for visualizing the results. One method for doing this is the `ggplot` function for train objects. The command `ggplot(plsFit)` produced the results seen in Figure ??? and shows the relationship between the resampled performance values and the number of PLS components.

```r
ggplot(plsFit)
```
<img width="857" height="621" alt="image" src="https://github.com/user-attachments/assets/1554935f-c999-4054-903b-b6b39d191d95" />

To predict new samples, `predict.train` can be used. For classification models, the default behavior is to calculate the predicted class. The option `type = "prob"` can be used to compute class probabilities from the model. For example:

```r
plsClasses <- predict(plsFit, newdata = testing)
str(plsClasses)
#>  Factor w/ 2 levels "M","R": 2 1 1 1 2 2 1 2 2 2 ...

plsProbs <- predict(plsFit, newdata = testing, type = "prob")
head(plsProbs)
#>        M     R
#> 6  0.288 0.712
#> 8  0.648 0.352
#> 9  0.659 0.341
#> 15 0.529 0.471
#> 26 0.430 0.570
#> 27 0.492 0.508
```

caret contains a function to compute the confusion matrix and associated statistics for the model fit:

```r
confusionMatrix(data = plsClasses, testing$Class)
```

```r
#> Confusion Matrix and Statistics
#> 
#>           Reference
#> Prediction  M  R
#>          M 21  7
#>          R  6 17
#> 
#>                Accuracy : 0.745         
#>                  95% CI : (0.604, 0.857)
#>     No Information Rate : 0.529         
#>     P-Value [Acc > NIR] : 0.00131       
#> 
#>                   Kappa : 0.487         
#> 
#>  Mcnemar's Test P-Value : 1.00000       
#> 
#>             Sensitivity : 0.778         
#>             Specificity : 0.708         
#>          Pos Pred Value : 0.750         
#>          Neg Pred Value : 0.739         
#>              Prevalence : 0.529         
#>          Detection Rate : 0.412         
#>    Detection Prevalence : 0.549         
#>       Balanced Accuracy : 0.743         
#> 
#>        'Positive' Class : M             
```

To fit an another model to the data, `train` can be invoked with minimal changes. Lists of models available can be found at
[https://topepo.github.io/caret/available-models.html](https://topepo.github.io/caret/available-models.html)
or
[https://topepo.github.io/caret/train-models-by-tag.html](https://topepo.github.io/caret/train-models-by-tag.html).

For example, to fit a regularized discriminant model to these data, the following syntax can be used:

```r
## To illustrate, a custom grid is used
rdaGrid = data.frame(gamma = (0:4)/4, lambda = 3/4)
set.seed(123)
rdaFit <- train(
  Class ~ .,
  data = training,
  method = "rda",
  tuneGrid = rdaGrid,
  trControl = ctrl,
  metric = "ROC"
)
rdaFit
```

```r
#> Regularized Discriminant Analysis 
#> 
#> 157 samples
#>  60 predictor
#>   2 classes: 'M', 'R' 
#> 
#> No pre-processing
#> Resampling: Cross-Validated (10 fold, repeated 3 times) 
#> Summary of sample sizes: 141, 141, 142, 142, 141, 142, ... 
#> Resampling results across tuning parameters:
#> 
#>   gamma  ROC    Sens   Spec 
#>   0.00   0.778  0.723  0.682
#>   0.25   0.887  0.864  0.786
#>   0.50   0.876  0.851  0.730
#>   0.75   0.863  0.830  0.710
#>   1.00   0.734  0.680  0.636
#> 
#> Tuning parameter 'lambda' was held constant at a
#>  value of 0.75
#> ROC was used to select the optimal model using
#>  the largest value.
#> The final values used for the model were gamma =
#>  0.25 and lambda = 0.75.
```

```r
rdaClasses <- predict(rdaFit, newdata = testing)
confusionMatrix(rdaClasses, testing$Class)
```

How do these models compare in terms of their resampling results? The `resamples` function can be used to collect, summarize and contrast the resampling results. Since the random number seeds were initialized to the same value prior to calling `train`, the same folds were used for each model. To assemble them:

```r
resamps <- resamples(list(pls = plsFit, rda = rdaFit))
summary(resamps)
```

There are several functions to visualize these results. For example, a Bland-Altman type plot can be created using:

```r
xyplot(resamps, what = "BlandAltman")
```
<img width="857" height="610" alt="image" src="https://github.com/user-attachments/assets/70d64bbd-e49c-4704-88a8-abefb28d020e" />

The results look similar. Since, for each resample, there are paired results a paired t-test can be used to assess whether there is a difference in the average resampled area under the ROC curve. The `diff.resamples` function can be used to compute this:

```r
diffs <- diff(resamps)
summary(diffs)
```

Based on this analysis, the difference between the models is -0.038 ROC units (the RDA model is slightly higher) and the two-sided p-value for this difference is 5e-04.


