

#  k-Nearest Neighbors (kNN)

## What kNN is (core idea)

**kNN is a distance-based classification algorithm.**

Instead of learning rules or equations, it works like this:

> “Tell me the class of a new sample by looking at the classes of its nearest neighbors.”

---

## How kNN works step by step

1. Each sample is represented as a **point in feature space**
2. For a new (unknown) sample:

   * Compute distance to all training samples
3. Select the **k closest samples**
4. Assign the class that appears **most frequently** among those k neighbors

---

## Simple example

Suppose:

* k = 5
* Among the 5 nearest samples:

  * 3 are **Cancer**
  * 2 are **Normal**

 Prediction = **Cancer**

---

## Why k matters

| k value | Behavior                           |
| ------- | ---------------------------------- |
| Small k | Very sensitive, noisy, overfitting |
| Large k | Smoother, may underfit             |

Your model tested multiple k values and chose **k = 5**, balancing noise and stability.

---

## Key properties of kNN

* No explicit training
* Sensitive to:

  * Feature scaling
  * High dimensionality
* Works best when:

  * Data is well-scaled
  * Classes are locally clustered

---

#  No Information Rate (NIR)

## What NIR means

**NIR is the accuracy you get without using a model at all.**

It is the accuracy of a **dumb baseline classifier** that always predicts the **most frequent class**.

---

## Example

Dataset:

* 50% Cancer
* 50% Normal

If you always guess **Cancer**:

* Accuracy = 50%

 **That is the NIR**

---

## Why NIR is important

Accuracy alone is misleading.

A model with:

* Accuracy = 60%
* NIR = 55%

Is **barely useful**.

Your model:

* Accuracy is clearly higher than NIR
* Indicates **real learning**

---

#  P-Value [Accuracy > NIR]

## What this p-value tests

It answers:

> “Is the model significantly better than random or naive guessing?”

---

## Statistical meaning

* **Null hypothesis (H₀)**:

  * Model accuracy ≤ NIR
* **Alternative hypothesis (H₁)**:

  * Model accuracy > NIR

---

## Interpretation

| P-value | Meaning                                     |
| ------- | ------------------------------------------- |
| < 0.05  | Model is statistically better than baseline |
| ≥ 0.05  | Model may be no better than guessing        |

Your result:

* p < 0.05
* Model adds **real predictive information**

---

#  Cohen’s Kappa

## Why accuracy is not enough

Accuracy ignores **chance agreement**.

Example:

* Balanced dataset
* Random guessing gives ~50% accuracy

Accuracy does not tell:

* Whether predictions are **better than chance**

---

## What Kappa measures

**Kappa measures agreement between prediction and truth, corrected for chance.**

Formula (conceptual):

```
Kappa = (Observed Accuracy − Chance Accuracy)
        -------------------------------------
        (1 − Chance Accuracy)
```

---

## Interpretation scale

| Kappa   | Agreement         |
| ------- | ----------------- |
| < 0     | Worse than chance |
| 0       | Random            |
| 0.4–0.6 | Moderate          |
| 0.6–0.8 | Strong            |
| > 0.8   | Almost perfect    |

Your Kappa:

* Indicates **strong agreement**
* Confirms the model is not guessing

---

#  Sensitivity (Recall / True Positive Rate)

## What sensitivity answers

> “Out of all real Cancer cases, how many did we detect?”

---

## Formula

```
Sensitivity = TP / (TP + FN)
```

---

## Medical intuition

* High sensitivity → fewer missed patients
* Low sensitivity → dangerous in disease prediction

---

## Example

* 4 real Cancer cases
* Model detects 3
 Sensitivity = 3 / 4 = 75%

Your model:

* Missed **one Cancer case**
* Sensitivity < 100%

---

#  Specificity (True Negative Rate)

## What specificity answers

> “Out of all Normal cases, how many were correctly identified as Normal?”

---

## Formula

```
Specificity = TN / (TN + FP)
```

---

## Medical intuition

* High specificity → no false alarms
* Low specificity → unnecessary anxiety or treatment

Your model:

* Classified **all Normal cases correctly**
* Perfect specificity

---

#  Precision (Positive Predictive Value, PPV)

## What precision answers

> “When the model predicts Cancer, how often is it correct?”

---

## Formula

```
Precision = TP / (TP + FP)
```

---

## Key difference from sensitivity

| Metric      | Question                                |
| ----------- | --------------------------------------- |
| Sensitivity | Did we find all sick patients?          |
| Precision   | Are predicted sick patients truly sick? |

Your model:

* No false positives
* Precision = 100%
* Cancer predictions are **highly reliable**

---

#  Negative Predictive Value (NPV)

## What NPV answers

> “When the model predicts Normal, how often is it correct?”

---

## Formula

```
NPV = TN / (TN + FN)
```

---

## Example

* Model predicts Normal 5 times
* 1 of those is actually Cancer

 NPV = 80%

Your model:

* One false reassurance
* Important clinical consideration

---

#  Balanced Accuracy

## Why we need it

Accuracy is biased when:

* Classes are imbalanced

Balanced accuracy fixes this.

---

## Formula

```
Balanced Accuracy = (Sensitivity + Specificity) / 2
```

---

## Interpretation

* Gives **equal importance** to both classes
* Robust metric for medical datasets

Your model:

* Balanced accuracy equals overall accuracy
* Indicates balanced performance

---

# McNemar’s Test

## What McNemar’s test checks

It tests **error symmetry**.

> “Does the model make more false positives than false negatives (or vice versa)?”

---

## What it focuses on

Only these two cells matter:

* False Positives
* False Negatives

---

## Interpretation

| Result          | Meaning              |
| --------------- | -------------------- |
| Significant     | Bias in errors       |
| Not significant | Errors are symmetric |

Your result:

* No statistical evidence of biased errors
* However, small sample size limits power

---

#  Big Picture Summary (Intuition)

| Concept           | What it tells you        |
| ----------------- | ------------------------ |
| kNN               | How predictions are made |
| NIR               | Baseline performance     |
| P-value           | Is model useful?         |
| Kappa             | Agreement beyond chance  |
| Sensitivity       | Missed disease risk      |
| Specificity       | False alarm risk         |
| Precision         | Trust in positives       |
| NPV               | Trust in negatives       |
| Balanced Accuracy | Fair performance         |
| McNemar’s Test    | Error bias               |

---

#  idea

> “Accuracy tells how often the model is correct, Kappa tells whether it is better than chance, sensitivity and specificity describe clinical reliability, and NIR with p-value confirms whether the model adds real predictive value beyond naive guessing.”

