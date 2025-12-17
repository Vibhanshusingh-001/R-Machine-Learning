## **Cohen’s Kappa (κ) > Chance-Corrected Agreement**

Cohen’s Kappa is a **statistical measure of agreement** between **two raters (or two classifiers)** who assign **categorical labels** to the same set of items.  
It answers a critical question:

> **“How much better is the observed agreement than what we would expect purely by chance?”**

Unlike **accuracy**, Cohen’s Kappa explicitly **corrects for chance agreement**, making it especially important for **imbalanced datasets** and **classification evaluation**.



## **1. When Do We Use Cohen’s Kappa?**

Cohen’s Kappa is used when:

- Two annotators label the same data  
- OR a **machine learning classifier** is compared against **ground truth**  
- Labels are **categorical** (binary or multi-class)  
- Chance agreement is **non-negligible**

### Common Applications

- Medical diagnosis (doctor vs doctor, or model vs expert)
- NLP annotation agreement
- Image classification
- Bioinformatics classification tasks (e.g., gene function, resistance prediction)



## **2. Why Accuracy Is Not Enough**

Accuracy counts **total matches**, but:

- It **does not consider class imbalance**
- High accuracy may occur **by chance alone**

### Example (Imbalanced Data)

- 95% samples are class A  
- A model predicts **only A**  
- Accuracy = 95%  
- **But the model learned nothing**

Cohen’s Kappa penalizes this behavior.



## **3. Mathematical Definition**

Cohen’s Kappa is defined as:

$$
\kappa = \frac{P_o - P_e}{1 - P_e}
$$

Where:

| Term | Meaning |
|-----:|---------|
| $P_o$ | **Observed agreement** |
| $P_e$ | **Expected agreement by chance** |



## **4. Step-by-Step Breakdown**

### Step 1: Confusion Matrix (Binary Example)

Assume two raters classify 100 samples as **Positive (P)** or **Negative (N)**

|                | Rater B: P | Rater B: N | Total |
|----------------|-----------:|-----------:|------:|
| **Rater A: P** | 40 | 10 | 50 |
| **Rater A: N** | 5  | 45 | 50 |
| **Total**      | 45 | 55 | 100 |



### Step 2: Observed Agreement ($P_o$)

$$
P_o = \frac{\text{Agreements}}{\text{Total}}
$$

Agreements = diagonal = $40 + 45 = 85$

$$
P_o = \frac{85}{100} = 0.85
$$



### Step 3: Expected Agreement ($P_e$)

Expected agreement is calculated **from marginal probabilities**.

#### Probability both say Positive:

$$
P(P) = \left(\frac{50}{100}\right) \times \left(\frac{45}{100}\right) = 0.225
$$

#### Probability both say Negative:

$$
P(N) = \left(\frac{50}{100}\right) \times \left(\frac{55}{100}\right) = 0.275
$$

#### Total expected agreement:

$$
P_e = 0.225 + 0.275 = 0.50
$$



### Step 4: Compute Cohen’s Kappa

$$
\kappa = \frac{0.85 - 0.50}{1 - 0.50}
$$

$$
\kappa = \frac{0.35}{0.50} = 0.70
$$



## **5. Interpretation of Kappa Values**

| κ Value | Interpretation |
|--------:|----------------|
| < 0 | Worse than chance |
| 0.00–0.20 | Slight agreement |
| 0.21–0.40 | Fair agreement |
| 0.41–0.60 | Moderate agreement |
| 0.61–0.80 | Substantial agreement |
| 0.81–1.00 | Almost perfect agreement |

➡ **κ = 0.70 ⇒ Substantial agreement**



## **6. Perfect Agreement Case**

If two raters **always agree**:

- $P_o = 1$
- $P_e < 1$

$$
\kappa = \frac{1 - P_e}{1 - P_e} = 1
$$

 Perfect agreement



## **7. Random Guessing Case**

If agreement equals chance:

$$
P_o = P_e \Rightarrow \kappa = 0
$$

➡ No real agreement



## **8. Negative Kappa (Important Insight)**

If:

$$
P_o < P_e \Rightarrow \kappa < 0
$$

This means **agreement is worse than random guessing**  
(e.g., systematic disagreement)



## **9. Multi-Class Cohen’s Kappa**

Cohen’s Kappa generalizes to **multiple classes**.

### General Form:

$$
P_o = \sum_i \frac{n_{ii}}{N}
$$

$$
P_e = \sum_i \left( \frac{n_{i+}}{N} \times \frac{n_{+i}}{N} \right)
$$

Where:

- $n_{ii}$: diagonal elements  
- $n_{i+}$, $n_{+i}$: row and column totals  



## **10. Why Cohen’s Kappa Is Better Than Accuracy**

| Metric | Considers Chance? | Handles Imbalance? |
|--------|------------------|-------------------|
| Accuracy |  No |  No |
| F1-Score |  No |  Partial |
| Cohen’s Kappa |  Yes |  Yes |



## **11. Practical ML Example**

### Scenario:

- Binary classifier for disease detection  
- Dataset: 90% healthy, 10% diseased  
- Model predicts all healthy  

| Metric | Value |
|--------|-------|
| Accuracy | 90% |
| Cohen’s Kappa | ≈ 0 |

➡ Model is **useless**, despite high accuracy



## **12. When NOT to Use Cohen’s Kappa**

- Continuous outputs (regression)
- More than two raters → use **Fleiss’ Kappa**
- Ordinal labels → use **Weighted Kappa**


## **13. Weighted Cohen’s Kappa (Ordinal Classes)**

Used when **misclassification severity matters**.

Example:

- Predicting disease stage (Stage 1–4)
- Misclassifying Stage 1 as 2 is **less severe** than as 4

Weights penalize large disagreements more.


## **14. Summary**

### Key Takeaways

- Cohen’s Kappa measures **agreement beyond chance**
- Corrects accuracy’s biggest weakness
- Robust for imbalanced datasets
- Values range from **−1 to 1**
- Widely used in **ML, medicine, NLP, bioinformatics**


