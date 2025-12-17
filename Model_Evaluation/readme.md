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



Below is a **clear, detailed, and structured explanation** of both **McNemar’s Test** and **Confidence Interval (CI)**, with **intuition, formulas, examples, and interpretation**.  
This is especially useful in **classification model comparison, medical studies, and ML evaluation**, where paired data are common.

---

# **1. McNemar’s Test**

## **What is McNemar’s Test?**

**McNemar’s test** is a **non-parametric statistical test** used to compare **two paired classifiers (or two paired methods)** on the **same dataset** when the outcome is **binary** (e.g., Yes/No, Positive/Negative).

It answers the question:

> **“Is there a statistically significant difference between two models’ error patterns on the same samples?”**



## **When Should You Use McNemar’s Test?**

Use McNemar’s test when:

Same dataset  
Same instances evaluated by **both models**  
Binary outcomes  
You care about **where the models disagree**, not just accuracy  

Common use cases:

- Comparing **two ML classifiers**
- Before–after medical treatment outcomes
- Human vs AI diagnostic decisions



## **McNemar’s Contingency Table (2 × 2)**

|                     | Model B Correct | Model B Wrong |
|---------------------|-----------------|---------------|
| **Model A Correct** | a               | b             |
| **Model A Wrong**   | c               | d             |

Where:

- **a** → both models correct  
- **d** → both models wrong  
- **b** → A correct, B wrong  
- **c** → A wrong, B correct  

👉 **Only b and c matter** for McNemar’s test.



## **Null Hypothesis (H₀)**

> **H₀:** Both models have the same error rate  
> (i.e., b = c)



## **Test Statistic (Chi-Square Form)**

$$
\chi^2 = \frac{(b - c)^2}{b + c}
$$

With **1 degree of freedom**

### Continuity-Corrected Version (recommended for small samples)

$$
\chi^2 = \frac{(|b - c| - 1)^2}{b + c}
$$



## **Decision Rule**

- Compute χ²  
- Find **p-value**  
- If **p < 0.05**, reject H₀  



## **Example**

Suppose two classifiers are tested on **100 samples**:

|                     | Model B Correct | Model B Wrong |
|---------------------|-----------------|---------------|
| **Model A Correct** | 60              | 15            |
| **Model A Wrong**   | 5               | 20            |

Here:

- b = 15  
- c = 5  

### Calculate:

$$
\chi^2 = \frac{(15 - 5)^2}{15 + 5}
= \frac{100}{20}
= 5
$$

p ≈ 0.025

### Interpretation:

 **Statistically significant difference**  
Model A performs better than Model B

---

## **Why Accuracy Alone Is Not Enough**

Two models can have the **same accuracy** but:

- Make errors on **different samples**

McNemar’s test captures this difference.

---

## **Key Assumptions**

- Paired observations  
- Binary classification  
- b + c ≥ 10 (for chi-square approximation)  

If **b + c < 10**, use **exact McNemar’s test** (binomial test).

---

## **Summary of McNemar’s Test**

| Feature    | Description      |
|------------|------------------|
| Data type  | Paired, binary   |
| Focus      | Disagreements    |
| Test type  | Non-parametric   |
| Output     | p-value          |
| Common use | Model comparison |

---

# **2. Confidence Interval (CI)**

## **What is a Confidence Interval?**

A **confidence interval (CI)** is a **range of values** that likely contains the **true population parameter** (mean, proportion, accuracy, etc.).

It answers:

> **“How precise is my estimate?”**


## **Common Confidence Levels**

| Confidence Level | Interpretation |
|------------------|----------------|
| 90% CI           | Less strict    |
| 95% CI           | Standard       |
| 99% CI           | Very strict    |

 **95% CI does NOT mean**

> “There is a 95% probability the parameter lies in this interval”

Correct meaning:

> If we repeated the experiment many times, **95% of such intervals would contain the true value**



## **General CI Formula**

$$
\text{Estimate} \pm (\text{Critical value}) \times (\text{Standard Error})
$$



## **Example 1: CI for Mean**

Suppose:

- Sample mean = 70  
- Standard deviation = 10  
- Sample size = 100  
- 95% CI → Z = 1.96  

$$
SE = \frac{10}{\sqrt{100}} = 1
$$

$$
CI = 70 \pm 1.96 \times 1
$$

$$
= (68.04,\; 71.96)
$$

### Interpretation:

We are **95% confident** the true mean lies between **68.04 and 71.96**.



## **Example 2: CI for Classification Accuracy**

Suppose:

- Accuracy = 0.85  
- Test samples = 200  

$$
SE = \sqrt{\frac{p(1 - p)}{n}}
= \sqrt{\frac{0.85 \times 0.15}{200}}
$$

$$
SE \approx 0.025
$$

$$
95\% \; CI = 0.85 \pm 1.96 \times 0.025
$$

$$
= (0.80,\; 0.90)
$$



## **Confidence Interval for Difference Between Models**

Often used **with McNemar’s test**

$$
CI = (b - c) \pm 1.96 \sqrt{b + c}
$$

If CI **does not include 0**, the difference is significant.



## **Why CI Is Important**

- Shows **uncertainty**  
- More informative than p-value alone  
- Indicates **practical significance**  
- Helps compare models robustly  



## **McNemar Test vs Confidence Interval**

| Aspect            | McNemar’s Test       | Confidence Interval  |
|-------------------|----------------------|----------------------|
| Output            | p-value              | Range                |
| Purpose           | Significance testing | Precision estimation |
| Binary decision   | Yes                  | No                   |
| Practical insight | Limited              | High                 |



## **Key Takeaway**

- **McNemar’s test** tells you **IF** two models differ  
- **Confidence interval** tells you **BY HOW MUCH** and **how reliable** the difference is  

 Best practice: **report both together**



