---
layout: post
title: "Project 1 Q2: Survival Analysis of Telco Customer Churn"
date: 2026-04-27
---

This post records the second part of Project 1 Q2: the survival-analysis process and its main results. The analysis is based on the IBM Telco customer churn dataset and rewrites the Databricks survival-analysis workflow so that it can run locally with PySpark, Pandas, and the `lifelines` package.

The main idea is to treat customer churn as a time-to-event problem. The duration variable is `tenure`, measured in months. The event variable is `churn`, where `1.0` means that the customer churned and `0.0` means that the customer was still active at the end of the observation period. Spark was used to prepare the curated analytical table, and the survival models were fitted after converting the necessary columns to Pandas.

## Dataset Profile

The raw bronze table contains **7,043** customer records, and the curated silver table also contains **7,043** records, so the local rewrite did not drop any rows. The cleaning step found **11** blank values in `TotalCharges`; these were converted to null values before writing the local silver Parquet output. In total, the dataset contains **1,869** churned customers and **5,174** active customers.

| Churn | Customer count | Avg. tenure | Avg. monthly charges | Avg. total charges |
|---:|---:|---:|---:|---:|
| 0.0 | 5,174 | 37.57 | 61.27 | 2,555.34 |
| 1.0 | 1,869 | 17.98 | 74.44 | 1,531.80 |

This profile already shows the core business pattern. Churned customers stayed for fewer months on average and paid higher monthly charges, while retained customers accumulated higher total charges because they remained active longer.

## Kaplan-Meier Survival Curves

The first survival-analysis step is Kaplan-Meier estimation. Kaplan-Meier is a non-parametric method for estimating the probability that a customer survives beyond a given month while accounting for right-censored customers. In this dataset, the population-level survival curve does not fall below 0.5 during the observed time range, so the median survival time is **not reached within the observation window**.

![Population-level Kaplan-Meier survival curve]({{ "/assets/project1-q2/km_population_curve.svg" | relative_url }})

Subgroup Kaplan-Meier curves reveal stronger differences. Contract type is the clearest example: month-to-month customers churn much faster, while one-year and especially two-year customers remain active longer.

![Kaplan-Meier curves by contract type]({{ "/assets/project1-q2/km_contract_curve.svg" | relative_url }})

Service-protection variables show similar separation. Customers with online security or technical support tend to have better survival patterns than customers without those services.

![Kaplan-Meier curves by online-security status]({{ "/assets/project1-q2/km_online_security_curve.svg" | relative_url }})

Gender is a useful contrast because its curves are much closer together than the contract and service-protection curves.

![Kaplan-Meier curves by gender]({{ "/assets/project1-q2/km_gender_curve.svg" | relative_url }})

The visual differences are confirmed by pairwise log-rank tests. The largest test statistics are concentrated in contract type, service-protection features, and payment method.

| Variable | Group A | Group B | Chi-square | p-value |
|---|---|---|---:|---:|
| `contract` | Month-to-month | Two year | 1550.51 | 0.0 |
| `contract` | Month-to-month | One year | 926.06 | 2.12e-203 |
| `onlineSecurity` | No | Yes | 660.53 | 1.15e-145 |
| `techSupport` | No | Yes | 639.35 | 4.63e-141 |
| `paymentMethod` | Credit card (automatic) | Electronic check | 539.74 | 2.15e-119 |

These results indicate that retention timing is strongly associated with contract commitment, security/support services, and payment behavior.

## Cox Proportional Hazards Model

Kaplan-Meier curves are useful for subgroup comparison, but the next step is multivariate modeling. I fitted a Cox proportional hazards model after one-hot encoding selected categorical covariates. The model estimates multiplicative effects on the churn hazard. A hazard ratio below 1 suggests lower churn hazard, while a hazard ratio above 1 suggests higher churn hazard.

| Feature | Coef. | Hazard ratio | p-value | Lower 95% | Upper 95% |
|---|---:|---:|---:|---:|---:|
| `dependents_Yes` | -0.8138 | 0.4432 | 4.23e-40 | -0.9341 | -0.6935 |
| `internetService_DSL` | -0.0747 | 0.9280 | 0.1795 | -0.1839 | 0.0344 |
| `onlineBackup_Yes` | -0.6415 | 0.5265 | 1.02e-32 | -0.7470 | -0.5359 |
| `techSupport_Yes` | -0.9515 | 0.3861 | 1.75e-47 | -1.0804 | -0.8227 |
| `paperlessBilling_Yes` | 0.7851 | 2.1927 | 1.33e-47 | 0.6789 | 0.8913 |

The strongest protective effects in this reduced specification are `techSupport_Yes`, `dependents_Yes`, and `onlineBackup_Yes`. For example, `techSupport_Yes` has a hazard ratio of **0.3861**, indicating a much lower churn hazard than the baseline group. In contrast, `paperlessBilling_Yes` has a hazard ratio of **2.1927**, suggesting higher churn hazard after adjusting for the other included covariates. The DSL indicator is directionally protective, but its p-value is **0.1795**, so it is not statistically significant here.

![Hazard-ratio view of the fitted Cox model]({{ "/assets/project1-q2/cox_hazard_ratios.svg" | relative_url }})

## Proportional-Hazards Check

A Cox model should be followed by a proportional-hazards diagnostic. The rank-based test flags several variables with p-values below 0.05, meaning that the strict proportional-hazards assumption is violated for important covariates.

| Feature | Test statistic | p-value |
|---|---:|---:|
| `dependents_Yes` | 2.8348 | 0.0922 |
| `internetService_DSL` | 51.4960 | 7.17e-13 |
| `onlineBackup_Yes` | 70.1003 | 5.64e-17 |
| `paperlessBilling_Yes` | 13.2930 | 2.66e-4 |
| `techSupport_Yes` | 28.8698 | 7.74e-8 |

This does not make the Cox model useless, but it changes the interpretation. The fitted coefficients are still informative for prediction and directional comparison, but the flagged effects should not be treated as perfectly time-invariant hazard multipliers. In practical terms, these variables are strong predictors of retention timing, while their relative effects may change across the customer lifecycle.

## Log-Logistic AFT Model

Because the Cox assumptions are not fully satisfied, I also fitted a parametric accelerated failure time model. The local rewrite uses a log-logistic AFT model with a broader set of service and payment features. In an AFT model, exponentiated coefficients greater than 1 indicate longer expected survival time.

| Feature | Coef. | Exp(coef.) | p-value |
|---|---:|---:|---:|
| `partner_Yes` | 1.1146 | 3.0484 | 4.66e-64 |
| `multipleLines_Yes` | 0.1173 | 1.1245 | 0.0855 |
| `internetService_DSL` | -0.0364 | 0.9642 | 0.6475 |
| `onlineSecurity_Yes` | 0.8932 | 2.4431 | 2.82e-24 |
| `onlineBackup_Yes` | 0.4493 | 1.5673 | 1.04e-9 |
| `deviceProtection_Yes` | 0.2395 | 1.2706 | 0.0012 |
| `techSupport_Yes` | 0.8363 | 2.3078 | 1.05e-21 |
| `paymentMethod_Bank transfer (automatic)` | 1.3367 | 3.8065 | 1.96e-51 |
| `paymentMethod_Credit card (automatic)` | 1.4475 | 4.2525 | 4.44e-56 |

The AFT model reinforces the same business interpretation as the Kaplan-Meier and Cox stages. Partner status, online security, online backup, technical support, and automatic payment methods are associated with longer customer survival.

![Coefficient view of the fitted log-logistic AFT model]({{ "/assets/project1-q2/aft_coefficients.svg" | relative_url }})

The fitted log-logistic model produces a very large exponentiated median statistic, so the most meaningful interpretation is not the raw global magnitude. The reliable takeaway is the direction and significance of the covariate effects.

## CLV Payback Example

The final step translates survival probabilities into a customer-lifetime-value style payback table. A representative customer profile with dependents, DSL, online backup, and tech support was scored with the fitted Cox model. The month-by-month survival probabilities were multiplied by a **$30** monthly profit assumption and discounted with a **10%** annual rate.

![Cumulative net present value for the example customer profile]({{ "/assets/project1-q2/clv_cumulative_npv.svg" | relative_url }})

| Month | Survival probability | Discounted expected profit | Cumulative NPV |
|---:|---:|---:|---:|
| 1 | 1.0000 | 29.75 | 29.75 |
| 3 | 0.9954 | 29.13 | 88.39 |
| 6 | 0.9911 | 28.29 | 174.08 |
| 12 | 0.9863 | 26.79 | 338.51 |

By month 12, the cumulative net present value reaches **338.51**. Across the full exported horizon, the final cumulative NPV reaches **1610.39**.

## Conclusion

The survival-analysis workflow follows a clear sequence. Spark first builds a reliable analytical table, Kaplan-Meier curves reveal survival differences, log-rank tests quantify those differences, the Cox model estimates multivariate churn hazards, diagnostics check whether the Cox assumptions hold, the AFT model provides a parametric time-based alternative, and the CLV table turns survival probabilities into business value.

The main practical conclusion is that **contract structure**, **support and security services**, and **payment behavior** are strongly associated with customer retention timing. Month-to-month contracts and electronic checks are linked with faster churn, while longer contracts, online security, tech support, online backup, partner/dependent status, and automatic payments are linked with longer customer survival. Since several Cox proportional-hazards diagnostics are significant, the safest interpretation is that these variables are powerful predictors of retention timing rather than perfectly constant hazard multipliers over time.
