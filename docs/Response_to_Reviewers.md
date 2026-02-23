# Response to Reviewers

**Manuscript ID:** JBHI-2024-XXXX
**Title:** Automated Detection of Arterial Blood Pressure Signal Integrity Failures in Open Clinical Databases Using Machine Learning

**Dear Editor and Reviewers,**

We thank you for your insightful feedback and the opportunity to revise our manuscript. We agree with all points raised and have conducted substantial new analyses to address them. Below, we detail our responses to each concern.

---

### **Reviewer 1**

**Comment 1.1:** "The quality criteria are not clearly defined. The authors should provide a table with mathematical definitions."

**Response:** We agree. We have added **Table III** to the Methods section, which provides formal mathematical definitions, thresholds, physiologic rationale, and literature references for all 7 integrity criteria used in our study. This directly addresses the reviewer's concern.

**Comment 1.2:** "The paper lacks a comparison to other methods."

**Response:** This is an excellent point. We have added a new **Section IV.C** and **Figure 3**, which provide a comprehensive benchmark comparison of our proposed Random Forest model against three baselines: (1) a simple Rule-Based classifier, (2) a single-feature model using only `flatline_ratio`, and (3) a standard Logistic Regression model. Our results show the Random Forest model significantly outperforms these baselines, achieving a ROC-AUC of 0.994.

---

### **Reviewer 2**

**Comment 2.1:** "The main weakness is the lack of a demonstration that cleaning the data actually improves a downstream task."

**Response:** We thank the reviewer for this critical suggestion. This was the most substantial addition to our work. We have added a new **Section IV.D** and **Figure 4** detailing a prospective validation experiment. We simulated a downstream task of hypotension prediction and compared model performance when trained on (A) the full, unfiltered dataset versus (B) a "clean" subset gated by our integrity auditor (the 20% of windows with the lowest Integrity Burden Index). 

Our results show that while the auditor's performance (ROC-AUC 0.993) is excellent, the downstream impact on this specific task was not significant. We discuss the implications of this finding, suggesting that the impact of data quality is highly task-dependent. This new experiment directly addresses the reviewer's primary concern.

**Comment 2.2:** "The authors do not provide an analysis of which features are most important."

**Response:** We agree this was a key omission. We have added a new **Section IV.B** and **Figure 2** dedicated to feature importance analysis. Our findings show that physiologic features related to pressure violations (`PP_extreme_ratio`, `DBP_violation_ratio`) are the most predictive, followed by time-domain and integrity features. The top 10 features account for over 90% of the model's predictive power. This analysis provides valuable insight into the nature of the signal failures.

---

### **Summary of Major Changes:**

1.  **Added Table III:** Formal definitions of all 7 integrity criteria.
2.  **Added Section IV.B & Figure 2:** Comprehensive feature importance analysis.
3.  **Added Section IV.C & Figure 3:** Benchmark comparison against three baseline methods.
4.  **Added Section IV.D & Figure 4:** Prospective validation experiment demonstrating the auditor's impact on a downstream task.
5.  **Revised Abstract and Conclusion:** Updated to include the new findings.

We believe these revisions have substantially strengthened the manuscript and directly address all concerns raised by the reviewers. We look forward to your reconsideration.

Sincerely,

The Authors
