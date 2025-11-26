# Investigative Data Analysis Code Sample
## OAG Data Scientist Position Application

**Applicant:** Arjun Kumar  
**Position:** Data Scientist (Reference No: RAD_NYC_DAS_6412)  
**GitHub:** github.com/Arjun-commit

---

## Overview

This code sample demonstrates investigative data analysis capabilities directly applicable to the New York State Attorney General's Office work in detecting deceptive practices, consumer fraud, and regulatory violations.

**Real-World Foundation:** The techniques shown here are based on actual work performed at **Rain Bird Corporation (June 2025 - August 2025)**, where I:
- Reconciled 150K+ records from fragmented systems using Python/SQL
- Built validation checks that reduced data discrepancies by 27%
- Performed hypothesis testing to identify root causes of fulfillment delays
- Created automated ELT pipelines that decreased reporting time by 42%
- Developed clear summaries and dashboards for non-technical teams

---

## What This Sample Demonstrates

### Core OAG Job Requirements

| Job Requirement | Demonstrated In Code |
|----------------|----------------------|
| Testing legal hypotheses using data | `InvestigativeAnalyzer.test_fee_discrimination()` |
| Data acquisition from multiple sources | `simulate_consumer_transaction_data()` |
| Testing data for reliability | `DataValidator` class with comprehensive checks |
| Developing appropriate methodologies | Statistical tests, anomaly detection algorithms |
| Quantifying uncertainty | Confidence intervals, p-values, effect sizes |
| Communicating findings clearly | `generate_findings_report()` method |
| Developing predictive models | `detect_outliers_isolation_forest()` |
| Programming in Python | Entire codebase |
| Advanced SQL queries | `sql_style_analysis()` function |

### Technical Skills Showcased

- **Python Libraries:** pandas, NumPy, scikit-learn, scipy, matplotlib, seaborn
- **Statistical Methods:** Welch's t-test, hypothesis testing, effect size calculation, confidence intervals
- **Machine Learning:** Isolation Forest for anomaly detection, StandardScaler for feature engineering
- **Data Validation:** Missing data checks, duplicate detection, schema drift detection, range validation
- **SQL-Equivalent Operations:** Complex GROUP BY, aggregations, window functions using pandas
- **Reproducibility:** Clear documentation, modular code, seed setting for reproducibility

---

## Code Structure

```
investigative_analysis_sample.py
├── DataValidator class
│   ├── check_missingness()           # Identify columns with missing data
│   ├── check_duplicates()            # Detect duplicate records
│   ├── check_value_ranges()          # Validate data falls within expected ranges
│   ├── detect_schema_drift()         # Check for schema changes
│   └── generate_report()             # Comprehensive validation summary
│
├── InvestigativeAnalyzer class
│   ├── test_fee_discrimination()     # Statistical test for pricing disparities
│   ├── test_temporal_anomaly()       # Detect suspicious time-based patterns
│   ├── detect_outliers_isolation_forest()  # ML-based anomaly detection
│   └── generate_findings_report()    # Non-technical report for legal teams
│
└── main()                            # Complete investigative workflow
```

---

## How to Run

### Prerequisites

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn
```

### Execution

```bash
python investigative_analysis_sample.py
```

### Expected Output

The script will demonstrate a complete investigative workflow:
1. Data acquisition and loading (50,000 simulated transactions)
2. Comprehensive data validation checks
3. Statistical hypothesis testing for discrimination and temporal anomalies
4. Machine learning-based anomaly detection
5. Clear, non-technical findings report suitable for legal teams
6. SQL-style aggregations and summaries

---

## Simulated Investigation Scenario

The code analyzes consumer transaction data to detect:

1. **Fee Discrimination:** Are certain customer segments charged significantly higher fees?
   - Uses Welch's t-test to compare fee distributions
   - Calculates Cohen's d for effect size
   - Provides 95% confidence intervals

2. **Temporal Rate Manipulation:** Did APR rates spike during a specific period?
   - Compares rates before, during, and after suspicious period
   - Quantifies percentage change and statistical significance

3. **Anomalous Transactions:** Which transactions exhibit unusual patterns?
   - Uses Isolation Forest to detect multivariate outliers
   - Identifies transactions requiring individual investigation

---

## Applicability to OAG Cases

This methodology is directly applicable to OAG investigations such as:

| OAG Case Type | Relevant Technique |
|--------------|-------------------|
| Auto lender deceptive practices | Fee discrimination testing |
| Predatory lending detection | Temporal anomaly detection, rate analysis |
| Consumer fraud | Anomaly detection, pattern recognition |
| FCC fake comments analysis | Duplicate detection, entity resolution |
| Affordable housing kickback schemes | Transaction anomaly detection |
| Systematic discrimination | Statistical hypothesis testing |

---

## Data Confidentiality Note

This sample uses **simulated data** to respect confidentiality agreements with previous employers. The data structure, complexity, and analytical challenges mirror real datasets I've worked with, but contain no actual confidential information.

In a real OAG investigation, this code would:
- Connect to actual data sources (databases, APIs, document repositories)
- Include additional validation for data governance and chain of custody
- Integrate with secure computing environments
- Generate audit trails for court submissions

---

## Key Differentiators

### Why This Sample Aligns with OAG Needs

1. **Investigation-Focused:** Unlike typical data science portfolios focused on prediction accuracy, this emphasizes **finding truth** and **supporting legal arguments**

2. **Uncertainty Quantification:** Every finding includes confidence intervals and p-values—critical for court-defensible analysis

3. **Non-Technical Communication:** Reports are written for attorneys, not data scientists

4. **Data Reliability First:** Extensive validation before analysis—inspired by my Rain Bird experience handling inconsistent multi-source data

5. **Reproducibility:** Clear documentation, modular design, and deterministic results for audit trails

---

## Extensions & Additional Capabilities

While not shown in this sample due to length constraints, I can also demonstrate:

- **Natural Language Processing:** Entity extraction, document classification (from Origins Tribe experience)
- **Cloud Infrastructure:** AWS S3/EC2, distributed computing (listed in resume skills)
- **Advanced SQL:** Window functions, CTEs, query optimization
- **Privacy-Preserving ML:** Differential privacy, federated learning (from Alchromist research)
- **Database Design:** Schema design, normalization, indexing strategies
- **Version Control:** Git workflows, branching strategies, code review

---

## Contact

**Arjun Kumar**  
Brooklyn, NY  
Email: arjhari95@gmail.com  
LinkedIn: linkedin.com/in/arjun-kumar-cse  
GitHub: github.com/Arjun-commit

---

## Acknowledgments

The analytical approaches demonstrated here synthesize techniques from:
- **Rain Bird Corporation:** Data reconciliation, validation, and hypothesis testing
- **Alchromist LLC:** Anomaly detection, statistical investigations, uncertainty quantification
- **Origins Tribe:** ETL pipeline design, reliability testing, stakeholder communication

All techniques have been refined through real-world application and are ready to support OAG's mission of protecting New Yorkers through data-driven investigations.
