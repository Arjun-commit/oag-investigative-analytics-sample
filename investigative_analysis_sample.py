# Investigative Data Analysis: Consumer Transaction Anomaly Detection
# Code Sample for OAG Data Scientist Position
# Author: Arjun Kumar
# 
# This sample demonstrates investigative analytics capabilities applicable to 
# detecting deceptive practices, regulatory violations, and fraudulent patterns.
#
# REAL-WORLD CONTEXT:
# These techniques are based on actual work performed at Rain Bird Corporation 
# (June 2025 - August 2025), where I:
# - Reconciled 150K+ records from fragmented systems using Python/SQL
# - Built validation checks reducing discrepancies by 27%
# - Performed hypothesis testing to identify root causes of operational issues
# - Created automated ELT pipelines and audit-ready datasets
# 
# This sample uses simulated data to demonstrate transferable skills for OAG investigations.

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PART 1: DATA ACQUISITION & ETL WITH RELIABILITY TESTING
# ============================================================================

class DataValidator:
    """
    Validates data reliability for investigative purposes.
    Checks for anomalies, missingness, schema drift, and data quality issues.
    
    Based on validation approaches developed at Rain Bird Corporation to 
    reconcile 150K+ records and reduce data discrepancies by 27%.
    """
    
    def __init__(self, data):
        self.data = data
        self.validation_report = {}
    
    def check_missingness(self, threshold=0.05):
        """Identify columns with excessive missing values"""
        missing_pct = self.data.isnull().sum() / len(self.data)
        problematic = missing_pct[missing_pct > threshold]
        
        self.validation_report['missingness'] = {
            'problematic_columns': problematic.to_dict(),
            'total_missing_pct': self.data.isnull().sum().sum() / (len(self.data) * len(self.data.columns))
        }
        return problematic
    
    def check_duplicates(self, subset=None):
        """Identify duplicate records that may indicate data quality issues"""
        if subset:
            duplicates = self.data.duplicated(subset=subset, keep=False)
        else:
            duplicates = self.data.duplicated(keep=False)
        
        duplicate_count = duplicates.sum()
        self.validation_report['duplicates'] = {
            'count': int(duplicate_count),
            'percentage': float(duplicate_count / len(self.data) * 100)
        }
        return duplicate_count
    
    def check_value_ranges(self, column, expected_min, expected_max):
        """Validate that values fall within expected ranges"""
        out_of_range = (self.data[column] < expected_min) | (self.data[column] > expected_max)
        out_of_range_count = out_of_range.sum()
        
        if column not in self.validation_report:
            self.validation_report[column] = {}
        
        self.validation_report[column]['out_of_range'] = {
            'count': int(out_of_range_count),
            'percentage': float(out_of_range_count / len(self.data) * 100),
            'expected_range': (expected_min, expected_max),
            'actual_range': (float(self.data[column].min()), float(self.data[column].max()))
        }
        return out_of_range_count
    
    def detect_schema_drift(self, expected_columns, expected_dtypes):
        """Check for schema changes that may indicate data source issues"""
        actual_columns = set(self.data.columns)
        expected_columns_set = set(expected_columns)
        
        missing_cols = expected_columns_set - actual_columns
        extra_cols = actual_columns - expected_columns_set
        
        dtype_mismatches = []
        for col in expected_columns:
            if col in self.data.columns:
                if str(self.data[col].dtype) != expected_dtypes.get(col):
                    dtype_mismatches.append({
                        'column': col,
                        'expected': expected_dtypes.get(col),
                        'actual': str(self.data[col].dtype)
                    })
        
        self.validation_report['schema'] = {
            'missing_columns': list(missing_cols),
            'extra_columns': list(extra_cols),
            'dtype_mismatches': dtype_mismatches
        }
        
        return len(missing_cols) == 0 and len(extra_cols) == 0 and len(dtype_mismatches) == 0
    
    def generate_report(self):
        """Generate comprehensive validation report"""
        return self.validation_report


def simulate_consumer_transaction_data(n_records=50000, seed=42):
    """
    Simulates consumer transaction data similar to what might be analyzed
    in consumer protection investigations (e.g., auto lending, retail)
    
    NOTE: This is simulated data for demonstration purposes only.
    Real investigations would use actual confidential data sources.
    """
    np.random.seed(seed)
    
    # Normal transaction patterns
    dates = pd.date_range(start='2023-01-01', end='2025-10-31', freq='H')
    sample_dates = np.random.choice(dates, n_records)
    
    data = {
        'transaction_id': range(1, n_records + 1),
        'transaction_date': sample_dates,
        'customer_id': np.random.randint(1000, 9000, n_records),
        'transaction_amount': np.random.gamma(shape=2, scale=150, size=n_records),
        'apr': np.random.normal(12, 3, n_records),  # Annual Percentage Rate
        'fee_amount': np.random.gamma(shape=1.5, scale=25, size=n_records),
        'payment_delay_days': np.random.poisson(5, n_records),
        'merchant_category': np.random.choice(['Auto', 'Retail', 'Finance', 'Services'], n_records),
        'state': np.random.choice(['NY', 'CA', 'TX', 'FL', 'PA'], n_records, p=[0.3, 0.2, 0.2, 0.15, 0.15])
    }
    
    df = pd.DataFrame(data)
    
    # Inject anomalies that might indicate deceptive practices
    # 1. Excessive fees for certain customer segments (potential discrimination)
    discriminatory_segment = (df['customer_id'] > 5000) & (df['customer_id'] < 6000)
    df.loc[discriminatory_segment, 'fee_amount'] *= 2.5
    
    # 2. Unusually high APR for specific time period (potential rate manipulation)
    suspicious_period = (df['transaction_date'] >= '2024-06-01') & (df['transaction_date'] <= '2024-09-30')
    df.loc[suspicious_period, 'apr'] += 5
    
    # 3. Spike in payment delays during specific period (systematic issue)
    df.loc[suspicious_period & (df['merchant_category'] == 'Auto'), 'payment_delay_days'] *= 3
    
    # Add some missing values to simulate real-world data quality issues
    missing_indices = np.random.choice(df.index, size=int(0.02 * len(df)), replace=False)
    df.loc[missing_indices, 'fee_amount'] = np.nan
    
    return df


# ============================================================================
# PART 2: INVESTIGATIVE ANALYSIS & HYPOTHESIS TESTING
# ============================================================================

class InvestigativeAnalyzer:
    """
    Performs statistical investigations to test hypotheses about potential
    deceptive practices, discrimination, or regulatory violations.
    
    Based on hypothesis testing approaches used at Rain Bird Corporation
    to identify root causes of fulfillment delays and operational issues.
    """
    
    def __init__(self, data):
        self.data = data
        self.findings = {}
    
    def test_fee_discrimination(self, segment_a, segment_b, metric='fee_amount'):
        """
        Test hypothesis: Are fees significantly different between customer segments?
        Uses Welch's t-test (doesn't assume equal variances)
        """
        group_a = self.data.loc[segment_a, metric].dropna()
        group_b = self.data.loc[segment_b, metric].dropna()
        
        # Welch's t-test
        t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((group_a.std()**2 + group_b.std()**2) / 2)
        cohens_d = (group_a.mean() - group_b.mean()) / pooled_std
        
        # Confidence intervals for means
        ci_a = stats.t.interval(0.95, len(group_a)-1, 
                               loc=group_a.mean(), 
                               scale=stats.sem(group_a))
        ci_b = stats.t.interval(0.95, len(group_b)-1, 
                               loc=group_b.mean(), 
                               scale=stats.sem(group_b))
        
        self.findings['fee_discrimination'] = {
            'group_a_mean': float(group_a.mean()),
            'group_a_std': float(group_a.std()),
            'group_a_ci': (float(ci_a[0]), float(ci_a[1])),
            'group_b_mean': float(group_b.mean()),
            'group_b_std': float(group_b.std()),
            'group_b_ci': (float(ci_b[0]), float(ci_b[1])),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'significant': p_value < 0.05,
            'interpretation': self._interpret_effect_size(cohens_d)
        }
        
        return self.findings['fee_discrimination']
    
    def test_temporal_anomaly(self, date_col, metric, suspicious_start, suspicious_end):
        """
        Test hypothesis: Did metric significantly change during a specific time period?
        Useful for detecting rate manipulation, systematic errors, etc.
        """
        suspicious_period = (self.data[date_col] >= suspicious_start) & \
                           (self.data[date_col] <= suspicious_end)
        
        suspicious_data = self.data.loc[suspicious_period, metric].dropna()
        normal_data = self.data.loc[~suspicious_period, metric].dropna()
        
        # Welch's t-test
        t_stat, p_value = stats.ttest_ind(suspicious_data, normal_data, equal_var=False)
        
        # Calculate percentage change
        pct_change = ((suspicious_data.mean() - normal_data.mean()) / normal_data.mean()) * 100
        
        self.findings['temporal_anomaly'] = {
            'suspicious_period_mean': float(suspicious_data.mean()),
            'normal_period_mean': float(normal_data.mean()),
            'percent_change': float(pct_change),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'n_suspicious': int(len(suspicious_data)),
            'n_normal': int(len(normal_data))
        }
        
        return self.findings['temporal_anomaly']
    
    def _interpret_effect_size(self, cohens_d):
        """Interpret Cohen's d effect size"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def detect_outliers_isolation_forest(self, features, contamination=0.1):
        """
        Detect anomalous transactions using Isolation Forest
        Useful for finding deceptive practices that don't follow normal patterns
        """
        # Prepare features
        X = self.data[features].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = iso_forest.fit_predict(X_scaled)
        anomaly_scores = iso_forest.score_samples(X_scaled)
        
        # Add results to dataframe
        self.data['is_anomaly'] = anomaly_labels == -1
        self.data['anomaly_score'] = anomaly_scores
        
        anomaly_count = (anomaly_labels == -1).sum()
        
        self.findings['isolation_forest'] = {
            'n_anomalies': int(anomaly_count),
            'pct_anomalies': float(anomaly_count / len(self.data) * 100),
            'features_used': features,
            'contamination': contamination
        }
        
        return self.data[self.data['is_anomaly']]
    
    def generate_findings_report(self):
        """Generate human-readable report of statistical findings"""
        report = []
        report.append("=" * 70)
        report.append("INVESTIGATIVE ANALYSIS FINDINGS REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Fee discrimination findings
        if 'fee_discrimination' in self.findings:
            f = self.findings['fee_discrimination']
            report.append("1. FEE DISCRIMINATION ANALYSIS")
            report.append("-" * 70)
            report.append(f"Group A Mean Fee: ${f['group_a_mean']:.2f} (±${f['group_a_std']:.2f})")
            report.append(f"95% CI: (${f['group_a_ci'][0]:.2f}, ${f['group_a_ci'][1]:.2f})")
            report.append(f"")
            report.append(f"Group B Mean Fee: ${f['group_b_mean']:.2f} (±${f['group_b_std']:.2f})")
            report.append(f"95% CI: (${f['group_b_ci'][0]:.2f}, ${f['group_b_ci'][1]:.2f})")
            report.append(f"")
            report.append(f"Statistical Significance: {'YES' if f['significant'] else 'NO'} (p={f['p_value']:.4f})")
            report.append(f"Effect Size: {f['interpretation'].upper()} (Cohen's d={f['cohens_d']:.3f})")
            report.append(f"")
            
            if f['significant']:
                diff = f['group_a_mean'] - f['group_b_mean']
                report.append(f"FINDING: Group A pays ${abs(diff):.2f} {'more' if diff > 0 else 'less'} on average.")
                report.append(f"This difference is statistically significant and may warrant investigation.")
            report.append("")
        
        # Temporal anomaly findings
        if 'temporal_anomaly' in self.findings:
            f = self.findings['temporal_anomaly']
            report.append("2. TEMPORAL ANOMALY ANALYSIS")
            report.append("-" * 70)
            report.append(f"Normal Period Mean: ${f['normal_period_mean']:.2f}")
            report.append(f"Suspicious Period Mean: ${f['suspicious_period_mean']:.2f}")
            report.append(f"Percent Change: {f['percent_change']:+.2f}%")
            report.append(f"")
            report.append(f"Statistical Significance: {'YES' if f['significant'] else 'NO'} (p={f['p_value']:.4f})")
            report.append(f"")
            
            if f['significant'] and abs(f['percent_change']) > 10:
                report.append(f"FINDING: Metric increased by {f['percent_change']:.1f}% during suspicious period.")
                report.append(f"This change is statistically significant and may indicate systematic issues.")
            report.append("")
        
        # Anomaly detection findings
        if 'isolation_forest' in self.findings:
            f = self.findings['isolation_forest']
            report.append("3. ANOMALY DETECTION (ISOLATION FOREST)")
            report.append("-" * 70)
            report.append(f"Total Anomalies Detected: {f['n_anomalies']} ({f['pct_anomalies']:.2f}%)")
            report.append(f"Features Analyzed: {', '.join(f['features_used'])}")
            report.append(f"")
            report.append(f"FINDING: {f['n_anomalies']} transactions exhibit anomalous patterns.")
            report.append(f"These cases may require individual review for potential violations.")
            report.append("")
        
        report.append("=" * 70)
        report.append("UNCERTAINTY & LIMITATIONS")
        report.append("=" * 70)
        report.append("- Statistical significance does not imply causation")
        report.append("- Findings should be corroborated with additional evidence")
        report.append("- Anomalies may have legitimate explanations requiring investigation")
        report.append("- Confidence intervals represent 95% confidence level")
        report.append("")
        
        return "\n".join(report)



# ============================================================================
# PART 3: MAIN EXECUTION & DEMONSTRATION
# ============================================================================

def main():
    """
    Main execution demonstrating complete investigative workflow:
    1. Data acquisition and validation
    2. Statistical hypothesis testing
    3. Anomaly detection
    4. Findings communication
    """
    
    print("=" * 70)
    print("INVESTIGATIVE DATA ANALYSIS DEMONSTRATION")
    print("Arjun Kumar - OAG Data Scientist Application")
    print("=" * 70)
    print()
    
    # Step 1: Generate simulated data
    print("Step 1: Acquiring and loading data...")
    df = simulate_consumer_transaction_data(n_records=50000)
    print(f"✓ Loaded {len(df):,} transaction records")
    print()
    
    # Step 2: Data validation (Rain Bird approach)
    print("Step 2: Validating data reliability...")
    validator = DataValidator(df)
    
    # Check for missing values
    missing = validator.check_missingness(threshold=0.05)
    if len(missing) > 0:
        print(f"⚠ Found {len(missing)} columns with excessive missing values")
    else:
        print("✓ No excessive missing values detected")
    
    # Check for duplicates
    dup_count = validator.check_duplicates(subset=['transaction_id'])
    if dup_count > 0:
        print(f"⚠ Found {dup_count} duplicate transaction IDs")
    else:
        print("✓ No duplicate transactions detected")
    
    # Validate value ranges
    validator.check_value_ranges('apr', expected_min=0, expected_max=30)
    validator.check_value_ranges('fee_amount', expected_min=0, expected_max=200)
    
    print("✓ Data validation complete")
    print()
    
    # Step 3: Investigative analysis (Rain Bird hypothesis testing approach)
    print("Step 3: Conducting investigative analysis...")
    analyzer = InvestigativeAnalyzer(df)
    
    # Test for fee discrimination
    print("\n→ Testing for fee discrimination between customer segments...")
    segment_a = (df['customer_id'] >= 5000) & (df['customer_id'] < 6000)
    segment_b = (df['customer_id'] < 5000)
    fee_test = analyzer.test_fee_discrimination(segment_a, segment_b)
    
    if fee_test['significant']:
        print(f"  ✓ SIGNIFICANT finding: {fee_test['interpretation'].upper()} effect detected")
    else:
        print(f"  ✗ No significant discrimination detected")
    
    # Test for temporal anomalies
    print("\n→ Testing for temporal anomalies in APR...")
    temporal_test = analyzer.test_temporal_anomaly(
        'transaction_date', 'apr',
        suspicious_start='2024-06-01',
        suspicious_end='2024-09-30'
    )
    
    if temporal_test['significant']:
        print(f"  ✓ SIGNIFICANT finding: {temporal_test['percent_change']:.1f}% change detected")
    else:
        print(f"  ✗ No significant temporal anomaly detected")
    
    # Anomaly detection
    print("\n→ Running anomaly detection (Isolation Forest)...")
    anomalies = analyzer.detect_outliers_isolation_forest(
        features=['transaction_amount', 'apr', 'fee_amount', 'payment_delay_days'],
        contamination=0.05
    )
    print(f"  ✓ Detected {len(anomalies):,} anomalous transactions")
    print()
    
    # Step 4: Generate findings report
    print("Step 4: Generating findings report for legal team...")
    report = analyzer.generate_findings_report()
    print("\n" + report)
    


if __name__ == "__main__":
    main()
