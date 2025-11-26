-- SQL Code Sample for OAG Data Scientist Position
-- Author: Arjun Kumar
-- 
-- This file demonstrates SQL proficiency for investigative data analysis
-- Based on work at Rain Bird Corporation reconciling 150K+ records
-- 
-- NOTE: These queries would typically run against databases like PostgreSQL,
-- but syntax is kept compatible with most SQL dialects

-- ============================================================================
-- PART 1: DATA VALIDATION & QUALITY CHECKS
-- ============================================================================

-- Check for duplicate transaction records
-- Used at Rain Bird to identify discrepancies in order data
SELECT 
    transaction_id,
    COUNT(*) as occurrence_count,
    MIN(transaction_date) as first_occurrence,
    MAX(transaction_date) as last_occurrence
FROM transactions
GROUP BY transaction_id
HAVING COUNT(*) > 1
ORDER BY occurrence_count DESC;

-- Identify records with missing critical fields
-- Essential for producing audit-ready datasets
SELECT 
    'fee_amount' as missing_field,
    COUNT(*) as missing_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM transactions), 2) as missing_pct
FROM transactions
WHERE fee_amount IS NULL

UNION ALL

SELECT 
    'apr' as missing_field,
    COUNT(*) as missing_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM transactions), 2) as missing_pct
FROM transactions
WHERE apr IS NULL

UNION ALL

SELECT 
    'customer_id' as missing_field,
    COUNT(*) as missing_count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM transactions), 2) as missing_pct
FROM transactions
WHERE customer_id IS NULL;

-- Detect out-of-range values that may indicate data quality issues
SELECT 
    'apr_out_of_range' as validation_check,
    COUNT(*) as affected_records,
    MIN(apr) as min_value,
    MAX(apr) as max_value
FROM transactions
WHERE apr < 0 OR apr > 30

UNION ALL

SELECT 
    'negative_fees' as validation_check,
    COUNT(*) as affected_records,
    MIN(fee_amount) as min_value,
    MAX(fee_amount) as max_value
FROM transactions
WHERE fee_amount < 0;


-- ============================================================================
-- PART 2: INVESTIGATIVE QUERIES FOR HYPOTHESIS TESTING
-- ============================================================================

-- Hypothesis: Are certain customer segments charged higher fees?
-- This query tests for potential discriminatory pricing practices
WITH segment_stats AS (
    SELECT 
        CASE 
            WHEN customer_id BETWEEN 5000 AND 6000 THEN 'Segment A'
            ELSE 'Segment B'
        END as customer_segment,
        AVG(fee_amount) as avg_fee,
        STDDEV(fee_amount) as stddev_fee,
        COUNT(*) as transaction_count,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY fee_amount) as median_fee,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY fee_amount) as q25_fee,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY fee_amount) as q75_fee
    FROM transactions
    WHERE fee_amount IS NOT NULL
    GROUP BY customer_segment
)
SELECT 
    customer_segment,
    ROUND(avg_fee, 2) as avg_fee,
    ROUND(stddev_fee, 2) as stddev_fee,
    ROUND(median_fee, 2) as median_fee,
    ROUND(q75_fee - q25_fee, 2) as iqr,
    transaction_count,
    -- Calculate 95% confidence interval
    ROUND(avg_fee - (1.96 * stddev_fee / SQRT(transaction_count)), 2) as ci_lower,
    ROUND(avg_fee + (1.96 * stddev_fee / SQRT(transaction_count)), 2) as ci_upper
FROM segment_stats
ORDER BY avg_fee DESC;


-- Hypothesis: Did APR rates spike during a specific time period?
-- Useful for detecting rate manipulation or systematic changes
WITH period_comparison AS (
    SELECT 
        CASE 
            WHEN transaction_date BETWEEN '2024-06-01' AND '2024-09-30' 
            THEN 'Suspicious Period'
            ELSE 'Normal Period'
        END as time_period,
        AVG(apr) as avg_apr,
        STDDEV(apr) as stddev_apr,
        COUNT(*) as transaction_count,
        MIN(apr) as min_apr,
        MAX(apr) as max_apr
    FROM transactions
    WHERE apr IS NOT NULL
    GROUP BY time_period
)
SELECT 
    time_period,
    ROUND(avg_apr, 2) as avg_apr,
    ROUND(stddev_apr, 2) as stddev_apr,
    ROUND(min_apr, 2) as min_apr,
    ROUND(max_apr, 2) as max_apr,
    transaction_count,
    -- Calculate percent difference from normal period
    ROUND(
        (avg_apr - LAG(avg_apr) OVER (ORDER BY time_period DESC)) * 100.0 / 
        LAG(avg_apr) OVER (ORDER BY time_period DESC), 
        2
    ) as pct_change_from_normal
FROM period_comparison
ORDER BY time_period DESC;


-- ============================================================================
-- PART 3: ANOMALY DETECTION & PATTERN RECOGNITION
-- ============================================================================

-- Identify transactions with unusually high fees (potential fraud/errors)
-- Using statistical approach: fees > mean + 2*stddev
WITH fee_stats AS (
    SELECT 
        AVG(fee_amount) as mean_fee,
        STDDEV(fee_amount) as stddev_fee
    FROM transactions
    WHERE fee_amount IS NOT NULL
)
SELECT 
    t.transaction_id,
    t.customer_id,
    t.transaction_date,
    t.fee_amount,
    t.transaction_amount,
    t.apr,
    t.merchant_category,
    t.state,
    ROUND((t.fee_amount - fs.mean_fee) / fs.stddev_fee, 2) as z_score,
    ROUND(t.fee_amount - fs.mean_fee, 2) as deviation_from_mean
FROM transactions t
CROSS JOIN fee_stats fs
WHERE t.fee_amount > (fs.mean_fee + 2 * fs.stddev_fee)
ORDER BY z_score DESC
LIMIT 100;


-- Identify customers with consistently high fees (pattern detection)
-- Useful for finding systematic targeting or discrimination
WITH customer_metrics AS (
    SELECT 
        customer_id,
        COUNT(*) as num_transactions,
        AVG(fee_amount) as avg_fee,
        AVG(apr) as avg_apr,
        SUM(transaction_amount) as total_transaction_value,
        AVG(payment_delay_days) as avg_delay,
        -- Calculate what percentage of their fees are above overall median
        SUM(CASE WHEN fee_amount > (
            SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY fee_amount) 
            FROM transactions
        ) THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as pct_above_median_fee
    FROM transactions
    WHERE fee_amount IS NOT NULL
    GROUP BY customer_id
    HAVING COUNT(*) >= 5  -- Only customers with 5+ transactions
)
SELECT 
    customer_id,
    num_transactions,
    ROUND(avg_fee, 2) as avg_fee,
    ROUND(avg_apr, 2) as avg_apr,
    ROUND(total_transaction_value, 2) as total_value,
    ROUND(avg_delay, 1) as avg_delay_days,
    ROUND(pct_above_median_fee, 1) as pct_above_median_fee
FROM customer_metrics
WHERE pct_above_median_fee > 75  -- Customers paying above median 75%+ of the time
ORDER BY avg_fee DESC
LIMIT 50;


-- ============================================================================
-- PART 4: TIME SERIES ANALYSIS FOR TREND DETECTION
-- ============================================================================

-- Monthly trend analysis with year-over-year comparison
-- Similar to analyses at Rain Bird for identifying fulfillment patterns
WITH monthly_metrics AS (
    SELECT 
        DATE_TRUNC('month', transaction_date) as month,
        EXTRACT(YEAR FROM transaction_date) as year,
        EXTRACT(MONTH FROM transaction_date) as month_num,
        COUNT(*) as transaction_count,
        AVG(fee_amount) as avg_fee,
        AVG(apr) as avg_apr,
        AVG(payment_delay_days) as avg_delay,
        SUM(transaction_amount) as total_volume
    FROM transactions
    GROUP BY DATE_TRUNC('month', transaction_date),
             EXTRACT(YEAR FROM transaction_date),
             EXTRACT(MONTH FROM transaction_date)
)
SELECT 
    month,
    transaction_count,
    ROUND(avg_fee, 2) as avg_fee,
    ROUND(avg_apr, 2) as avg_apr,
    ROUND(avg_delay, 1) as avg_delay,
    ROUND(total_volume, 2) as total_volume,
    -- Calculate month-over-month change
    ROUND(
        (avg_fee - LAG(avg_fee) OVER (ORDER BY month)) * 100.0 / 
        LAG(avg_fee) OVER (ORDER BY month),
        2
    ) as mom_fee_change_pct,
    -- Calculate year-over-year change
    ROUND(
        (avg_fee - LAG(avg_fee, 12) OVER (ORDER BY month)) * 100.0 / 
        LAG(avg_fee, 12) OVER (ORDER BY month),
        2
    ) as yoy_fee_change_pct
FROM monthly_metrics
ORDER BY month DESC;


-- ============================================================================
-- PART 5: COMPLEX JOINS FOR MULTI-SOURCE RECONCILIATION
-- ============================================================================

-- Example: Reconcile transactions across multiple data sources
-- This type of query was essential at Rain Bird for the 150K+ record reconciliation
WITH source_a AS (
    SELECT 
        transaction_id,
        customer_id,
        transaction_date,
        transaction_amount as amount_source_a,
        'Source A' as data_source
    FROM transactions_source_a
),
source_b AS (
    SELECT 
        transaction_id,
        customer_id,
        transaction_date,
        transaction_amount as amount_source_b,
        'Source B' as data_source
    FROM transactions_source_b
),
reconciliation AS (
    SELECT 
        COALESCE(a.transaction_id, b.transaction_id) as transaction_id,
        COALESCE(a.customer_id, b.customer_id) as customer_id,
        COALESCE(a.transaction_date, b.transaction_date) as transaction_date,
        a.amount_source_a,
        b.amount_source_b,
        ABS(COALESCE(a.amount_source_a, 0) - COALESCE(b.amount_source_b, 0)) as amount_difference,
        CASE 
            WHEN a.transaction_id IS NULL THEN 'Missing in Source A'
            WHEN b.transaction_id IS NULL THEN 'Missing in Source B'
            WHEN ABS(a.amount_source_a - b.amount_source_b) > 0.01 THEN 'Amount Mismatch'
            ELSE 'Match'
        END as reconciliation_status
    FROM source_a a
    FULL OUTER JOIN source_b b 
        ON a.transaction_id = b.transaction_id
)
SELECT 
    reconciliation_status,
    COUNT(*) as record_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage,
    ROUND(AVG(amount_difference), 2) as avg_difference,
    ROUND(SUM(amount_difference), 2) as total_difference
FROM reconciliation
GROUP BY reconciliation_status
ORDER BY record_count DESC;


-- ============================================================================
-- PART 6: ADVANCED AGGREGATIONS FOR REPORTING
-- ============================================================================

-- Comprehensive summary by merchant category and state
-- Type of query used at Rain Bird for operational dashboards
SELECT 
    merchant_category,
    state,
    COUNT(*) as num_transactions,
    COUNT(DISTINCT customer_id) as unique_customers,
    ROUND(AVG(transaction_amount), 2) as avg_transaction_amt,
    ROUND(AVG(fee_amount), 2) as avg_fee,
    ROUND(AVG(apr), 2) as avg_apr,
    ROUND(AVG(payment_delay_days), 1) as avg_delay_days,
    -- Calculate fee-to-transaction ratio
    ROUND(AVG(fee_amount * 100.0 / NULLIF(transaction_amount, 0)), 2) as avg_fee_pct,
    -- Identify high-risk categories
    SUM(CASE WHEN payment_delay_days > 10 THEN 1 ELSE 0 END) as high_delay_count,
    ROUND(
        SUM(CASE WHEN payment_delay_days > 10 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
        2
    ) as high_delay_pct
FROM transactions
WHERE transaction_amount > 0  -- Avoid division by zero
GROUP BY merchant_category, state
HAVING COUNT(*) >= 100  -- Only include categories with sufficient data
ORDER BY avg_fee DESC;


-- ============================================================================
-- PART 7: COHORT ANALYSIS
-- ============================================================================

-- Customer cohort analysis based on first transaction date
-- Useful for detecting changes in pricing or practices over time
WITH customer_first_transaction AS (
    SELECT 
        customer_id,
        MIN(transaction_date) as first_transaction_date,
        DATE_TRUNC('quarter', MIN(transaction_date)) as cohort_quarter
    FROM transactions
    GROUP BY customer_id
),
cohort_metrics AS (
    SELECT 
        cft.cohort_quarter,
        COUNT(DISTINCT t.customer_id) as cohort_size,
        COUNT(*) as total_transactions,
        AVG(t.fee_amount) as avg_fee,
        AVG(t.apr) as avg_apr,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY t.fee_amount) as median_fee
    FROM customer_first_transaction cft
    JOIN transactions t ON cft.customer_id = t.customer_id
    GROUP BY cft.cohort_quarter
)
SELECT 
    cohort_quarter,
    cohort_size,
    total_transactions,
    ROUND(avg_fee, 2) as avg_fee,
    ROUND(avg_apr, 2) as avg_apr,
    ROUND(median_fee, 2) as median_fee,
    ROUND(total_transactions * 1.0 / cohort_size, 1) as avg_transactions_per_customer
FROM cohort_metrics
ORDER BY cohort_quarter DESC;


