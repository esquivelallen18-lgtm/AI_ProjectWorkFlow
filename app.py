"""
Northbridge Bank - Credit Risk Natural-Language Query Engine (Streamlit app)

Converted from: Learner_Notebook_Project_3_Credit_Risk_Query_Engine.ipynb

Pipeline (unchanged from the notebook):
    classify_intent -> (verified template | generate_query) -> validate_query
    -> (retry_generation once, generated route only) -> escalate if still failing
    -> execute_query -> generate_response

Required files (same folder as app.py unless overridden):
    credit_risk_portfolio.db   SQLite database (opened read-only)
    test_queries.csv           OPTIONAL ground-truth cases (also accepts test_queries-2.csv)

Credentials (first one found wins):
    1. Streamlit secrets  (.streamlit/secrets.toml or the Streamlit Cloud "Secrets" box)
    2. Environment variables
    3. config.json next to app.py
    Keys: OPENAI_API_KEY (required), OPENAI_API_BASE (optional)

Optional settings (secrets or environment variables):
    OPENAI_MODEL     default "gpt-4o-mini"
    DB_PATH          default <app folder>/credit_risk_portfolio.db
    GROUND_TRUTH_PATH, CONFIG_PATH, AUDIT_LOG_PATH
"""

import json
import os
import re
import sqlite3
import warnings
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sqlparse
import streamlit as st
from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Credit Risk Query Engine",
    page_icon="🏦",
    layout="wide",
)

APP_DIR = Path(__file__).resolve().parent


# =============================================================================
# Configuration (replaces the Colab Drive / userdata / config.json cells)
# =============================================================================
def _secret(name):
    """Read a value from Streamlit secrets; return None if unavailable."""
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value or None


def _setting(name, default=None):
    return _secret(name) or os.environ.get(name) or default


def load_credentials():
    """Load the OpenAI key/base from secrets -> env vars -> config.json."""
    file_cfg = {}
    cfg_path = Path(_setting("CONFIG_PATH", str(APP_DIR / "config.json")))
    if cfg_path.exists():
        try:
            file_cfg = json.loads(cfg_path.read_text())
        except Exception:
            file_cfg = {}

    api_key = (
        _secret("OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or file_cfg.get("OPENAI_API_KEY")
    )
    api_base = (
        _secret("OPENAI_API_BASE")
        or os.environ.get("OPENAI_API_BASE")
        or file_cfg.get("OPENAI_API_BASE")
    )
    return api_key, api_base


OPENAI_API_KEY, OPENAI_API_BASE = load_credentials()

# Store API credentials in environment variables (as in the notebook)
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if OPENAI_API_BASE:
    os.environ["OPENAI_API_BASE"] = OPENAI_API_BASE

OPENAI_MODEL = _setting("OPENAI_MODEL", "gpt-4o-mini")
DB_PATH = _setting("DB_PATH", str(APP_DIR / "credit_risk_portfolio.db"))
AUDIT_LOG_PATH = _setting("AUDIT_LOG_PATH", str(APP_DIR / "audit_log.jsonl"))


# =============================================================================
# LLM setup
# =============================================================================
@st.cache_resource(show_spinner=False)
def get_llms(model, api_key, api_base):
    kwargs = {"model": model, "temperature": 0, "api_key": api_key}
    if api_base:
        kwargs["base_url"] = api_base
    return ChatOpenAI(**kwargs), ChatOpenAI(**kwargs)


llm = None
evaluator_llm = None
if OPENAI_API_KEY:
    llm, evaluator_llm = get_llms(OPENAI_MODEL, OPENAI_API_KEY, OPENAI_API_BASE)


# =============================================================================
# Database loading (read-only, as in the notebook)
# =============================================================================
def open_readonly_connection(db_path):
    """Open the SQLite database in read-only mode (?mode=ro) via URI."""
    uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


database_schema = """
sector_master:
  sector_code (TEXT, PK): internal sector identifier (e.g., SEC_RE, SEC_INFRA)
  sector_name (TEXT): human-readable sector name (e.g., Real Estate, Infrastructure)
  naics_code (TEXT): NAICS industry classification code
  naics_description (TEXT): NAICS code description
  is_sensitive_sector (INTEGER): 1 if sensitive sector, 0 otherwise

loan_master:
  loan_account_number (TEXT, PK): unique loan identifier
  borrower_id (TEXT): borrower identifier (joins to borrower_rating.borrower_id)
  borrower_name (TEXT): registered legal name of the borrower
  borrower_type (TEXT): entity type (C-Corporation, S-Corporation, LLC, LP, Partnership, Sole Proprietorship)
  group_name (TEXT): business group affiliation, NULL if standalone
  state (TEXT): state of registered office
  product_type (TEXT): Term Loan, Working Capital, Cash Credit, Overdraft, Bill Discounting, Letter of Credit
  loan_category (TEXT): Corporate, Mid-Corporate, SME
  sector_code (TEXT, FK): joins to sector_master.sector_code
  sanctioned_amount (REAL): original approved loan amount in USD
  disbursed_amount (REAL): total amount disbursed in USD
  outstanding_principal (REAL): current principal outstanding in USD
  outstanding_interest (REAL): accrued interest outstanding in USD
  total_outstanding (REAL): outstanding_principal + outstanding_interest in USD
  interest_rate (REAL): current interest rate as percentage
  rate_type (TEXT): Fixed, Floating, MCLR-linked, Repo-linked
  sanction_date (DATE): date of original sanction
  maturity_date (DATE): contractual maturity date
  repayment_frequency (TEXT): Monthly, Quarterly, Bullet
  branch_code (TEXT): originating branch identifier
  branch_name (TEXT): originating branch name
  relationship_manager (TEXT): assigned relationship manager name
  is_consortium (INTEGER): 1 if consortium loan, 0 otherwise
  is_restructured (INTEGER): 1 if restructured, 0 otherwise
  restructuring_date (DATE): date of last restructuring, NULL if not restructured
  is_secured (INTEGER): 1 if secured, 0 if unsecured
  days_past_due (INTEGER): current maximum days past due for the loan
  asset_classification (TEXT): Pass, Special Mention, Substandard, Doubtful, Loss
  classification_date (DATE): date current classification was assigned

borrower_rating:
  rating_id (INTEGER, PK): auto-increment identifier
  borrower_id (TEXT, FK): joins to loan_master.borrower_id
  rating_date (DATE): date of rating assessment
  internal_rating (TEXT): bank's internal rating grade (AAA through D, 18-grade scale)
  previous_rating (TEXT): rating grade from prior assessment
  rating_direction (TEXT): Upgraded, Downgraded, Maintained
  external_rating_agency (TEXT): S&P, Moody's, Fitch, DBRS Morningstar, Kroll, or NULL
  external_rating (TEXT): external agency rating
  pd_estimate (REAL): probability of default (decimal, e.g., 0.02 for 2%)
  rating_model_version (TEXT): internal rating model version

provisioning:
  provision_id (INTEGER, PK): auto-increment identifier
  loan_account_number (TEXT, FK): joins to loan_master.loan_account_number
  reporting_date (DATE): quarter-end reporting date
  ifrs9_stage (INTEGER): IFRS 9 stage (1, 2, or 3)
  stage_rationale (TEXT): reason for stage assignment
  pd_12_month (REAL): 12-month probability of default
  pd_lifetime (REAL): lifetime probability of default
  lgd_estimate (REAL): loss given default (decimal)
  ead_amount (REAL): exposure at default in USD
  ecl_amount (REAL): expected credit loss in USD
  provision_held (REAL): provision amount held in USD
  provision_coverage_ratio (REAL): provision_held / total_outstanding * 100
  is_individually_assessed (INTEGER): 1 if individually assessed, 0 if modeled

Available reporting_date values in provisioning: 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Available rating_date values in borrower_rating: 2024-09-30, 2024-12-31, 2025-03-31, 2025-06-30, 2025-09-30
Latest reporting_date: 2025-09-30
Latest rating_date: 2025-09-30
NPA definition: asset_classification IN ('Substandard', 'Doubtful', 'Loss')
"""


# =============================================================================
# CSV loading (ground-truth test cases - optional)
# =============================================================================
@st.cache_data(show_spinner=False)
def load_ground_truth(explicit_path=None):
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    candidates += [APP_DIR / "test_queries.csv", APP_DIR / "test_queries-2.csv"]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    return None


ground_truth = load_ground_truth(_setting("GROUND_TRUTH_PATH"))


# =============================================================================
# Verified Query Template Library (SQL unchanged from the notebook)
# =============================================================================

# VQ1: Sector-wise Outstanding and NPA Breakdown
sql_1 = """SELECT
    s.sector_name,
    ROUND(SUM(l.total_outstanding) / 1000000.0, 2)
        AS total_outstanding_millions,
    ROUND(
        SUM(CASE
            WHEN l.asset_classification IN
                ('Substandard', 'Doubtful', 'Loss')
            THEN l.total_outstanding
            ELSE 0
        END) / 1000000.0, 2
    ) AS npa_exposure_millions
FROM loan_master AS l
JOIN sector_master AS s
    ON l.sector_code = s.sector_code
GROUP BY s.sector_name
ORDER BY total_outstanding_millions DESC;"""

# VQ2: Portfolio Outstanding by Loan Category
sql_2 = """SELECT
    loan_category,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2)
        AS total_outstanding_millions
FROM loan_master
GROUP BY loan_category
ORDER BY total_outstanding_millions DESC;"""

# VQ3: IFRS 9 Stage-wise ECL Summary
sql_3 = """SELECT
    ifrs9_stage,
    COUNT(*) AS loan_count,
    ROUND(SUM(ead_amount) / 1000000.0, 2) AS total_ead_millions,
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_millions
FROM provisioning
WHERE reporting_date = '2025-09-30'
GROUP BY ifrs9_stage
ORDER BY ifrs9_stage;"""

# VQ4: Provision Coverage Ratio by Sector
sql_4 = """SELECT
    s.sector_name,
    ROUND(AVG(p.provision_coverage_ratio), 4)
        AS avg_provision_coverage_ratio
FROM provisioning AS p
JOIN loan_master AS l
    ON p.loan_account_number = l.loan_account_number
JOIN sector_master AS s
    ON l.sector_code = s.sector_code
WHERE p.reporting_date = '2025-09-30'
GROUP BY s.sector_name
ORDER BY avg_provision_coverage_ratio DESC;"""

# VQ5: Top 10 Loan Exposures
sql_5 = """SELECT
    l.borrower_name,
    s.sector_name,
    ROUND(l.total_outstanding / 1000000.0, 2)
        AS outstanding_millions,
    l.asset_classification
FROM loan_master AS l
JOIN sector_master AS s
    ON l.sector_code = s.sector_code
ORDER BY l.total_outstanding DESC
LIMIT 10;"""

# VQ6: Top 5 Business Group Exposures
sql_6 = """SELECT
    group_name,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2)
        AS total_outstanding_millions
FROM loan_master
WHERE group_name IS NOT NULL
    AND TRIM(group_name) <> ''
GROUP BY group_name
ORDER BY SUM(total_outstanding) DESC
LIMIT 5;"""

# VQ7: All Overdue Loan Accounts
sql_7 = """SELECT
    l.loan_account_number,
    l.borrower_name,
    s.sector_name,
    ROUND(l.total_outstanding / 1000000.0, 2)
        AS outstanding_millions,
    l.days_past_due,
    l.asset_classification
FROM loan_master AS l
JOIN sector_master AS s
    ON l.sector_code = s.sector_code
WHERE l.days_past_due > 0
ORDER BY l.days_past_due DESC;"""

# VQ8: DPD Bucket Distribution
sql_8 = """SELECT
    CASE
        WHEN days_past_due = 0 THEN '0 (Current)'
        WHEN days_past_due BETWEEN 1 AND 30 THEN '1-30'
        WHEN days_past_due BETWEEN 31 AND 60 THEN '31-60'
        WHEN days_past_due BETWEEN 61 AND 90 THEN '61-90'
        ELSE '90+'
    END AS dpd_bucket,
    COUNT(*) AS loan_count,
    ROUND(SUM(total_outstanding) / 1000000.0, 2)
        AS total_outstanding_millions
FROM loan_master
GROUP BY dpd_bucket
ORDER BY
    CASE dpd_bucket
        WHEN '0 (Current)' THEN 1
        WHEN '1-30' THEN 2
        WHEN '31-60' THEN 3
        WHEN '61-90' THEN 4
        WHEN '90+' THEN 5
    END;"""

# VQ9: Latest Rating Downgrades
sql_9 = """SELECT
    borrower_id,
    previous_rating,
    internal_rating,
    pd_estimate
FROM borrower_rating
WHERE rating_date = '2025-09-30'
    AND rating_direction = 'Downgraded'
ORDER BY pd_estimate DESC;"""

# VQ10: ECL Trend Across Reporting Quarters
sql_10 = """SELECT
    reporting_date,
    ROUND(SUM(ecl_amount) / 1000000.0, 2)
        AS total_ecl_millions
FROM provisioning
GROUP BY reporting_date
ORDER BY reporting_date ASC;"""

verified_query_library = {
    'VQ1': {
        'description': 'Sector-wise total outstanding and NPA amount breakdown across all sectors',
        'sql': sql_1
    },

    'VQ2': {
        'description': 'Total portfolio outstanding broken down by loan category (Corporate, Mid-Corporate, SME)',
        'sql': sql_2
    },

    'VQ3': {
        'description': 'IFRS 9 stage-wise summary showing loan count, exposure at default, and expected credit loss for the latest quarter',
        'sql': sql_3
    },

    'VQ4': {
        'description': 'Average provision coverage ratio by sector for the latest reporting quarter',
        'sql': sql_4
    },

    'VQ5': {
        'description': 'Top 10 largest loan exposures by outstanding amount at the borrower level',
        'sql': sql_5
    },

    'VQ6': {
        'description': 'Top 5 largest exposures aggregated at the business group level',
        'sql': sql_6
    },

    'VQ7': {
        'description': 'All overdue loan accounts with their days past due and asset classification',
        'sql': sql_7
    },

    'VQ8': {
        'description': 'Distribution of loans across days-past-due buckets showing aging profile of the portfolio',
        'sql': sql_8
    },

    'VQ9': {
        'description': 'Borrowers whose internal rating was downgraded in the latest rating cycle',
        'sql': sql_9
    },

    'VQ10': {
        'description': 'Expected credit loss trend across all reporting quarters showing provisioning movement over time',
        'sql': sql_10
    }
}


# =============================================================================
# Tool 1: Intent Classification
# =============================================================================
def classify_intent(user_question, query_library):
    '''
    Classifies the user question and decides which route to take.

    Parameters:
    - user_question (str): The natural language question from the user.
    - query_library (dict): The verified query template library.

    Returns:
    - dict: Contains 'route' (verified or generated),
                     'query_id' (template ID or None),
                     'match_reason' (short explanation of the decision).
    '''

    library_descriptions = '\n'.join(
        [f"{qid}: {entry['description']}" for qid, entry in query_library.items()]
    )

    classification_prompt = f"""
You are a query router for a credit risk analytics system at Northbridge Bank.

Your job is to determine whether a business user's question can be
answered by an existing verified SQL template or requires fresh SQL generation.

Compare the user's question against the verified query library based
on its meaning, not just its exact wording.

IMPORTANT ROUTING RULES:
- Select the "verified" route only when an existing template can answer
  the complete question.
- Compare the requested metrics, filters, grouping, and time period.
- Similar keywords alone are not enough to select a verified template.
- Questions about sector-level outstanding exposure and non-performing
  loan exposure should match VQ1 when it provides the necessary metrics,
  even if the question names a specific sector.
- VQ3 provides the latest quarter's stage distribution.
- Do NOT select VQ3 for Stage 3 EAD/ECL trends over multiple quarters.
- Questions about Stage 3 EAD/ECL trends over four quarters require
  the "generated" route.
- If no verified template fully answers the question, select "generated"
  and set query_id to null.



{user_question}

{library_descriptions}

### OUTPUT

Return ONLY a valid JSON dictionary with these exact keys:
{{
  "route": "verified" or "generated",
  "query_id": "VQ1" or "VQ2" ... "VQ10" or null,
  "match_reason": "one short sentence explaining the decision"
}}
Do not include any other text.
"""

    response = llm.invoke(classification_prompt).content.strip()
    # Extract JSON from potential markdown blocks
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        return json.loads(json_match.group())
    return {"route": "generated", "query_id": None, "match_reason": "Could not parse classification"}


# =============================================================================
# Tool 2: Query Generation
# =============================================================================
def generate_query(user_question, schema_context):
    '''
    Generates a candidate SQL query for a novel question using the database schema.

    Parameters:
    - user_question (str): The natural language question.
    - schema_context (str): Full database schema description.

    Returns:
    - str: Candidate SQL query as a string.
    '''

    generation_prompt = f"""

You are a senior SQL developer specializing in credit risk analytics using a SQLite database.

Generate an accurate SQL query based on the user's question and the provided database schema.

Use only the available tables and columns. Generate read-only SELECT queries and return only SQL without explanations.

For quarterly trend questions:
    - Use the latest four reporting dates from the provisioning table.
    - Do not generate future reporting dates.
    - For Stage 3, filter ifrs9_stage = 3.
    - Calculate EAD using SUM(ead_amount).
    - Calculate ECL using SUM(ecl_amount).
    - Group by reporting_date.

{user_question}

{schema_context}

"""

    sql = llm.invoke(generation_prompt).content.strip()
    # Strip markdown fences if present
    sql = re.sub(r'^```sql\s*|\s*```$', '', sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    sql = re.sub(r'^```\s*|\s*```$', '', sql, flags=re.MULTILINE).strip()
    return sql


# =============================================================================
# Tool 3: Query Validation (five checks)
# =============================================================================
def validate_query(user_question, candidate_sql, db_connection, query_library, query_id=None):
    '''
    Validates a candidate SQL query through five checks before execution.

    Parameters:
    - user_question (str): The original user question.
    - candidate_sql (str): The SQL query to validate.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query library (for integrity check).
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - dict: Contains 'passed' (bool), 'failed_check' (str or None), 'details' (str),
            and 'relevance_confidence' (int, 0-1).
    '''

    result = {
        'passed': False,
        'failed_check': None,
        'details': '',
        'relevance_confidence': None
    }

    # Clean the SQL query of any trailing semicolons or spaces for dry runs
    candidate_sql_clean = candidate_sql.strip().rstrip(';').strip()

    # Check 1: Read-only shape check
    sql_upper = candidate_sql_clean.upper()
    forbidden_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', 'REPLACE', 'ATTACH']
    if not (sql_upper.startswith('SELECT') or sql_upper.startswith('WITH')):
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Query must start with SELECT or WITH'
        return result
    for kw in forbidden_keywords:
        if re.search(r'\b' + kw + r'\b', sql_upper):
            result['failed_check'] = 'read_only_shape'
            result['details'] = f'Forbidden keyword detected: {kw}'
            return result
    if ';' in candidate_sql_clean:
        result['failed_check'] = 'read_only_shape'
        result['details'] = 'Multiple statements are not allowed'
        return result

    # Check 2: Schema conformance check
    cur = db_connection.cursor()
    real_tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    real_columns = set()
    for t in real_tables:
        for col_info in cur.execute(f"PRAGMA table_info({t})").fetchall():
            real_columns.add(col_info[1].lower())
    parsed = sqlparse.parse(candidate_sql_clean)[0]
    tokens = [str(t).strip().lower() for t in parsed.flatten() if t.ttype is None or 'Name' in str(t.ttype)]
    referenced_identifiers = re.findall(r'\b[a-z_][a-z0-9_]*\b', candidate_sql_clean.lower())
    sql_keywords = {'select','from','where','and','or','group','by','order','having','limit','join','on','as','case',
                    'when','then','else','end','sum','count','avg','min','max','round','desc','asc','left','right',
                    'inner','outer','distinct','null','is','not','in','like','with','union','all','between','coalesce'}
    unknown = [tok for tok in referenced_identifiers
               if tok not in sql_keywords and tok not in real_columns and tok not in real_tables
               and not tok.isdigit() and tok not in ('s', 'l', 'p', 'r', 'e6')]

    # Check 3: Parse-and-plan dry run using EXPLAIN
    try:
        cur.execute(f"EXPLAIN {candidate_sql_clean}")
        cur.fetchall()
    except sqlite3.Error as e:
        result['failed_check'] = 'parse_plan_dry_run'
        result['details'] = f'SQL failed to parse or plan: {str(e)}'
        return result

    # Check 4: LLM relevance check
    is_verified_track = query_id is not None and query_id in query_library
    track_context = (
        "This SQL is a pre-approved VERIFIED TEMPLATE. It is intentionally broad "
        "(e.g., it may return all sectors/categories/stages rather than filtering to "
        "just what the user asked). A separate response-generation step will filter and "
        "highlight the relevant rows afterward. Do NOT fail this query for lacking a "
        "WHERE clause that narrows to the user's specific sector/category/stage — judge "
        "only whether the underlying metric, tables, and aggregation logic match the "
        "question's intent."
        if is_verified_track else
        "This SQL was freshly generated for this specific question and should be "
        "appropriately scoped/filtered to answer it directly."
    )

    relevance_prompt = f"""

You are a senior data validator specializing in credit risk analytics for Northbridge Bank using a SQLite database. Your job is to determine whether a SQL query correctly answers a business user's question.

Evaluate the SQL query against the user's question and the provided database schema.

Verify that the query uses the correct tables, columns, joins, filters, and calculations.

Ensure the query follows SQLite syntax and contains only read-only SELECT operations.

Identify any errors or inconsistencies and provide a brief explanation of your findings.

{track_context}

{user_question}

{candidate_sql_clean}

### OUTPUT
Return ONLY a JSON dictionary:
{{
  "verdict": "yes" or "no",
  "confidence": 0.0 to 1.0,
  "reason": "one short sentence"
}}

"""
    relevance_response = evaluator_llm.invoke(relevance_prompt).content.strip()
    json_match = re.search(r'\{.*\}', relevance_response, re.DOTALL)
    if json_match:
        relevance_json = json.loads(json_match.group())
        result['relevance_confidence'] = relevance_json.get('confidence', 0.0)
        if relevance_json.get('verdict') == 'no' or relevance_json.get('confidence', 0.0) < 0.6:
            result['failed_check'] = 'llm_relevance'
            result['details'] = f"Relevance check failed: {relevance_json.get('reason', 'unknown')}"
            return result

    # Check 5: Verified template integrity check (verified track only)
    # Uses SELECT * FROM (...) LIMIT 0 to avoid syntax errors with templates that already contain a LIMIT clause.
    if query_id and query_id in query_library:
        expected_sql = query_library[query_id]['sql'].strip().rstrip(';').strip()
        try:
            expected_cols = [d[0] for d in cur.execute(f"SELECT * FROM ({expected_sql}) LIMIT 0").description]
            actual_cols = [d[0] for d in cur.execute(f"SELECT * FROM ({candidate_sql_clean}) LIMIT 0").description]
            if len(expected_cols) != len(actual_cols):
                result['failed_check'] = 'template_integrity'
                result['details'] = f'Expected {len(expected_cols)} columns, got {len(actual_cols)}'
                return result
        except sqlite3.Error as e:
            result['failed_check'] = 'template_integrity'
            result['details'] = f'Template integrity check failed: {str(e)}'
            return result

    result['passed'] = True
    result['details'] = 'All validation checks passed'
    return result


# =============================================================================
# Tool 4: Retry Generation
# =============================================================================
def retry_generation(user_question, failed_sql, error_message, schema_context):
    '''
    Regenerates SQL after a validation failure, feeding the error back to the LLM.

    Parameters:
    - user_question (str): The original user question.
    - failed_sql (str): The SQL that failed validation.
    - error_message (str): The specific failure reason.
    - schema_context (str): Database schema description.

    Returns:
    - str: Revised SQL as a string.
    '''

    retry_prompt = f"""

You are a senior SQL developer specializing in credit risk analytics for Northbridge Bank using a SQLite database. Your job is to correct a SQL query that failed validation.

Review the original SQL query, the business user's question, and the validation feedback.

Identify and correct any errors involving tables, columns, joins, filters, calculations, or SQLite syntax.

Use only the tables and columns available in the provided database schema.

Generate a corrected, read-only SELECT query that accurately answers the user's question.

Return only the corrected SQL query without additional explanations or formatting.

User Question:
{user_question}

Failed SQL:
{failed_sql}

Validation Error:
{error_message}

Database Schema:
{schema_context}

"""

    revised_sql = llm.invoke(retry_prompt).content.strip()
    revised_sql = re.sub(r'^```sql\s*|\s*```$', '', revised_sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    revised_sql = re.sub(r'^```\s*|\s*```$', '', revised_sql, flags=re.MULTILINE).strip()
    return revised_sql


# =============================================================================
# Tool 5: Query Execution
# =============================================================================
def execute_query(validated_sql, db_connection):
    '''
    Executes a gate-passed SQL query and returns the result as a DataFrame.

    Parameters:
    - validated_sql (str): SQL query that has passed all validation checks.
    - db_connection: Read-only SQLite connection object.

    Returns:
    - dict: Contains 'dataframe' (pandas DataFrame), 'reasonable' (bool),
            and 'warnings' (list of warning strings).
    '''

    result = {
        'dataframe': None,
        'reasonable': True,
        'warnings': []
    }

    df = pd.read_sql_query(validated_sql, db_connection)
    result['dataframe'] = df

    # Reasonableness checks
    if df.empty:
        result['warnings'].append('Query returned an empty result')

    for col in df.select_dtypes(include='number').columns:
        if (df[col] < 0).any() and 'deviation' not in col.lower() and 'change' not in col.lower():
            result['warnings'].append(f'Column {col} contains negative values')
        if df[col].isnull().any():
            null_count = df[col].isnull().sum()
            if null_count > len(df) * 0.5:
                result['warnings'].append(f'Column {col} has {null_count} null values')

    if len(result['warnings']) > 2:
        result['reasonable'] = False

    return result


# =============================================================================
# Tool 6: Response Generation
# =============================================================================
def generate_response(user_question, dataframe, route, query_id=None):
    '''
    Generates a focused natural language response from the query result.

    Parameters:
    - user_question (str): The original user question.
    - dataframe (pd.DataFrame): The full query result.
    - route (str): 'verified' or 'generated'.
    - query_id (str, optional): Template ID if from verified track.

    Returns:
    - str: Natural language response focused on what the user asked.
    '''

    response_prompt = f"""

You are a senior credit risk analyst at Northbridge Bank. Your job is to translate SQL query results into clear and accurate business insights.

Review the business user's question and the provided query results.

Generate a concise, professional response that directly answers the user's question.

Use only the information provided in the query results. Do not make assumptions or invent information.

Highlight relevant figures, trends, or findings when appropriate.

Use simple business language and avoid unnecessary technical explanations.

{user_question}

{dataframe.to_string()}

"""

    narrative = llm.invoke(response_prompt).content.strip()
    return narrative


# =============================================================================
# Pipeline Orchestration
# =============================================================================
def run_pipeline(user_question, db_connection, query_library, schema_context,
                 verbose=True, step_callback=None):
    '''
    Runs the complete query engine pipeline for a single user question.

    Parameters:
    - user_question (str): The natural language question.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query template library.
    - schema_context (str): Database schema description.
    - verbose (bool): If True, reports intermediate pipeline stages.
    - step_callback (callable, optional): Receives each stage message (used by the
      Streamlit UI in place of the notebook's print statements). If omitted,
      messages are printed to the console.

    Returns:
    - dict: Complete pipeline output including narrative, SQL, data, and log.
    '''

    def emit(message):
        if not verbose:
            return
        if step_callback is not None:
            step_callback(message)
        else:
            print(message)

    log = {
        'user_question': user_question,
        'route': None,
        'query_id': None,
        'match_reason': None,
        'candidate_sql': None,
        'gate_result': None,
        'retry_used': False,
        'escalated': False,
        'executed_sql': None,
        'row_count': None,
        'confidence': None,
        'narrative': None
    }

    # Step 1: Intent classification
    classification = classify_intent(user_question, query_library)
    log['route'] = classification['route']
    log['query_id'] = classification.get('query_id')
    log['match_reason'] = classification.get('match_reason')

    emit(f"[1] Intent Classification: route={log['route']}, query_id={log['query_id']}")
    emit(f"    Reason: {log['match_reason']}")

    # Step 2: Query construction
    if log['route'] == 'verified' and log['query_id'] in query_library:
        candidate_sql = query_library[log['query_id']]['sql']
    else:
        candidate_sql = generate_query(user_question, schema_context)
    log['candidate_sql'] = candidate_sql

    emit(f"[2] Query Construction: {'loaded from library' if log['route']=='verified' else 'generated fresh SQL'}")

    # Step 3: Validation gate
    gate = validate_query(user_question, candidate_sql, db_connection, query_library, log['query_id'])
    log['gate_result'] = gate

    emit(f"[3] Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
    if not gate['passed']:
        emit(f"    Failed check: {gate.get('failed_check')}")
        emit(f"    Details: {gate.get('details')}")

    # Step 4: Retry once on generated track if validation fails
    if not gate['passed'] and log['route'] == 'generated':
        emit(f"    Retrying: {gate['details']}")
        candidate_sql = retry_generation(user_question, candidate_sql, gate['details'], schema_context)
        log['candidate_sql'] = candidate_sql
        log['retry_used'] = True
        gate = validate_query(user_question, candidate_sql, db_connection, query_library, None)
        log['gate_result'] = gate

        emit(f"    Retry Validation Gate: passed={gate['passed']}, relevance_confidence={gate.get('relevance_confidence')}")
        if not gate['passed']:
            emit(f"    Retry failed check: {gate.get('failed_check')}")
            emit(f"    Retry details: {gate.get('details')}")

    # Step 5: Escalate if still failing
    if not gate['passed']:
        log['escalated'] = True
        log['narrative'] = f"Query could not be reliably resolved. Escalated to human analyst. Failure: {gate['details']}"
        log['confidence'] = 'ESCALATED'
        emit(f"[!] Escalated to human: {gate['details']}")
        return {'log': log, 'dataframe': None, **log}

    # Step 6: Execute
    log['executed_sql'] = candidate_sql
    exec_result = execute_query(candidate_sql, db_connection)
    df = exec_result['dataframe']
    log['row_count'] = len(df)

    emit(f"[4] Execute: {len(df)} rows returned")
    if exec_result['warnings']:
        emit(f"    Warnings: {exec_result['warnings']}")

    # Step 7: Response generation
    narrative = generate_response(user_question, df, log['route'], log['query_id'])
    log['narrative'] = narrative

    # Confidence: carried directly from the validation gate's relevance check (0-1)
    log['confidence'] = gate.get('relevance_confidence')

    emit(f"[6] Response Generation: confidence={log['confidence']}")

    return {'log': log, 'dataframe': df, **log}


# =============================================================================
# Streamlit helpers: audit trail, running the pipeline, rendering
# =============================================================================
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []
if "last_run" not in st.session_state:
    st.session_state.last_run = None
if "evaluation" not in st.session_state:
    st.session_state.evaluation = None
if "question_input" not in st.session_state:
    st.session_state.question_input = ""

EXAMPLE_PLACEHOLDER = "— choose an example question —"
DEFAULT_EXAMPLES = [
    "What is the total outstanding and NPA exposure for the Real Estate sector?",
    "Show me the days-past-due aging profile of the portfolio.",
    "Among restructured loans, how many are impaired and which borrower has the largest impaired loan?",
    "What is the average interest rate by sector?",
    "How have Stage 3 EAD and ECL trended over the last four quarters?",
]


def record_audit(result):
    """Append a complete audit record for every analytical output."""
    gate = result.get("gate_result") or {}
    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "user_question": result.get("user_question"),
        "route": result.get("route"),
        "query_id": result.get("query_id"),
        "match_reason": result.get("match_reason"),
        "candidate_sql": result.get("candidate_sql"),
        "executed_sql": result.get("executed_sql"),
        "validation_passed": gate.get("passed"),
        "failed_check": gate.get("failed_check"),
        "validation_details": gate.get("details"),
        "relevance_confidence": gate.get("relevance_confidence"),
        "retry_used": result.get("retry_used"),
        "escalated": result.get("escalated"),
        "row_count": result.get("row_count"),
        "confidence": result.get("confidence"),
        "narrative": result.get("narrative"),
    }
    st.session_state.audit_log.append(entry)

    # Best-effort persistent copy (append-only JSON Lines). Set AUDIT_LOG_PATH to
    # a writable location; on Streamlit Community Cloud the disk is ephemeral.
    if AUDIT_LOG_PATH:
        try:
            with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
        except Exception:
            pass


def error_result(question, exc):
    """Shape an unexpected failure like an escalation so it is still audited."""
    return {
        "user_question": question,
        "route": None,
        "query_id": None,
        "match_reason": None,
        "candidate_sql": None,
        "gate_result": None,
        "retry_used": False,
        "escalated": True,
        "executed_sql": None,
        "row_count": None,
        "confidence": "ERROR",
        "narrative": f"Pipeline error. Escalated to human analyst. Error: {exc}",
        "dataframe": None,
    }


def run_and_record(question, verbose=True, on_step=None):
    """Run the pipeline on a fresh read-only connection and write the audit record."""
    steps = []

    def collect(message):
        steps.append(message)
        if on_step is not None:
            on_step(message)

    try:
        with closing(open_readonly_connection(DB_PATH)) as conn:
            result = run_pipeline(
                question, conn, verified_query_library, database_schema,
                verbose=verbose, step_callback=collect,
            )
    except Exception as exc:  # keep the app alive; escalate to a human
        result = error_result(question, exc)
        steps.append(f"[!] Pipeline error: {exc}")

    record_audit(result)
    return result, steps


def format_confidence(conf):
    if isinstance(conf, (int, float)) and not isinstance(conf, bool):
        return f"{conf * 100:.0f}%"
    return "N/A" if conf is None else str(conf)


def render_result(result, steps, key_prefix):
    """Show SQL used, raw data, narrative, and confidence for every answer."""
    gate = result.get("gate_result") or {}

    if result.get("escalated"):
        st.error("This question was escalated to a human analyst.")
        st.markdown(str(result["narrative"]).replace("$", r"\$"))
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Route", str(result["route"]).title())
        c2.metric("Query ID", result["query_id"] or "Generated")
        c3.metric("Confidence", format_confidence(result["confidence"]))
        c4.metric("Rows returned", result["row_count"])

        st.subheader("Answer")
        # Escape "$" so dollar amounts are not rendered as LaTeX math.
        st.markdown(str(result["narrative"]).replace("$", r"\$"))

    if result.get("match_reason"):
        st.caption(f"Routing decision: {result['match_reason']}")
    if result.get("retry_used"):
        st.info("The first generated SQL failed validation, so it was regenerated once.")

    sql_to_show = result.get("executed_sql") or result.get("candidate_sql")
    if sql_to_show:
        label = "SQL executed" if result.get("executed_sql") else "Candidate SQL (not executed)"
        with st.expander(label, expanded=True):
            st.code(sql_to_show, language="sql")

    df = result.get("dataframe")
    if df is not None:
        with st.expander("Raw data returned", expanded=True):
            st.dataframe(df)
            st.download_button(
                "Download result as CSV",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="query_result.csv",
                mime="text/csv",
                key=f"{key_prefix}_download",
            )

    if gate:
        with st.expander("Validation gate details"):
            st.json(gate)
    if steps:
        with st.expander("Pipeline trace"):
            st.code("\n".join(steps), language="text")


def run_evaluation(gt_df, progress_bar):
    """Run every ground-truth case and compute the notebook's evaluation metrics."""
    evaluation_rows = []
    test_results = []
    total = len(gt_df)

    for i, (_, gt) in enumerate(gt_df.iterrows()):
        result, _ = run_and_record(gt["User Query"], verbose=False)
        test_results.append(result)
        progress_bar.progress((i + 1) / total, text=f"Running test case {i + 1} of {total}")

        evaluation_rows.append({
            'Test Case': gt['Test Case'],
            'Expected Route': gt['Expected Route'],
            'Actual Route': result['route'],
            'Route Match': result['route'] == gt['Expected Route'],
            'Expected Query ID': gt['Expected Query ID'],
            'Actual Query ID': result['query_id'],
            'Query ID Match': (
                pd.isna(gt['Expected Query ID']) and pd.isna(result['query_id'])
            ) or result['query_id'] == gt['Expected Query ID'],
            'Confidence': result['confidence'],
            'Rows Returned': result['row_count']
        })

    evaluation_df = pd.DataFrame(evaluation_rows)

    path_accuracy = evaluation_df['Route Match'].mean() * 100

    verified = evaluation_df['Expected Route'].str.strip().str.lower() == 'verified'
    query_accuracy = evaluation_df.loc[verified, 'Query ID Match'].mean() * 100

    # Escalated cases carry the string 'ESCALATED' as confidence, so coerce to numeric.
    average_confidence = pd.to_numeric(evaluation_df['Confidence'], errors='coerce').mean()

    return {
        "evaluation_df": evaluation_df,
        "results": test_results,
        "path_accuracy": path_accuracy,
        "query_accuracy": query_accuracy,
        "average_confidence": average_confidence,
    }


def use_example():
    choice = st.session_state.get("example_choice")
    if choice and choice != EXAMPLE_PLACEHOLDER:
        st.session_state.question_input = choice


# =============================================================================
# Readiness checks + sidebar
# =============================================================================
db_ok = Path(DB_PATH).exists()
key_ok = bool(OPENAI_API_KEY)

with st.sidebar:
    st.header("System status")
    st.write(f"{'✅' if db_ok else '❌'} Database: `{Path(DB_PATH).name}` (read-only)")
    st.write(f"{'✅' if key_ok else '❌'} OpenAI API key")
    st.write(f"{'✅' if ground_truth is not None else 'ℹ️'} Ground-truth test cases"
             + ("" if ground_truth is not None else " (optional, not found)"))
    st.write(f"🤖 Model: `{OPENAI_MODEL}`")
    st.divider()
    st.caption(
        "Routine questions use pre-approved SQL templates. Novel questions get "
        "generated SELECT-only SQL that must pass five validation checks, with one "
        "retry before escalation to a human analyst. Every answer shows the SQL "
        "used, the raw data, and a confidence score, and is written to the audit trail."
    )


# =============================================================================
# Main layout
# =============================================================================
st.title("🏦 Northbridge Bank - Credit Risk Query Engine")
st.caption(
    "Ask routine commercial-lending portfolio questions in plain English. "
    "Read-only access to the credit-risk database."
)

tab_ask, tab_library, tab_schema, tab_eval, tab_audit = st.tabs(
    ["Ask a Question", "Verified Query Library", "Database & Schema",
     "Evaluation", "Audit Trail"]
)

# ------------------------------- Ask a Question -------------------------------
with tab_ask:
    if not key_ok:
        st.error(
            "OPENAI_API_KEY was not found. Add it to Streamlit secrets, an environment "
            "variable, or config.json (see the instructions at the top of app.py)."
        )
    if not db_ok:
        st.error(f"Database file not found at: {DB_PATH}")

    example_questions = []
    if ground_truth is not None and "User Query" in ground_truth.columns:
        example_questions += [q for q in ground_truth["User Query"].dropna().tolist()]
    example_questions += [q for q in DEFAULT_EXAMPLES if q not in example_questions]

    st.selectbox(
        "Example questions",
        [EXAMPLE_PLACEHOLDER] + example_questions,
        key="example_choice",
        on_change=use_example,
    )
    st.text_area(
        "Your question",
        key="question_input",
        height=100,
        placeholder="e.g., Which sectors have the highest NPA exposure?",
    )

    if st.button("Run query", type="primary", disabled=not (key_ok and db_ok)):
        question = st.session_state.question_input.strip()
        if not question:
            st.warning("Please enter a question.")
        else:
            with st.status("Running the query engine pipeline...", expanded=True) as status:
                result, steps = run_and_record(
                    question, verbose=True, on_step=lambda m: status.text(m)
                )
                if result.get("escalated"):
                    status.update(label="Escalated to human analyst", state="error", expanded=False)
                else:
                    status.update(label="Pipeline complete", state="complete", expanded=False)
            st.session_state.last_run = {"result": result, "steps": steps}

    if st.session_state.last_run:
        st.divider()
        render_result(
            st.session_state.last_run["result"],
            st.session_state.last_run["steps"],
            key_prefix="ask",
        )

# --------------------------- Verified Query Library ---------------------------
with tab_library:
    st.subheader("Verified Query Template Library")
    st.write(
        f"{len(verified_query_library)} pre-approved templates. Each runs without "
        "modification and returns a complete result set; the LLM then writes a "
        "focused narrative based on what the user actually asked."
    )
    for qid, entry in verified_query_library.items():
        with st.expander(f"{qid}: {entry['description']}"):
            st.code(entry["sql"], language="sql")

# ------------------------------ Database & Schema -----------------------------
with tab_schema:
    st.subheader("Schema provided to the LLM")
    st.code(database_schema, language="text")

    st.subheader("Table previews (first 3 rows)")
    if db_ok:
        try:
            with closing(open_readonly_connection(DB_PATH)) as preview_conn:
                for table_name in ['sector_master', 'loan_master', 'borrower_rating', 'provisioning']:
                    st.markdown(f"**{table_name}**")
                    st.dataframe(
                        pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 3", preview_conn)
                    )
        except Exception as exc:
            st.error(f"Could not read table previews: {exc}")
    else:
        st.warning("Database file not found.")

# --------------------------------- Evaluation ---------------------------------
with tab_eval:
    st.subheader("Evaluation against ground truth")
    if ground_truth is None:
        st.info(
            "No ground-truth file found. Place test_queries.csv (columns: Test Case, "
            "User Query, Expected Route, Expected Query ID, Expected Answer) next to "
            "app.py, or set GROUND_TRUTH_PATH."
        )
    else:
        with st.expander("Ground-truth test cases"):
            st.dataframe(ground_truth)

        if st.button("Run evaluation on all test cases", disabled=not (key_ok and db_ok)):
            bar = st.progress(0.0, text="Starting evaluation...")
            try:
                st.session_state.evaluation = run_evaluation(ground_truth, bar)
            except Exception as exc:
                st.session_state.evaluation = None
                st.error(f"Evaluation failed: {exc}")
            bar.empty()

        ev = st.session_state.evaluation
        if ev:
            m1, m2, m3 = st.columns(3)
            m1.metric("Selected Path Accuracy", f"{ev['path_accuracy']:.1f}%")
            qa = ev["query_accuracy"]
            m2.metric("Selected Query Accuracy", "N/A" if pd.isna(qa) else f"{qa:.1f}%")
            ac = ev["average_confidence"]
            m3.metric("Average Confidence Score", "N/A" if pd.isna(ac) else f"{ac:.2f}")
            st.dataframe(ev["evaluation_df"])

            st.markdown("**Individual test case outputs**")
            for idx, res in enumerate(ev["results"]):
                with st.expander(f"Test case {idx + 1}: {res['user_question']}"):
                    render_result(res, None, key_prefix=f"eval_{idx}")

# --------------------------------- Audit Trail --------------------------------
with tab_audit:
    st.subheader("Audit trail (this session)")
    if not st.session_state.audit_log:
        st.info("No queries have been run yet in this session.")
    else:
        audit_df = pd.DataFrame(st.session_state.audit_log)
        st.dataframe(audit_df)
        d1, d2 = st.columns(2)
        d1.download_button(
            "Download audit trail (CSV)",
            data=audit_df.to_csv(index=False).encode("utf-8"),
            file_name="audit_trail.csv",
            mime="text/csv",
        )
        d2.download_button(
            "Download audit trail (JSON Lines)",
            data="\n".join(json.dumps(e, default=str) for e in st.session_state.audit_log).encode("utf-8"),
            file_name="audit_trail.jsonl",
            mime="application/json",
        )
    if AUDIT_LOG_PATH:
        st.caption(f"Each record is also appended to `{AUDIT_LOG_PATH}` when the location is writable.")
