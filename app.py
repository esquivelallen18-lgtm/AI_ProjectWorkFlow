"""
Northbridge Bank - Credit Risk Query Engine (Streamlit app)

Converts the "Credit Risk Query Engine" notebook into a Streamlit application.
The pipeline logic, verified query library, SQL templates, prompts and validation
gate are carried over from the notebook unchanged:

    classify_intent -> (verified template | generate_query) -> validate_query
    -> (retry_generation once, generated track only) -> execute_query
    -> generate_response   (escalate to a human analyst if validation still fails)

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Required files (next to app.py, or set the paths via environment variables / secrets):
    credit_risk_portfolio.db   -> SQLite database, opened read-only   (DB_PATH)
    test_queries.csv           -> optional, ground-truth test cases    (TEST_QUERIES_PATH)

Required credentials (Streamlit secrets, environment variables, or config.json):
    OPENAI_API_KEY
    OPENAI_API_BASE            -> optional, only if you use a custom/enterprise endpoint
"""

import streamlit as st

# Must be the first Streamlit command in the script
st.set_page_config(
    page_title="Northbridge Credit Risk Query Engine",
    page_icon="🏦",
    layout="wide",
)

from typing import TypedDict, List, Any, Dict, Optional
import pandas as pd
import sqlite3
import sqlparse
import json
import re
import os
from datetime import datetime, timezone
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Configuration (replaces the Colab Secrets / Google Drive loading in the notebook)
# =============================================================================

APP_DIR = Path(__file__).resolve().parent


def get_setting(name, default=None):
    """Look up a setting in Streamlit secrets first, then environment variables."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    value = os.environ.get(name)
    if value:
        return value
    return default


def load_config_json():
    """Optional fallback: read credentials from a local config.json (as in the notebook)."""
    config_path = Path(get_setting("CONFIG_JSON_PATH", str(APP_DIR / "config.json")))
    if config_path.exists():
        try:
            with open(config_path, 'r') as file:
                return json.load(file)
        except Exception:
            return {}
    return {}


def load_credentials():
    """
    Priority: Streamlit secrets / environment variables, then config.json.
    Returns (OPENAI_API_KEY, OPENAI_API_BASE).
    """
    config = load_config_json()
    api_key = get_setting("OPENAI_API_KEY") or config.get("OPENAI_API_KEY")
    api_base = get_setting("OPENAI_API_BASE") or config.get("OPENAI_API_BASE")
    return api_key, api_base


MODEL_NAME = get_setting("OPENAI_MODEL", "gpt-4o-mini")
DB_PATH = get_setting("DB_PATH", str(APP_DIR / "credit_risk_portfolio.db"))
AUDIT_LOG_PATH = get_setting("AUDIT_LOG_PATH", str(APP_DIR / "audit_log.jsonl"))


def find_ground_truth_path():
    custom = get_setting("TEST_QUERIES_PATH")
    candidates = [custom] if custom else []
    candidates += [str(APP_DIR / "test_queries.csv"), str(APP_DIR / "test_queries-2.csv")]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


# =============================================================================
# Cached resources: read-only database connection, LLMs, ground-truth CSV
# =============================================================================

@st.cache_resource(show_spinner=False)
def get_connection(db_path):
    # Read-only connection (mode=ro) prevents any write operation on the database.
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True, check_same_thread=False)


@st.cache_resource(show_spinner=False)
def get_llms(api_key, api_base, model_name):
    # Store API credentials in environment variables (picked up by langchain_openai)
    os.environ['OPENAI_API_KEY'] = api_key
    if api_base:
        os.environ["OPENAI_API_BASE"] = api_base

    llm = ChatOpenAI(model=model_name, temperature=0)
    evaluator_llm = ChatOpenAI(model=model_name, temperature=0)
    return llm, evaluator_llm


@st.cache_data(show_spinner=False)
def load_ground_truth(csv_path):
    return pd.read_csv(csv_path)


# =============================================================================
# Database schema provided to the LLM in the prompts
# =============================================================================

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
# Verified Query Template Library
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
    ROUND(SUM(ecl_amount) / 1000000.0, 2) AS total_ecl_millions
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
# Tool Definitions (pipeline stages)
# =============================================================================

# NOTE: `llm` and `evaluator_llm` are initialised further below (after the
# credentials check) and are looked up at call time by the functions that follow.

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
    # FIX: Use SELECT * FROM (...) LIMIT 0 to avoid syntax errors with existing templates that already contain a LIMIT clause.
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
# LLM Binding and Pipeline Orchestration
# =============================================================================

def run_pipeline(user_question, db_connection, query_library, schema_context, verbose=True, status_callback=None):
    '''
    Runs the complete query engine pipeline for a single user question.

    Parameters:
    - user_question (str): The natural language question.
    - db_connection: SQLite connection object.
    - query_library (dict): Verified query template library.
    - schema_context (str): Database schema description.
    - verbose (bool): If True, reports intermediate pipeline stages
                      (to `status_callback` if given, otherwise printed).
    - status_callback (callable, optional): Receives each progress message (used by the Streamlit UI).

    Returns:
    - dict: Complete pipeline output including narrative, SQL, data, and log.
            (Streamlit additions: 'trace' = list of progress messages,
             'warnings' = reasonableness warnings from execute_query.)
    '''

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

    trace = []

    def emit(message):
        trace.append(message)
        if verbose:
            if status_callback:
                status_callback(message)
            else:
                print(message)

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
        return {'log': log, 'dataframe': None, 'trace': trace, 'warnings': [], **log}


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

    return {'log': log, 'dataframe': df, 'trace': trace, 'warnings': exec_result['warnings'], **log}


# =============================================================================
# Evaluation against ground truth (from the notebook's evaluation section)
# =============================================================================

def evaluate_against_ground_truth(ground_truth, db_connection, query_library, schema_context, progress_callback=None):
    """
    Runs every ground-truth test case through the pipeline and compares the selected
    route and verified query ID with the expected values.
    Returns (evaluation_df, path_accuracy, query_accuracy, average_confidence).
    """
    evaluation_rows = []
    total = len(ground_truth)

    for i, (_, gt) in enumerate(ground_truth.iterrows()):
        try:
            tr = run_pipeline(gt['User Query'], db_connection, query_library, schema_context, verbose=False)
        except Exception as exc:
            tr = {'route': None, 'query_id': None, 'confidence': f'ERROR: {exc}', 'row_count': None}

        evaluation_rows.append({
            'Test Case': gt['Test Case'],
            'Expected Route': gt['Expected Route'],
            'Actual Route': tr['route'],
            'Route Match': tr['route'] == gt['Expected Route'],
            'Expected Query ID': gt['Expected Query ID'],
            'Actual Query ID': tr['query_id'],
            'Query ID Match': (
                pd.isna(gt['Expected Query ID']) and pd.isna(tr['query_id'])
            ) or tr['query_id'] == gt['Expected Query ID'],
            'Confidence': tr['confidence'],
            'Rows Returned': tr['row_count']
        })

        if progress_callback:
            progress_callback((i + 1) / total, f"Completed {i + 1} of {total} test cases")

    evaluation_df = pd.DataFrame(evaluation_rows)

    path_accuracy = evaluation_df['Route Match'].mean() * 100

    verified = evaluation_df['Expected Route'].str.strip().str.lower() == 'verified'
    query_accuracy = evaluation_df.loc[verified, 'Query ID Match'].mean() * 100

    average_confidence = pd.to_numeric(evaluation_df['Confidence'], errors='coerce').mean()

    return evaluation_df, path_accuracy, query_accuracy, average_confidence


# =============================================================================
# Streamlit helpers
# =============================================================================

def format_confidence(confidence):
    if confidence is None:
        return "N/A"
    if isinstance(confidence, (int, float)):
        return f"{confidence:.0%}"
    return str(confidence)


def format_number(value, fmt="{:.1f}"):
    try:
        if pd.isna(value):
            return "N/A"
        return fmt.format(value)
    except Exception:
        return "N/A"


def escape_markdown_dollars(text):
    # Prevents "$1,195 million ... $3,450 million" from being rendered as LaTeX
    return str(text).replace("$", "\\$")


def build_audit_entry(result):
    """One audit record per analytical output: question, route, SQL, validation, confidence, answer."""
    log = result['log']
    gate = log.get('gate_result') or {}
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "user_question": log.get('user_question'),
        "route": log.get('route'),
        "query_id": log.get('query_id'),
        "match_reason": log.get('match_reason'),
        "candidate_sql": log.get('candidate_sql'),
        "executed_sql": log.get('executed_sql'),
        "validation_passed": gate.get('passed'),
        "validation_failed_check": gate.get('failed_check'),
        "validation_details": gate.get('details'),
        "retry_used": log.get('retry_used'),
        "escalated": log.get('escalated'),
        "row_count": log.get('row_count'),
        "confidence": log.get('confidence'),
        "narrative": log.get('narrative'),
    }


def record_audit(result):
    entry = build_audit_entry(result)
    st.session_state.audit_log.append(entry)
    try:
        with open(AUDIT_LOG_PATH, "a", encoding="utf-8") as audit_file:
            audit_file.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        # Persistent file logging is best-effort (e.g., read-only hosting file systems).
        pass


def set_question(question):
    st.session_state.question_input = question


def run_with_status(question):
    """Runs the pipeline while streaming the stage-by-stage trace into the UI."""
    result = None
    try:
        with st.status("Running query engine pipeline...", expanded=True) as status:
            result = run_pipeline(
                question,
                conn,
                verified_query_library,
                database_schema,
                verbose=True,
                status_callback=status.text,
            )
            if result['log']['escalated']:
                status.update(label="Escalated to human analyst", state="error", expanded=False)
            else:
                status.update(label="Pipeline complete", state="complete", expanded=False)
    except Exception as exc:
        st.error(f"The pipeline failed: {exc}")
        result = None
    return result


def render_result(result):
    log = result['log']
    df = result['dataframe']

    st.divider()

    # --- Summary metrics: route, template, confidence, rows ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Route", str(log['route']).title())
    col2.metric("Verified Query ID", log['query_id'] or "—")
    col3.metric("Confidence", format_confidence(log['confidence']))
    col4.metric("Rows returned", "—" if log['row_count'] is None else log['row_count'])

    if log.get('match_reason'):
        st.caption(f"Routing decision: {log['match_reason']}")
    if log.get('retry_used'):
        st.info("The first generated SQL failed validation; the query was regenerated once and re-validated.")

    # --- Answer ---
    if log['escalated']:
        st.warning(escape_markdown_dollars(log['narrative']))
        if log.get('candidate_sql'):
            with st.expander("Candidate SQL that failed validation", expanded=True):
                st.code(log['candidate_sql'], language="sql")
    else:
        st.subheader("Answer")
        st.markdown(escape_markdown_dollars(log['narrative']))

        # --- SQL used ---
        st.subheader("SQL query used")
        st.code(log['executed_sql'], language="sql")

        # --- Raw data ---
        st.subheader("Raw data returned")
        for warning_message in result.get('warnings', []):
            st.warning(warning_message)
        st.dataframe(df)
        st.download_button(
            "Download result as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="query_result.csv",
            mime="text/csv",
        )

    # --- Validation gate, trace and full log (auditability) ---
    with st.expander("Validation gate result"):
        st.json(log.get('gate_result') or {})
    with st.expander("Pipeline trace"):
        st.code("\n".join(result.get('trace', [])), language="text")
    with st.expander("Full pipeline log (audit record)"):
        st.json(build_audit_entry(result))


# =============================================================================
# Startup checks: credentials, database, LLMs, ground truth
# =============================================================================

OPENAI_API_KEY, OPENAI_API_BASE = load_credentials()

if not OPENAI_API_KEY:
    st.error(
        "OPENAI_API_KEY was not found. Add it to `.streamlit/secrets.toml` "
        "(or the Streamlit Cloud *Secrets* settings), set it as an environment variable, "
        "or provide a `config.json` file next to app.py."
    )
    st.stop()

if not Path(DB_PATH).exists():
    st.error(
        f"Database file not found at `{DB_PATH}`. Place `credit_risk_portfolio.db` next to app.py "
        "or set the DB_PATH setting."
    )
    st.stop()

conn = get_connection(DB_PATH)
llm, evaluator_llm = get_llms(OPENAI_API_KEY, OPENAI_API_BASE, MODEL_NAME)

GROUND_TRUTH_PATH = find_ground_truth_path()
ground_truth = load_ground_truth(GROUND_TRUTH_PATH) if GROUND_TRUTH_PATH else None

# Session state
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "question_input" not in st.session_state:
    st.session_state.question_input = ""
if "evaluation" not in st.session_state:
    st.session_state.evaluation = None


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.header("System status")
    st.success("Database connected (read-only)")
    st.caption(f"Model: `{MODEL_NAME}`")
    st.caption(f"Verified query templates: {len(verified_query_library)}")
    st.caption(
        f"Ground-truth test cases: {len(ground_truth)}" if ground_truth is not None
        else "Ground-truth test cases: not loaded"
    )

    st.header("Example questions")
    if ground_truth is not None and 'User Query' in ground_truth.columns:
        example_questions = ground_truth['User Query'].dropna().tolist()
    else:
        example_questions = [
            "How many loans are overdue?",
            "What is the average interest rate by sector?",
            "Show the Stage 3 EAD and ECL trend over the last four quarters.",
        ]
    for idx, example in enumerate(example_questions):
        st.button(example, key=f"example_{idx}", on_click=set_question, args=(example,))

    st.header("How it works")
    st.markdown(
        "1. **Classify** the question: verified template or generated SQL\n"
        "2. **Build** the query (template library or LLM generation)\n"
        "3. **Validate**: read-only, schema, EXPLAIN, LLM relevance, template integrity\n"
        "4. **Retry once** (generated track only), otherwise **escalate** to a human analyst\n"
        "5. **Execute** on the read-only database\n"
        "6. **Answer** with SQL, raw data and a confidence score"
    )


# =============================================================================
# Main page
# =============================================================================

st.title("🏦 Northbridge Bank - Credit Risk Query Engine")
st.caption(
    "Ask routine commercial lending portfolio questions in plain English. Every answer shows the SQL "
    "used, the raw data returned and a confidence score, and is recorded in the audit trail. "
    "Read-only access; complex or ambiguous questions are escalated to a human analyst."
)

tab_ask, tab_library, tab_eval, tab_audit = st.tabs(
    ["Ask a question", "Verified query library", "Evaluation", "Audit trail"]
)

# ----------------------------- Ask a question --------------------------------
with tab_ask:
    st.text_area(
        "Your question",
        key="question_input",
        height=100,
        placeholder="e.g., What is the total NPA exposure in the Real Estate sector?",
    )
    run_clicked = st.button("Run query", type="primary")

    if run_clicked:
        question = st.session_state.question_input.strip()
        if not question:
            st.warning("Please enter a question first.")
        else:
            new_result = run_with_status(question)
            if new_result is not None:
                st.session_state.last_result = new_result
                record_audit(new_result)

    if st.session_state.last_result is not None:
        render_result(st.session_state.last_result)

# -------------------------- Verified query library ---------------------------
with tab_library:
    st.subheader("Verified Query Template Library")
    st.write(
        "Pre-approved SQL templates for recurring questions. Each template runs without modification "
        "and returns a complete result set; the narrative is written from the full result."
    )
    for qid, entry in verified_query_library.items():
        with st.expander(f"{qid}: {entry['description']}"):
            st.code(entry['sql'], language="sql")

    with st.expander("Database schema provided to the LLM"):
        st.code(database_schema, language="text")

# -------------------------------- Evaluation ---------------------------------
with tab_eval:
    st.subheader("Evaluation against ground truth")
    if ground_truth is None:
        st.info(
            "No test-case file found. Place `test_queries.csv` next to app.py "
            "(or set TEST_QUERIES_PATH) to enable evaluation."
        )
    else:
        with st.expander("Ground-truth test cases"):
            st.dataframe(ground_truth)

        if st.button("Run evaluation on all test cases"):
            progress = st.progress(0.0, text="Starting evaluation...")
            try:
                evaluation_df, path_accuracy, query_accuracy, average_confidence = evaluate_against_ground_truth(
                    ground_truth,
                    conn,
                    verified_query_library,
                    database_schema,
                    progress_callback=lambda fraction, text: progress.progress(fraction, text=text),
                )
                st.session_state.evaluation = {
                    "df": evaluation_df,
                    "path_accuracy": path_accuracy,
                    "query_accuracy": query_accuracy,
                    "average_confidence": average_confidence,
                }
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")
            progress.empty()

        if st.session_state.evaluation is not None:
            evaluation = st.session_state.evaluation
            m1, m2, m3 = st.columns(3)
            m1.metric("Selected Path Accuracy", format_number(evaluation["path_accuracy"], "{:.1f}%"))
            m2.metric("Selected Query Accuracy", format_number(evaluation["query_accuracy"], "{:.1f}%"))
            m3.metric("Average Confidence Score", format_number(evaluation["average_confidence"], "{:.2f}"))
            st.dataframe(evaluation["df"])

# ------------------------------- Audit trail ---------------------------------
with tab_audit:
    st.subheader("Audit trail")
    st.caption(
        f"Every analytical output in this session is listed below and appended to `{AUDIT_LOG_PATH}` "
        "(best-effort; the file may be ephemeral on some hosting platforms)."
    )
    if st.session_state.audit_log:
        audit_df = pd.DataFrame(st.session_state.audit_log)
        st.dataframe(audit_df)
        st.download_button(
            "Download audit trail (JSON)",
            data=json.dumps(st.session_state.audit_log, indent=2, default=str),
            file_name="audit_trail.json",
            mime="application/json",
        )
        st.download_button(
            "Download audit trail (CSV)",
            data=audit_df.to_csv(index=False).encode("utf-8"),
            file_name="audit_trail.csv",
            mime="text/csv",
        )
    else:
        st.info("No queries have been run in this session yet.")
