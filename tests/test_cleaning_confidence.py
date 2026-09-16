import pytest
import pandas as pd
import numpy as np
from app.agents.cleaning import detect_missing, detect_outliers, detect_duplicates, detect_format_inconsistency
from app.agents.confidence import column_signature, CONFIDENCE_THRESHOLD

def test_missing_values():
    # a) Known semantic metadata case -> Confidence 1.0
    df = pd.DataFrame({"col": [1, 2, np.nan, 4]})
    res = detect_missing(df, "col", {"missing_meaning": "skipped"})
    assert res is not None
    assert res["confidence"] == 1.0
    assert res["proposed_fix"] == "drop_rows"

    # b) <1% missingness case -> Confidence 0.90
    # 200 rows, 1 missing = 0.5%
    df2 = pd.DataFrame({"col": list(range(199)) + [np.nan]})
    res2 = detect_missing(df2, "col", {})
    assert res2 is not None
    assert res2["confidence"] == 0.90
    assert res2["proposed_fix"] == "drop_rows"

    # c) Ambiguous case (e.g. 50% missing) -> Confidence 0.30
    df3 = pd.DataFrame({"col": [1, 2, np.nan, np.nan]})
    res3 = detect_missing(df3, "col", {})
    assert res3 is not None
    assert res3["confidence"] == 0.30
    assert res3["proposed_fix"] == "impute_median"

def test_outliers():
    # 1. High agreement + low % (<5%)
    # Let's create a large enough array so 1 outlier is < 5%
    # 100 rows, 1 massive outlier. Use a very tight distribution so IQR only catches the 100.0
    data = [0.0]*99 + [100.0]
    df = pd.DataFrame({"col": data})
    res = detect_outliers(df, "col", {})
    assert res is not None
    # Both Z-score and IQR will catch 100.0.
    # intersection/union = 1.0 > 0.9, affected = 1% < 5%
    assert res["confidence"] == 0.88
    
    # 2. Low agreement
    # We can craft data where IQR catches more than Z-score.
    # Z-score uses std.
    df2 = pd.DataFrame({"col": [1,1,1,1,1,1,1,1, 10, 15, 20]})
    res2 = detect_outliers(df2, "col", {})
    assert res2 is not None
    # Since agreement is low, it uses formula: 0.4 + (ratio * 0.3)
    # We just ensure it's not 0.88 and matches the formula
    z = res2["evidence"]["z_outliers"]
    iqr = res2["evidence"]["iqr_outliers"]
    inter = res2["evidence"]["intersection"]
    union = z + iqr - inter
    ratio = inter / union
    assert res2["confidence"] == round(0.4 + (ratio * 0.3), 2)
    assert res2["confidence"] < 0.85 # triggers HITL

    # 3. Edge case: exactly at the boundaries or just over 5%
    data3 = [0.0]*94 + [100.0]*6 # 6% affected
    df3 = pd.DataFrame({"col": data3})
    res3 = detect_outliers(df3, "col", {})
    assert res3 is not None
    # Agreement will be 1.0, but affected > 5%, so it falls back to formula
    z3 = res3["evidence"]["z_outliers"]
    iqr3 = res3["evidence"]["iqr_outliers"]
    inter3 = res3["evidence"]["intersection"]
    union3 = z3 + iqr3 - inter3
    ratio3 = inter3 / union3
    assert ratio3 == 1.0
    assert res3["confidence"] == round(0.4 + (ratio3 * 0.3), 2)
    assert res3["confidence"] == 0.70

def test_duplicates():
    # <5%
    df1 = pd.DataFrame({"col": list(range(100)) + [1]}) # 1/101 < 5%
    res1 = detect_duplicates(df1)
    assert res1["confidence"] == 0.95
    
    # 5-10%
    df2 = pd.DataFrame({"col": list(range(100)) + [1,2,3,4,5,6,7]}) # 7/107 = 6.5%
    res2 = detect_duplicates(df2)
    assert res2["confidence"] == 0.80
    
    # >10%
    df3 = pd.DataFrame({"col": list(range(10)) + [1,2,3]}) # 3/13 = 23%
    res3 = detect_duplicates(df3)
    assert res3["confidence"] == 0.50

def test_mixed_casing():
    # >90% dominant casing
    df1 = pd.DataFrame({"col": ["apple"]*95 + ["Apple"]*5})
    res1 = detect_format_inconsistency(df1, "col")
    assert res1 is not None
    assert res1["confidence"] == 0.90
    
    # Ambiguous case
    df2 = pd.DataFrame({"col": ["apple"]*60 + ["Apple"]*40})
    res2 = detect_format_inconsistency(df2, "col")
    assert res2 is not None
    assert res2["confidence"] == 0.60

def test_column_signature():
    # 1. Fuzzy matching inside a bucket
    # Bucket 0-5% (e.g. 1% and 4% missing should be same)
    sig1 = column_signature("income", "float64", 0.01)
    sig2 = column_signature("income", "float64", 0.04)
    assert sig1 == sig2
    
    # 2. Different bucket
    sig3 = column_signature("income", "float64", 0.10) # 5-15% bucket
    assert sig1 != sig3
