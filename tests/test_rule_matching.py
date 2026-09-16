import pytest
import asyncpg
from app.core.queries.rules import upsert_rule, find_matching_rule
from app.agents.confidence import column_signature
from app.core.db import init_db_pool, get_db

@pytest.mark.asyncio
async def test_rule_matching():
    # Setup test DB or use existing DB pool if acceptable for tests
    await init_db_pool()
    db_gen = get_db()
    conn = await anext(db_gen)
    
    try:
        # Generate a test signature
        col_name = "test_col_match"
        dtype_str = "float64"
        null_pct = 0.02
        sig = column_signature(col_name, dtype_str, null_pct)
        
        # Ensure it doesn't match initially or if it does, we just overwrite
        # We can just upsert and check
        await upsert_rule(
            conn=conn,
            column_signature=sig,
            question="How to handle missing in test_col_match?",
            answer="impute_median",
            transformation="impute_median"
        )
        
        # Find matching rule
        rule = await find_matching_rule(conn, sig)
        assert rule is not None
        assert rule["transformation"] == "impute_median"
        assert rule["answer"] == "impute_median"
        
        # Test non-matching
        sig_non = column_signature("other_col", "float64", 0.02)
        rule_non = await find_matching_rule(conn, sig_non)
        
        # Assuming other_col hasn't been saved, or at least it shouldn't match `sig`
        # We can't guarantee `rule_non` is None in a persistent DB without clearing,
        # but we can verify it's not the same rule ID.
        if rule_non:
            assert rule_non["id"] != rule["id"]

    finally:
        await conn.execute("DELETE FROM rules WHERE column_signature = $1", sig)
        await conn.close()
