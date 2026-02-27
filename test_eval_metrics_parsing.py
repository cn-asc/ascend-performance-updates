#!/usr/bin/env python3
"""Unit tests for eval_metrics_harness parsing and canonicalization."""
import sys
from eval_metrics_harness import (
    extract_json_from_response,
    normalize_prediction,
    resolve_other_pred_from_canon,
    unit_normalize_pred,
    canonicalize_label,
    compute_score_from_gt_and_matches,
    extract_all_performance_numbers,
    gt_value_appears_in_set,
    CANONICAL_KEYS,
    OTHER_LABEL_TO_CANONICAL,
)


def test_extract_json_strict():
    """Strict JSON parse: first JSON object only; nested braces handled. Returns (data, parse_error)."""
    # Nested object
    out, err = extract_json_from_response('  before {"irr": 10.3, "moic": 1.2, "nested": {"a": 1}} after ')
    assert err is None
    assert out.get("irr") == 10.3
    assert out.get("moic") == 1.2
    # Markdown code block
    out2, err2 = extract_json_from_response('```json\n{"net_irr": 15.5}\n```')
    assert err2 is None
    assert out2.get("net_irr") == 15.5
    # No object -> empty dict, parse_error set
    out3, err3 = extract_json_from_response("no json here")
    assert out3 == {}
    assert err3 is not None
    # Invalid JSON -> empty dict, parse_error set
    out4, err4 = extract_json_from_response("{ invalid }")
    assert out4 == {}
    assert err4 is not None


def test_normalize_prediction_old_schema():
    """Input JSON old schema => canonical dict has irr/moic/dpi populated."""
    old = {"net_irr": 10.3, "net_moic": 1.09, "net_dpi": 0.5, "other_metric_label": None, "other_metric_value": None}
    canon = normalize_prediction(old)
    assert canon.get("irr") == 10.3
    assert canon.get("moic") == 1.09
    assert canon.get("dpi") == 0.5
    assert canon.get("tvpi") is None


def test_normalize_prediction_new_schema():
    """Input JSON new schema => canonical dict has irr/moic/dpi populated."""
    new = {"irr": 15.3, "moic": 1.2, "dpi": 0.38}
    canon = normalize_prediction(new)
    assert canon.get("irr") == 15.3
    assert canon.get("moic") == 1.2
    assert canon.get("dpi") == 0.38


def test_normalize_prediction_other_metric_current_yield():
    """Input with other_metric_label/value ('current yield', 7.84) => canonical current_yield populated."""
    raw = {"other_metric_label": "current yield", "other_metric_value": 7.84}
    canon = normalize_prediction(raw)
    # 7.84 as percentage may stay 7.84 (unit norm happens at score time)
    assert canon.get("current_yield") is not None
    assert canon.get("current_yield") == 7.84
    raw2 = {"other_metric_label": "Current Yield", "other_metric_value": 0.0784}
    canon2 = normalize_prediction(raw2)
    assert canon2.get("current_yield") == 0.0784


def test_resolve_other_pred_from_canon():
    """resolve_other_pred_from_canon maps GT label to canon value."""
    canon = {"current_yield": 0.0784, "irr": None, "moic": None}
    assert resolve_other_pred_from_canon(canon, "Current Yield") == 0.0784
    assert resolve_other_pred_from_canon(canon, "Income Distribution Rate") is None
    canon2 = {"income_distribution_rate": 12.85}
    assert resolve_other_pred_from_canon(canon2, "Income Distribution Rate") == 12.85


def test_canonical_keys_cover_expected():
    """Canonical keys include expected metric names."""
    for k in ["irr", "moic", "dpi", "tvpi", "current_yield", "income_distribution_rate", "yield", "ytd_whcm", "gmv"]:
        assert k in CANONICAL_KEYS, k
    assert "current yield" in (k.replace("_", " ") for k in OTHER_LABEL_TO_CANONICAL)


def test_unit_normalize_pred_gt_less_than_one_pred_greater():
    """If gt < 1 and pred > 1, use pred/100 when it reduces absolute error (e.g. 16.3 vs 0.163)."""
    # gt=0.163 (fraction), pred=16.3 (percent) -> normalize to 0.163
    out = unit_normalize_pred(16.3, 0.163, "irr")
    assert out is not None
    assert abs(out - 0.163) < 0.001
    # gt=0.0784, pred=7.84
    out2 = unit_normalize_pred(7.84, 0.0784, "current_yield")
    assert out2 is not None
    assert abs(out2 - 0.0784) < 0.001


def test_unit_normalize_pred_bps():
    """If label_hint contains 'bp' or 'bps', convert pred from bps to percent (divide by 100)."""
    # 352 bps -> 3.52
    out = unit_normalize_pred(352.0, 3.52, "yield", label_hint="Spread (bps)")
    assert out is not None
    assert abs(out - 3.52) < 0.01
    out2 = unit_normalize_pred(100.0, 1.0, "yield", label_hint="bps")
    assert out2 is not None
    assert abs(out2 - 1.0) < 0.01


def test_canonicalize_label():
    """GT and model labels normalized: lowercase, strip punctuation, collapse whitespace."""
    assert canonicalize_label("  Gross  Annualized  Debt (ITD)  ") == "gross annualized debt itd"
    assert canonicalize_label("Current Yield") == "current yield"
    assert canonicalize_label("") == ""


def test_gt_label_matching_column_name():
    """GT label matching column name (e.g. gross_annualized_debt_itd_portfolio_yield) resolves to canon value."""
    canon = {"gross_annualized_debt_itd_portfolio_yield": 0.163, "irr": None}
    # GT label exactly like column name (underscores)
    val = resolve_other_pred_from_canon(canon, "gross_annualized_debt_itd_portfolio_yield")
    assert val == 0.163
    # GT label with spaces and punctuation
    val2 = resolve_other_pred_from_canon(canon, "Gross Annualized Debt ITD Portfolio Yield")
    assert val2 == 0.163
    # Raw JSON with that key
    raw = {"gross_annualized_debt_itd_portfolio_yield": 0.163}
    canon_from_raw = normalize_prediction(raw)
    assert canon_from_raw.get("gross_annualized_debt_itd_portfolio_yield") == 0.163


def test_wolf_hill_scenario():
    """Wolf Hill: GT has ytd_whcm = -0.045, model returns null -> ytd_whcm_match = 0. overall_score == 0, score_denom == 1."""
    gt = {
        "net_irr": None,
        "net_moic": None,
        "net_dpi": None,
        "other_metric_label": "YTD WHCM",
        "other_metric_value": -0.045,
    }
    irr_match = moic_match = dpi_match = 0
    other_match = 0  # model returned null / mismatch
    (irr_in_scope, moic_in_scope, dpi_in_scope, other_in_scope, score_denom, _score_num, overall_score, no_metrics_in_scope, overall_score_is_na) = compute_score_from_gt_and_matches(
        gt, irr_match, moic_match, dpi_match, other_match
    )
    assert other_in_scope is True
    assert score_denom == 1
    assert overall_score == 0
    assert no_metrics_in_scope is False
    assert overall_score_is_na == 0


def test_no_gt_metrics():
    """GT has literally no metrics (all None). no_metrics_in_scope == True, overall_score blank, overall_score_is_na == 1."""
    gt = {
        "net_irr": None,
        "net_moic": None,
        "net_dpi": None,
        "other_metric_label": None,
        "other_metric_value": None,
    }
    (irr_in_scope, moic_in_scope, dpi_in_scope, other_in_scope, score_denom, _score_num, overall_score, no_metrics_in_scope, overall_score_is_na) = compute_score_from_gt_and_matches(
        gt, 0, 0, 0, 0
    )
    assert irr_in_scope is False and moic_in_scope is False and dpi_in_scope is False and other_in_scope is False
    assert score_denom == 0
    assert no_metrics_in_scope is True
    assert overall_score == ""
    assert overall_score_is_na == 1


def test_extract_all_performance_numbers_and_liberal_match():
    """Liberal matching: GT value can appear in top-level fields or inside array text."""
    # Top-level only
    data1 = {"net_irr": 15.5, "net_moic": 1.2, "net_dpi": None, "investment_performance": [], "key_takeaways": [], "business_updates": []}
    nums1 = extract_all_performance_numbers(data1)
    assert 15.5 in nums1 and 1.2 in nums1
    assert gt_value_appears_in_set(15.5, nums1, 0.01, allow_scale_flex=True)
    assert gt_value_appears_in_set(1.2, nums1, 0.01, allow_scale_flex=False)
    # Value only in investment_performance text (no exact field name)
    data2 = {"net_irr": None, "net_moic": None, "net_dpi": None, "investment_performance": ["Net IRR: 15.5%", "Some other line"], "key_takeaways": [], "business_updates": []}
    nums2 = extract_all_performance_numbers(data2)
    assert 15.5 in nums2
    assert gt_value_appears_in_set(15.5, nums2, 0.01, allow_scale_flex=True)
    # Scale flexibility: 0.155 (fraction) matches GT 15.5 (percent)
    assert gt_value_appears_in_set(15.5, [0.155], 0.01, allow_scale_flex=True)
    assert not gt_value_appears_in_set(15.5, [0.155], 0.01, allow_scale_flex=False)


if __name__ == "__main__":
    test_extract_json_strict()
    test_normalize_prediction_old_schema()
    test_normalize_prediction_new_schema()
    test_normalize_prediction_other_metric_current_yield()
    test_resolve_other_pred_from_canon()
    test_canonical_keys_cover_expected()
    test_unit_normalize_pred_gt_less_than_one_pred_greater()
    test_unit_normalize_pred_bps()
    test_canonicalize_label()
    test_gt_label_matching_column_name()
    test_wolf_hill_scenario()
    test_no_gt_metrics()
    test_extract_all_performance_numbers_and_liberal_match()
    print("All tests passed.")
    sys.exit(0)
