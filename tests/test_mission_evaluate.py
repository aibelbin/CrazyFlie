from drone_follow.mission_evaluate import make_policy, markdown_table, run_legs


def test_straight_line_succeeds_on_empty_short_legs():
    r = run_legs(make_policy("straight"), level=0, legs=2, length=25.0)
    assert r["legs"] == 2 and r["success_rate"] == 1.0 and r["crash_rate"] == 0.0
    assert 0.0 <= r["extra_distance"] < 0.15 and r["guided_fraction"] == 0.0


def test_supervised_avoider_reports_guided_time_and_outcomes_add_up():
    r = run_legs(make_policy("bendy"), level=3, legs=2, length=60.0)
    assert r["crash_rate"] + r["success_rate"] + r["failure_rate"] == 1.0
    assert r["guided_fraction"] > 0.0


def test_markdown_table_lists_every_result():
    rows = [dict(policy="bendy", level=2, legs=500, crash_rate=0.04, success_rate=0.9, failure_rate=0.06,
                 extra_distance=0.12, time_ratio=1.3, guided_fraction=0.2, mean_action_change=0.05,
                 p95_accel_ms2=3.0, crash_kinds={"tree": 20}, failures={"timeout": 30})]
    table = markdown_table(rows)
    assert "| bendy | 2 | 4.0% | 90.0% |" in table and table.count("\n") == 2
