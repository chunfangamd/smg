import importlib.util
import json
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "bfcl_run_ab", Path(__file__).resolve().parents[1] / "bfcl" / "run_ab.py"
)
run_ab = importlib.util.module_from_spec(SPEC)
sys.modules["bfcl_run_ab"] = run_ab
SPEC.loader.exec_module(run_ab)


def write_score(score_root, section, category, accuracy, total, prefix="BFCL_v4_"):
    path = score_root / "Qwen_Qwen3-4B-FC" / section / f"{prefix}{category}_score.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {"accuracy": accuracy, "correct_count": round(accuracy * total), "total_count": total}
    path.write_text(json.dumps(summary) + "\n" + json.dumps({"id": f"{category}_0"}) + "\n")


def test_a_category_is_not_read_from_a_longer_category_with_the_same_suffix(tmp_path):
    write_score(tmp_path, "non_live", "parallel_multiple", 0.72, 200)
    write_score(tmp_path, "live", "live_multiple", 0.60, 1053)
    write_score(tmp_path, "live", "live_parallel_multiple", 0.50, 24)
    write_score(tmp_path, "live", "live_parallel", 0.55, 16)
    write_score(tmp_path, "live", "live_irrelevance", 0.80, 884)

    assert run_ab._find_category_summary(tmp_path, "multiple") is None
    assert run_ab._find_category_summary(tmp_path, "parallel") is None
    assert run_ab._find_category_summary(tmp_path, "irrelevance") is None


def test_each_category_reads_its_own_score_file(tmp_path):
    scores = {
        "multiple": ("non_live", 0.865, 200),
        "parallel_multiple": ("non_live", 0.72, 200),
        "parallel": ("non_live", 0.71, 200),
        "irrelevance": ("non_live", 0.8333, 240),
        "live_multiple": ("live", 0.60, 1053),
        "live_parallel_multiple": ("live", 0.50, 24),
        "live_parallel": ("live", 0.55, 16),
        "live_irrelevance": ("live", 0.80, 884),
    }
    for category, (section, accuracy, total) in scores.items():
        write_score(tmp_path, section, category, accuracy, total)

    for category, (_, accuracy, total) in scores.items():
        assert run_ab._find_category_summary(tmp_path, category) == (accuracy, total)


def test_score_files_without_the_version_prefix_still_match(tmp_path):
    write_score(tmp_path, "non_live", "multiple", 0.865, 200, prefix="")
    write_score(tmp_path, "non_live", "parallel_multiple", 0.72, 200, prefix="")

    assert run_ab._find_category_summary(tmp_path, "multiple") == (0.865, 200)


CATEGORIES = ["simple_python", "irrelevance"]


def write_run_report(path, baseline, candidate, timed_out=()):
    counts = {category: 200 for category in CATEGORIES}
    base = run_ab.Arm(name="vllm", base_url="", scores=baseline, counts=counts)
    cand = run_ab.Arm(
        name="smg", base_url="", scores=candidate, counts=counts, timed_out=list(timed_out)
    )
    _, payload = run_ab.build_report(base, cand, CATEGORIES)
    path.write_text(json.dumps(payload))
    return path


def combine(monkeypatch, tmp_path, reports):
    out_md, out_json = tmp_path / "ab.md", tmp_path / "ab.json"
    argv = ["run_ab.py", "--combine", *map(str, reports), "--categories", ",".join(CATEGORIES)]
    monkeypatch.setattr(sys, "argv", argv + ["--out", str(out_md), "--json-out", str(out_json)])
    code = run_ab.main()
    return code, out_md.read_text(), json.loads(out_json.read_text())


def test_runs_are_averaged_per_category_and_gated_on_the_mean(monkeypatch, tmp_path):
    first = write_run_report(
        tmp_path / "run0.json",
        {"simple_python": 0.90, "irrelevance": 0.80},
        {"simple_python": 0.86, "irrelevance": 0.78},
    )
    second = write_run_report(
        tmp_path / "run1.json",
        {"simple_python": 0.88, "irrelevance": 0.80},
        {"simple_python": 0.90, "irrelevance": 0.82},
    )

    code, report, payload = combine(monkeypatch, tmp_path, [first, second])

    # The first run alone (-3.00) would fail the 2-point gate; the mean (-0.50) does not.
    assert code == run_ab.EXIT_OK
    assert payload["per_category"][0]["baseline"] == pytest.approx(0.89)
    assert payload["overall"]["delta"] == pytest.approx(-0.005)
    assert "(-3.00)" in report and "(+2.00)" in report
    assert "-3.00 to +2.00" in report


def test_a_mean_drop_beyond_the_tolerance_is_a_regression(monkeypatch, tmp_path):
    first = write_run_report(
        tmp_path / "run0.json",
        {"simple_python": 0.90, "irrelevance": 0.80},
        {"simple_python": 0.86, "irrelevance": 0.78},
    )
    second = write_run_report(
        tmp_path / "run1.json",
        {"simple_python": 0.90, "irrelevance": 0.80},
        {"simple_python": 0.87, "irrelevance": 0.78},
    )

    code, _, payload = combine(monkeypatch, tmp_path, [first, second])

    assert code == run_ab.EXIT_REGRESSION
    assert payload["overall"]["delta"] == pytest.approx(-0.0275)


def test_a_run_without_a_report_makes_the_result_incomplete(monkeypatch, tmp_path):
    first = write_run_report(
        tmp_path / "run0.json",
        {"simple_python": 0.90, "irrelevance": 0.80},
        {"simple_python": 0.90, "irrelevance": 0.80},
    )

    code, report, payload = combine(monkeypatch, tmp_path, [first, tmp_path / "run1.json"])

    assert code == run_ab.EXIT_INCOMPLETE
    assert payload["incomplete_runs"] == {"2": "no report"}
    assert "Incomplete run 2" in report


def test_an_incomplete_run_makes_the_result_incomplete(monkeypatch, tmp_path):
    first = write_run_report(
        tmp_path / "run0.json",
        {"simple_python": 0.90, "irrelevance": 0.80},
        {"simple_python": 0.90, "irrelevance": 0.80},
    )
    second = write_run_report(
        tmp_path / "run1.json",
        {"simple_python": 0.90, "irrelevance": 0.80},
        {"simple_python": 0.90},
        timed_out=["generate"],
    )

    code, _, payload = combine(monkeypatch, tmp_path, [first, second])

    assert code == run_ab.EXIT_INCOMPLETE
    assert "timed out during: generate" in payload["incomplete_runs"]["2"]
    # The category the second run missed still has the first run's score.
    assert payload["per_category"][1]["candidate"] == pytest.approx(0.80)
