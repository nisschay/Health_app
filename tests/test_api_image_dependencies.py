import subprocess
import sys
import textwrap

UI_ONLY_PACKAGES = ("streamlit", "plotly")

PROBE = textwrap.dedent(
    """
    import sys
    from importlib.abc import MetaPathFinder

    BLOCKED = {blocked!r}

    class Blocker(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in BLOCKED:
                raise ImportError(fullname + " is not installed in the API image")
            return None

    sys.meta_path.insert(0, Blocker())
    for name in list(sys.modules):
        if name.split(".")[0] in BLOCKED:
            del sys.modules[name]

    import main  # noqa: F401
    import Helper_Functions as helpers
    import pandas as pd

    assert helpers.st is None and helpers.go is None
    assert helpers._has_streamlit_context() is False
    assert helpers._canonical_status("Abnormal") == "Flagged"

    df = pd.DataFrame([{{
        "Test_Name": "Haemoglobin (Hb)", "Test_Category": "Haematology",
        "Test_Date": "05-03-2024", "Result": "12.4", "Unit": "g/dL",
        "Reference_Range": "12 - 15", "Status": "Normal",
    }}])
    assert len(helpers.generate_pdf_health_report(df, {{"name": "T"}}, api_key=None)) > 0
    print("ok")
    """
)


def test_backend_runs_without_the_ui_packages():
    """requirements.txt ships neither; importing either on a cold start is a regression."""
    result = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", PROBE.format(blocked=set(UI_ONLY_PACKAGES))],
        capture_output=True,
        text=True,
        env={"DATABASE_URL": "sqlite://", "GEMINI_API_KEY": "test-key-not-used", "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "ok" in result.stdout


def test_ui_packages_are_not_in_the_api_requirements():
    requirements = open("requirements.txt").read().lower()
    for package in UI_ONLY_PACKAGES:
        assert package not in requirements, f"{package} belongs in requirements-ui.txt"
