import inspect

from backend_api.app import main


def test_blocking_export_handlers_are_sync_so_fastapi_threads_them():
    """An async def doing minutes of work stalls every other request, including /health."""
    assert not inspect.iscoroutinefunction(main.export_excel)
    assert not inspect.iscoroutinefunction(main.export_pdf)


def test_analyze_offloads_its_blocking_work():
    assert "run_in_threadpool(" in inspect.getsource(main.analyze_reports)


def test_stream_worker_runs_in_its_own_thread():
    assert "threading.Thread(" in inspect.getsource(main.analyze_reports_stream)
