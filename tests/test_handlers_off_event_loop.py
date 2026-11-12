import inspect

from backend_api.app import main


def test_blocking_export_handlers_are_sync_so_fastapi_threads_them():
    """An async def doing minutes of work stalls every other request, including /health."""
    assert not inspect.iscoroutinefunction(main.export_excel)
    assert not inspect.iscoroutinefunction(main.export_pdf)


def test_job_creation_hands_the_work_to_the_pool():
    """The request only reads the upload; extraction runs in the job pool and the client polls."""
    source = inspect.getsource(main.create_job)
    assert "jobs.submit(" in source
    assert "analyze_reports(" not in source


def test_job_polling_is_sync_so_fastapi_threads_it():
    assert not inspect.iscoroutinefunction(main.read_job)
