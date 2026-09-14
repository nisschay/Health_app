import io

import pytest
from fastapi import HTTPException, UploadFile

from backend_api.app.config import settings
from backend_api.app.security import read_existing_data_upload, read_pdf_uploads

PDF = b"%PDF-1.7\n" + b"x" * 64


def _upload(name: str, payload: bytes) -> UploadFile:
    return UploadFile(filename=name, file=io.BytesIO(payload))


@pytest.mark.asyncio
async def test_accepts_a_real_pdf():
    payloads = await read_pdf_uploads([_upload("report.pdf", PDF)])
    assert payloads == [("report.pdf", PDF)]


@pytest.mark.asyncio
async def test_rejects_a_file_that_is_not_a_pdf():
    """A renamed .exe used to be handed straight to the PDF parser."""
    with pytest.raises(HTTPException) as excinfo:
        await read_pdf_uploads([_upload("report.pdf", b"MZ\x90\x00not a pdf")])
    assert excinfo.value.status_code == 415


@pytest.mark.asyncio
async def test_rejects_a_file_over_the_per_file_limit():
    oversized = PDF + b"0" * (settings.max_upload_file_mb * 1024 * 1024)
    with pytest.raises(HTTPException) as excinfo:
        await read_pdf_uploads([_upload("big.pdf", oversized)])
    assert excinfo.value.status_code == 413


@pytest.mark.asyncio
async def test_rejects_too_many_files():
    uploads = [_upload(f"r{i}.pdf", PDF) for i in range(settings.max_upload_files + 1)]
    with pytest.raises(HTTPException) as excinfo:
        await read_pdf_uploads(uploads)
    assert excinfo.value.status_code == 413


@pytest.mark.asyncio
async def test_rejects_a_batch_over_the_total_limit():
    chunk = PDF + b"0" * (settings.max_upload_file_mb * 1024 * 1024 - len(PDF) - 1)
    uploads = [_upload(f"r{i}.pdf", chunk) for i in range(settings.max_upload_total_mb + 1)]
    with pytest.raises(HTTPException) as excinfo:
        await read_pdf_uploads(uploads)
    assert excinfo.value.status_code == 413


@pytest.mark.asyncio
async def test_no_uploads_is_not_an_error():
    assert await read_pdf_uploads(None) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["data.csv", "data.xlsx", "DATA.XLSX"])
async def test_accepts_spreadsheet_extensions(name):
    result = await read_existing_data_upload(_upload(name, b"a,b\n1,2\n"))
    assert result is not None


@pytest.mark.asyncio
async def test_rejects_other_extensions_for_existing_data():
    with pytest.raises(HTTPException) as excinfo:
        await read_existing_data_upload(_upload("payload.exe", b"MZ"))
    assert excinfo.value.status_code == 415
