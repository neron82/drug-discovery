"""Literature ingestion regression tests using synthetic records and isolated storage."""
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from drugdiscovery import db, repos, research
from drugdiscovery.config import settings
from drugdiscovery.lit.providers import PaperRecord
from drugdiscovery.models import Experiment, JobStatus, Paper, Workspace


@pytest.fixture(autouse=True)
def clean_db(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.storage, "db_path", str(tmp_path / "research.db"))
    monkeypatch.setattr(settings.storage, "data_dir", str(tmp_path))
    monkeypatch.setattr(settings.storage, "artifact_dir", str(tmp_path / "artifacts"))
    monkeypatch.setattr(settings.storage, "log_dir", str(tmp_path / "logs"))
    db._engine = None
    db._SessionLocal = None
    try:
        db.init_db()
        yield
    finally:
        if db._engine is not None:
            db._engine.dispose()
        db._engine = None
        db._SessionLocal = None


@pytest.mark.parametrize("kind", ["literature_search", "contradiction_search"])
def test_doi_only_ingestion_queues_persisted_papers_without_duplicates(monkeypatch, kind):
    records = [
        PaperRecord(
            pmid="", doi=f"10.0000/synthetic-ingestion-{index}",
            title=f"Synthetic ingestion fixture {index}", abstract=abstract,
            authors=[], journal="", year=None, source="synthetic-test", url="",
        )
        for index, abstract in enumerate([
            "Synthetic abstract for first ingestion fixture.",
            "Synthetic abstract for second ingestion fixture.",
            "",
        ])
    ]
    provider = SimpleNamespace(search=lambda query, size: records)
    monkeypatch.setattr(research, "get_provider", lambda name: provider)
    expected_records = {
        (record.doi, record.title, record.abstract)
        for record in records
        if kind == "literature_search" or record.abstract
    }

    with db.get_session() as session:
        ws = repos.create_workspace(session, "Synthetic ingestion", "test-only")
        experiment = repos.create_experiment(
            session, ws.id, kind,
            params={"query": "synthetic test", "topic": "synthetic test"},
        )
        workspace_id, experiment_id = ws.id, experiment.id
        session.commit()

    queued_ids = None
    for expected_new in (len(records), 0):
        with db.get_session() as session:
            ws = session.get(Workspace, workspace_id)
            experiment = session.get(Experiment, experiment_id)
            _, result = research.execute_experiment(session, ws, experiment, llm=None)
            assert result["new"] == expected_new
            session.commit()

        # Resolve the persisted queue in a fresh session, as an extraction worker would.
        with db.get_session() as session:
            jobs = session.execute(select(Experiment).where(
                Experiment.workspace_id == workspace_id,
                Experiment.kind == "paper_extraction",
            )).scalars().all()
            assert len(jobs) == len(expected_records)
            resolved_records = set()
            for job in jobs:
                assert job.status == JobStatus.QUEUED.value
                paper_id = job.params_json.get("paper_id")
                assert paper_id
                paper = session.get(Paper, paper_id)
                assert paper is not None
                assert paper.workspace_id == workspace_id
                assert not paper.pmid
                resolved_records.add((paper.doi, paper.title, paper.abstract))
            assert resolved_records == expected_records
            assert repos.paper_count(session, workspace_id) == len(records)
            if queued_ids is None:
                queued_ids = {job.id for job in jobs}
            else:
                assert {job.id for job in jobs} == queued_ids
