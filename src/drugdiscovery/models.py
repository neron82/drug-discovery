"""Persistent data model — knowledge graph, papers, claims, hypotheses, experiment queue, timeline.

Design invariants:
  * Every graph edge carries provenance + knowledge_type; never silent edges.
  * Knowledge types: SOURCE_FACT | MODEL_INFERENCE | COMPUTATIONAL_PREDICTION | HYPOTHESIS
  * Evidence levels are separated (established/probable/speculative/computational) and
    never collapsed into one opaque score — evidence profiles live in `evidence_profile` JSON.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Enum,
)
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class NodeKind(str, __import__("enum").Enum):
    DISEASE = "disease"
    PHENOTYPE = "phenotype"
    CELL_TYPE = "cell_type"
    TISSUE = "tissue"
    BIOLOGICAL_PROCESS = "biological_process"
    PATHWAY = "pathway"
    GENE = "gene"
    PROTEIN = "protein"
    RECEPTOR = "receptor"
    ENZYME = "enzyme"
    TRANSCRIPTION_FACTOR = "transcription_factor"
    LIGAND = "ligand"  # endogenous signaling molecule / hormone / metabolite
    METABOLITE = "metabolite"
    HORMONE = "hormone"
    COMPOUND = "compound"
    DRUG = "drug"
    DRUG_CLASS = "drug_class"
    BINDING_SITE = "binding_site"
    INTERACTION = "interaction"
    PUBLICATION = "publication"
    DATASET = "dataset"
    EXPERIMENT = "experiment"
    SIMULATION = "simulation"
    HYPOTHESIS = "hypothesis"
    QUESTION = "question"
    OTHER = "other"


# Semantic edge types (directional).
class EdgeKind(str, __import__("enum").Enum):
    INHIBITS = "INHIBITS"
    ACTIVATES = "ACTIVATES"
    AGONIST_OF = "AGONIST_OF"
    ANTAGONIST_OF = "ANTAGONIST_OF"
    REGULATES = "REGULATES"
    UPREGULATES = "UPREGULATES"
    DOWNREGULATES = "DOWNREGULATES"
    INDUCES = "INDUCES"
    INCREASES = "INCREASES"
    DECREASES = "DECREASES"
    PART_OF = "PART_OF"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    INVOLVES = "INVOLVES"
    EXPRESSED_IN = "EXPRESSED_IN"
    METABOLIZED_BY = "METABOLIZED_BY"
    CONVERTS_TO = "CONVERTS_TO"
    TARGETS = "TARGETS"
    BINDS_TO = "BINDS_TO"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    PREDICTS_BINDING_TO = "PREDICTS_BINDING_TO"
    ENCODES = "ENCODES"
    HOMOLOG_OF = "HOMOLOG_OF"
    RELATED_TO = "RELATED_TO"


class KnowledgeType(str, __import__("enum").Enum):
    SOURCE_FACT = "source_fact"
    MODEL_INFERENCE = "model_inference"
    COMPUTATIONAL_PREDICTION = "computational_prediction"
    HYPOTHESIS = "hypothesis"


class EvidenceLevel(str, __import__("enum").Enum):
    ESTABLISHED = "established"      # replicated across independent human/solid evidence
    PROBABLE = "probable"
    SPECULATIVE = "speculative"
    COMPUTATIONAL = "computational"  # model-generated, not yet validated


class ResearchState(str, __import__("enum").Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"


class ExperimentKind(str, __import__("enum").Enum):
    LITERATURE_SEARCH = "literature_search"
    PAPER_EXTRACTION = "paper_extraction"
    TARGET_INVESTIGATION = "target_investigation"
    DATABASE_QUERY = "database_query"
    CHEMICAL_SEARCH = "chemical_search"
    DOCKING = "docking"
    STRUCTURE_ANALYSIS = "structure_analysis"
    PROPERTY_PREDICTION = "property_prediction"
    MOLECULAR_DYNAMICS = "molecular_dynamics"
    NETWORK_ANALYSIS = "network_analysis"
    HYPOTHESIS_CRITIQUE = "hypothesis_critique"
    CONTRADICTION_SEARCH = "contradiction_search"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    QUESTION_GENERATION = "question_generation"
    KG_EMBED = "kg_embed"


class JobStatus(str, __import__("enum").Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Workspace (disease-agnostic project container)
# ---------------------------------------------------------------------------
class Workspace(Base):
    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(160))
    slug: Mapped[str] = mapped_column(String(160), unique=True)
    disease: Mapped[str] = mapped_column(String(160), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    config_json: Mapped[dict] = mapped_column(JSON, default=dict)  # disease-specific: tissues, cell types, phenotypes
    research_state: Mapped[str] = mapped_column(String(16), default=ResearchState.CREATED.value)
    research_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    budgets_json: Mapped[dict] = mapped_column(JSON, default=dict)
    state_json: Mapped[dict] = mapped_column(JSON, default=dict)  # open questions, priorities, counters
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    nodes = relationship("Node", back_populates="workspace", cascade="all, delete-orphan")
    edges = relationship("Edge", back_populates="workspace", cascade="all, delete-orphan")
    statements = relationship("Statement", back_populates="workspace", cascade="all, delete-orphan")
    papers = relationship("Paper", back_populates="workspace", cascade="all, delete-orphan")
    hypotheses = relationship("Hypothesis", back_populates="workspace", cascade="all, delete-orphan")
    experiments = relationship("Experiment", back_populates="workspace", cascade="all, delete-orphan")
    timeline = relationship("TimelineEntry", back_populates="workspace", cascade="all, delete-orphan")


# ---------------------------------------------------------------------------
# Knowledge graph
# ---------------------------------------------------------------------------
class Node(Base):
    __tablename__ = "nodes"
    __table_args__ = (UniqueConstraint("workspace_id", "kind", "canonical_name", name="uq_node_ws_kind_name"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(40), index=True)
    canonical_name: Mapped[str] = mapped_column(String(200))
    label: Mapped[str] = mapped_column(String(200), default="")
    synonyms: Mapped[list] = mapped_column(JSON, default=list)
    properties: Mapped[dict] = mapped_column(JSON, default=dict)
    knowledge_type: Mapped[str] = mapped_column(String(32), default=KnowledgeType.SOURCE_FACT.value)
    evidence_level: Mapped[str] = mapped_column(String(24), default=EvidenceLevel.SPECULATIVE.value)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    summary: Mapped[str] = mapped_column(Text, default="")
    source_hash: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    workspace = relationship("Workspace", back_populates="nodes")
    out_edges = relationship("Edge", back_populates="source", foreign_keys="Edge.source_id", cascade="all, delete-orphan")
    in_edges = relationship("Edge", back_populates="target", foreign_keys="Edge.target_id", cascade="all, delete-orphan")


class Edge(Base):
    __tablename__ = "edges"
    __table_args__ = (
        UniqueConstraint("workspace_id", "source_id", "type", "target_id", "knowledge_type", name="uq_edge"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"), index=True)
    source_id: Mapped[str] = mapped_column(ForeignKey("nodes.id", ondelete="CASCADE"), index=True)
    target_id: Mapped[str] = mapped_column(ForeignKey("nodes.id", ondelete="CASCADE"), index=True)
    type: Mapped[str] = mapped_column(String(40), index=True)
    directional: Mapped[bool] = mapped_column(Boolean, default=True)
    knowledge_type: Mapped[str] = mapped_column(String(32), default=KnowledgeType.SOURCE_FACT.value)
    evidence_level: Mapped[str] = mapped_column(String(24), default=EvidenceLevel.SPECULATIVE.value)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    properties: Mapped[dict] = mapped_column(JSON, default=dict)
    provenance: Mapped[list] = mapped_column(JSON, default=list)  # [{paper_id, statement_id, kind}]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    workspace = relationship("Workspace", back_populates="edges")
    source = relationship("Node", back_populates="out_edges", foreign_keys=[source_id])
    target = relationship("Node", back_populates="in_edges", foreign_keys=[target_id])


# ---------------------------------------------------------------------------
# Papers + structured claims/statements
# ---------------------------------------------------------------------------
class Paper(Base):
    __tablename__ = "papers"
    # dedupe by content hash (pmid/doi often empty for preprints) — truly unique per paper content
    __table_args__ = (UniqueConstraint("workspace_id", "content_hash", name="uq_paper_ws_hash"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"), index=True)
    pmid: Mapped[str] = mapped_column(String(32), default="")
    doi: Mapped[str] = mapped_column(String(120), default="")
    title: Mapped[str] = mapped_column(Text, default="")
    abstract: Mapped[str] = mapped_column(Text, default="")
    authors_json: Mapped[list] = mapped_column(JSON, default=list)
    journal: Mapped[str] = mapped_column(String(200), default="")
    year: Mapped[int] = mapped_column(Integer, nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="europepmc")
    url: Mapped[str] = mapped_column(String(400), default="")
    content_hash: Mapped[str] = mapped_column(String(64), default="")
    status: Mapped[str] = mapped_column(String(24), default="fetched")  # fetched|extracted|failed
    extractor: Mapped[str] = mapped_column(String(80), default="")
    fetch_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    workspace = relationship("Workspace", back_populates="papers")
    statements = relationship("Statement", back_populates="paper", cascade="all, delete-orphan")


class Statement(Base):
    """A single groundable claim/citation extracted from a paper (or stored evidence)."""
    __tablename__ = "statements"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"), index=True)
    source_paper_id: Mapped[str | None] = mapped_column(ForeignKey("papers.id", ondelete="SET NULL"), nullable=True)
    text: Mapped[str] = mapped_column(Text)
    claim_type: Mapped[str] = mapped_column(String(40), default="mechanism")  # mechanism|target_relationship|compound_target|expression|clinical|association
    relation: Mapped[str] = mapped_column(String(40), default="")
    subject: Mapped[str] = mapped_column(String(160), default="")
    object: Mapped[str] = mapped_column(String(160), default="")
    direction: Mapped[str] = mapped_column(String(24), default="modulate")  # increase|decrease|modulate|none
    effect_size: Mapped[float | None] = mapped_column(Float, nullable=True)
    significance: Mapped[str] = mapped_column(String(40), default="")
    species: Mapped[str] = mapped_column(String(40), default="unknown")  # human|mouse|rat|in_silico|unknown
    model_context: Mapped[str] = mapped_column(String(60), default="unknown")  # clinical|animal_model|cell_culture|ex_vivo|in_silico
    evidence_level: Mapped[str] = mapped_column(String(24), default=EvidenceLevel.SPECULATIVE.value)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    knowledge_type: Mapped[str] = mapped_column(String(32), default=KnowledgeType.SOURCE_FACT.value)
    support_citations: Mapped[list] = mapped_column(JSON, default=list)
    contradict_citations: Mapped[list] = mapped_column(JSON, default=list)
    properties: Mapped[dict] = mapped_column(JSON, default=dict)
    quotation: Mapped[str] = mapped_column(Text, default="")
    extractor_model: Mapped[str] = mapped_column(String(80), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    workspace = relationship("Workspace", back_populates="statements")
    paper = relationship("Paper", back_populates="statements")


# ---------------------------------------------------------------------------
# Hypotheses
# ---------------------------------------------------------------------------
class Hypothesis(Base):
    __tablename__ = "hypotheses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(240))
    body: Mapped[str] = mapped_column(Text)
    mechanism_reasoning: Mapped[str] = mapped_column(Text, default="")
    predicted_effect: Mapped[str] = mapped_column(Text, default="")
    target_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    pathway_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    supporting_evidence: Mapped[list] = mapped_column(JSON, default=list)  # statement ids + summaries
    counter_evidence: Mapped[list] = mapped_column(JSON, default=list)
    uncertainty: Mapped[str] = mapped_column(Text, default="")
    evidence_profile: Mapped[dict] = mapped_column(JSON, default=dict)
    testable_predictions: Mapped[list] = mapped_column(JSON, default=list)
    next_experiment: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(24), default="active")  # active|promising|speculative|rejected|closed
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    source: Mapped[str] = mapped_column(String(80), default="llm")
    source_knowledge_type: Mapped[str] = mapped_column(String(32), default=KnowledgeType.HYPOTHESIS.value)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    workspace = relationship("Workspace", back_populates="hypotheses")


# ---------------------------------------------------------------------------
# Experiment / job queue (compute orchestration)
# ---------------------------------------------------------------------------
class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(40), index=True)
    question: Mapped[str] = mapped_column(Text, default="")
    reason: Mapped[str] = mapped_column(Text, default="")
    expected_information_gain: Mapped[float] = mapped_column(Float, default=0.0)
    scientific_importance: Mapped[float] = mapped_column(Float, default=0.0)
    confidence_impact: Mapped[float] = mapped_column(Float, default=0.0)
    computational_cost: Mapped[float] = mapped_column(Float, default=1.0)
    priority: Mapped[float] = mapped_column(Float, default=0.0, index=True)
    status: Mapped[str] = mapped_column(String(16), default=JobStatus.QUEUED.value, index=True)
    params_json: Mapped[dict] = mapped_column(JSON, default=dict)
    dependencies: Mapped[list] = mapped_column(JSON, default=list)
    result_summary: Mapped[str] = mapped_column(Text, default="")
    result_json: Mapped[dict] = mapped_column(JSON, default=dict)
    worker: Mapped[str] = mapped_column(String(60), default="")
    error: Mapped[str] = mapped_column(Text, default="")
    log_file: Mapped[str] = mapped_column(String(300), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    workspace = relationship("Workspace", back_populates="experiments")


# ---------------------------------------------------------------------------
# Research timeline / activity log
# ---------------------------------------------------------------------------
class TimelineEntry(Base):
    __tablename__ = "timeline"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    agent: Mapped[str] = mapped_column(String(40), default="system")  # researcher|biologist|chemist|structural|simulation|skeptic|curator|director|system
    level: Mapped[str] = mapped_column(String(16), default="info")  # info|success|warn|error
    message: Mapped[str] = mapped_column(Text)
    data_json: Mapped[dict] = mapped_column(JSON, default=dict)

    workspace = relationship("Workspace", back_populates="timeline")


# ---------------------------------------------------------------------------
# Compounds (Phase 2 — placeholder schema so structure arrives early)
# ---------------------------------------------------------------------------
class Compound(Base):
    __tablename__ = "compounds"
    __table_args__ = (UniqueConstraint("inchikey", name="uq_compound_inchikey"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    canonical_name: Mapped[str] = mapped_column(String(200), index=True)
    synonyms: Mapped[list] = mapped_column(JSON, default=list)
    canonical_smiles: Mapped[str] = mapped_column(Text, default="")
    inchi: Mapped[str] = mapped_column(Text, default="")
    inchikey: Mapped[str] = mapped_column(String(27), default="")
    formula: Mapped[str] = mapped_column(String(80), default="")
    molmass: Mapped[float | None] = mapped_column(Float, nullable=True)
    logp: Mapped[float | None] = mapped_column(Float, nullable=True)
    tpsa: Mapped[float | None] = mapped_column(Float, nullable=True)
    hbd: Mapped[int | None] = mapped_column(Integer, nullable=True)
    hba: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rotb: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sources_json: Mapped[dict] = mapped_column(JSON, default=dict)
    properties_json: Mapped[dict] = mapped_column(JSON, default=dict)
    research_status: Mapped[str] = mapped_column(String(24), default="catalogued")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


# ---------------------------------------------------------------------------
# Protein structures (Phase 3) — PDB experimental + AlphaFold predicted
# ---------------------------------------------------------------------------
class Structure(Base):
    __tablename__ = "structures"
    __table_args__ = (UniqueConstraint("source", "source_id", name="uq_structure_source_id"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    source: Mapped[str] = mapped_column(String(16), index=True)      # pdb | alphafold
    source_id: Mapped[str] = mapped_column(String(32), index=True)   # 4-digit PDB id / AF-P10275-F1
    protein_node_id: Mapped[str | None] = mapped_column(ForeignKey("nodes.id", ondelete="SET NULL"), nullable=True, index=True)
    workspace_id: Mapped[str] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"), index=True)
    uniprot_accession: Mapped[str] = mapped_column(String(16), default="", index=True)
    method: Mapped[str] = mapped_column(String(24), default="")      # x-ray | cryo-em | nmr | predicted
    resolution: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)  # pLDDT mean for AF
    organism: Mapped[str] = mapped_column(String(80), default="")
    chains: Mapped[list] = mapped_column(JSON, default=list)
    ligands: Mapped[list] = mapped_column(JSON, default=list)        # [{pdb_id,name,compound_id}]
    file_path: Mapped[str] = mapped_column(String(500), default="")  # local cif/pdb artifact
    file_hash: Mapped[str] = mapped_column(String(64), default="")   # sha256 of file
    limitations: Mapped[str] = mapped_column(Text, default="")
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
