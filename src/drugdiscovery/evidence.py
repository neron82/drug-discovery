"""Evidence engine — multi-dimensional, inspectable evidence scoring.

Never collapses to one opaque number: returns component breakdown (independent sources,
human/animal/in-vitro/clinical mass, replication, contradiction ratio, recency, model
relevance) plus an overall confidence that is explainable from those parts.
"""
from __future__ import annotations

# quality weights by context / species / statement evidence level
_CONTEXT_W = {"clinical": 1.0, "ex_vivo": 0.7, "animal_model": 0.6, "cell_culture": 0.5,
              "in_silico": 0.2, "unknown": 0.3, "human_tissue": 0.7}
_SPECIES_W = {"human": 1.0, "mouse": 0.6, "rat": 0.6, "in_silico": 0.2, "unknown": 0.4}
_LEVEL_W = {"established": 0.95, "probable": 0.75, "speculative": 0.45, "computational": 0.2}


def _stat_weight(stmt) -> float:
    ctx = _CONTEXT_W.get(getattr(stmt, "model_context", "unknown") or "unknown", 0.3)
    sp = _SPECIES_W.get(getattr(stmt, "species", "unknown") or "unknown", 0.4)
    lvl = _LEVEL_W.get(getattr(stmt, "evidence_level", "speculative") or "speculative", 0.45)
    conf = getattr(stmt, "confidence", 0.5) or 0.5
    return round(ctx * sp * lvl * (0.5 + 0.5 * conf), 4)


def _recency_score(years: list[int], current_year: int) -> float:
    if not years:
        return 0.3
    newest = max(years)
    age = max(0, current_year - newest)
    return round(max(0.0, 1.0 - age / 25.0), 3)


def score_evidence(support_stmts, contradict_stmts, current_year: int | None = None) -> dict:
    """Compute a transparent evidence profile from two statement lists.

    Each statement must expose: evidence_level, model_context, species, confidence,
    and its paper.year.
    """
    from datetime import datetime
    cy = current_year or datetime.now().year

    support = list(support_stmts or [])
    contra = list(contradict_stmts or [])

    def paper_year(st):
        p = getattr(st, "paper", None)
        if p is not None:
            return getattr(p, "year", None)
        return None

    s_weights = [_stat_weight(s) for s in support]
    c_weights = [_stat_weight(c) for c in contra]
    pos_mass = round(sum(s_weights), 4)
    neg_mass = round(sum(c_weights), 4)
    total = pos_mass + neg_mass
    support_ratio = round(pos_mass / total, 4) if total else 0.0
    contradiction_ratio = round(neg_mass / total, 4) if total else 0.0

    distinct_papers = set()
    study_types = set()
    species = set()
    years = []
    for s in support:
        pid = getattr(s.paper, "id", None) if s.paper is not None else None
        if pid:
            distinct_papers.add(pid)
        study_types.add(getattr(s, "model_context", "unknown") or "unknown")
        species.add(getattr(s, "species", "unknown") or "unknown")
        y = paper_year(s)
        if y:
            years.append(y)

    rep = len(distinct_papers)
    recency = _recency_score(years, cy)

    # model relevance: human + clinical/ex-vivo/human-tissue presence
    human_mark = 1.0 if "human" in species else 0.4
    ctx_mark = 1.0 if any(x in study_types for x in ("clinical", "ex_vivo", "human_tissue")) else 0.5
    model_relevance = round(0.5 * human_mark + 0.5 * ctx_mark, 3)

    # prevalence of study types across all evidence
    all_study_types = set(study_types)
    for c in contra:
        all_study_types.add(getattr(c, "model_context", "unknown") or "unknown")

    confidence = (0.5 * support_ratio + 0.3 * min(1.0, rep / 4.0)
                  + 0.1 * recency + 0.1 * model_relevance - 0.3 * contradiction_ratio)
    confidence = round(max(0.0, min(1.0, confidence)), 3)

    return {
        "independent_sources": len(distinct_papers),
        "supporting_statements": len(support),
        "contradicting_statements": len(contra),
        "positive_evidence_mass": pos_mass,
        "negative_evidence_mass": neg_mass,
        "support_ratio": support_ratio,
        "contradiction_ratio": contradiction_ratio,
        "replication": rep,
        "study_types": sorted(all_study_types),
        "species": sorted(species),
        "clinical_count": sum(1 for s in support + contra if getattr(s, "model_context", "") == "clinical"),
        "animal_count": sum(1 for s in support + contra if getattr(s, "model_context", "") == "animal_model"),
        "in_vitro_count": sum(1 for s in support + contra if getattr(s, "model_context", "") == "cell_culture"),
        "recent_years": sorted(set(years), reverse=True)[:5],
        "recency": recency,
        "model_relevance": model_relevance,
        "overall_confidence": confidence,
    }
