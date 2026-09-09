"""Compound database providers (PubChem / ChEMBL).

Demonstrates the CompoundDatabaseProvider adapter seam: these are local-friendly,
public, keyless REST APIs. Providers are swappable behind a registry — no other
module may talk to these services directly (API-first, single source of truth).
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

log = logging.getLogger("dd.compound")

PUBCHEM = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"


class CompoundFetchError(RuntimeError):
    pass


@dataclass
class CompoundRecord:
    """Normalized compound record pulled from an external catalogue."""

    name: str
    smiles: str | None = None
    inchi: str | None = None
    inchikey: str | None = None
    molecular_formula: str | None = None
    molecular_weight: float | None = None
    synonyms: list[str] = field(default_factory=list)
    source: str = ""
    source_id: str = ""
    source_url: str = ""
    attributes: dict = field(default_factory=dict)


def _get(url: str, timeout: float = 20.0, accept="application/json", _retries: int = 3) -> str:
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": "drug-discovery-lab/0.1"})
    last = None
    for i in range(_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            # 429/500 = transient throttle → brief backoff, then retry
            if e.code in (429, 500, 502, 503) and i < _retries - 1:
                import time as _t
                _t.sleep(1.5 * (i + 1))
                continue
            last = f"HTTP {e.code} {url}"
        except Exception as e:  # noqa: BLE001
            if i < _retries - 1:
                import time as _t
                _t.sleep(1.5 * (i + 1))
                continue
            last = f"{url}: {e}"
    raise CompoundFetchError(last)


# ---------------------------------------------------------------------------
# PubChem provider
# ---------------------------------------------------------------------------
class PubChemProvider:
    """Name -> CID -> full identity/property fetch via PUG REST."""

    name = "pubchem"

    def _cids_for_name(self, name: str) -> list[str]:
        url = f"{PUBCHEM}/compound/name/{urllib.parse.quote(name)}/cids/JSON?MaxRecords=5"
        try:
            data = json.loads(_get(url))
            ids = data.get("IdentifierList", {}).get("CID", [])
            return [str(i) for i in ids]
        except Exception:
            return []

    def fetch_by_name(self, name: str) -> CompoundRecord | None:
        cids = self._cids_for_name(name)
        if not cids:
            log.info("PubChem: no CID for %r", name)
            return None
        cid = cids[0]
        prop_names = "MolecularFormula,MolecularWeight,CanonicalSMILES,InChI,InChIKey,IUPACName,XLogP,TPSA"
        url = (f"{PUBCHEM}/compound/cid/{cid}/property/{prop_names},Complexity,"
               "HBondDonorCount,HBondAcceptorCount,RotatableBondCount,HeavyAtomCount/JSON")
        try:
            props = json.loads(_get(url))["PropertyTable"]["Properties"][0]
        except Exception as e:
            log.warning("PubChem property fetch failed for %r: %s", name, e)
            props = {}

        def cfg(k):
            return props.get(k)

        smiles = cfg("CanonicalSMILES") or cfg("ConnectivitySMILES")
        if not smiles:
            inchi_raw = cfg("InChI")
            if inchi_raw:
                try:
                    from rdkit import Chem
                    m = Chem.MolFromInchi(inchi_raw)
                    smiles = Chem.MolToSmiles(m, canonical=True, isomericSmiles=True) if m else None
                except Exception:  # noqa: BLE001
                    smiles = None
        return CompoundRecord(
            name=name,
            smiles=smiles,
            inchi=cfg("InChI"),
            inchikey=cfg("InChIKey"),
            molecular_formula=cfg("MolecularFormula"),
            molecular_weight=cfg("MolecularWeight"),
            synonyms=[name, cfg("IUPACName")] if cfg("IUPACName") else [name],
            source="pubchem",
            source_id=cid,
            source_url=f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
            attributes={
                "xlogp": cfg("XLogP"),
                "tpsa": cfg("TPSA"),
                "hbd": cfg("HBondDonorCount"),
                "hba": cfg("HBondAcceptorCount"),
                "rotb": cfg("RotatableBondCount"),
                "heavy_atoms": cfg("HeavyAtomCount"),
                "complexity": cfg("Complexity"),
                "iupac": cfg("IUPACName"),
            },
        )


# ---------------------------------------------------------------------------
# ChEMBL provider
# ---------------------------------------------------------------------------
class ChEMBLProvider:
    """ChEMBL molecule + target-activity lookup (public API, keyless)."""

    name = "chembl"

    def fetch_by_name(self, name: str) -> CompoundRecord | None:
        url = f"{CHEMBL}/molecule/search?q={urllib.parse.quote(name)}&limit=1"
        try:
            data = json.loads(_get(url))
        except Exception as e:
            log.info("ChEMBL search failed for %r: %s", name, e)
            return None
        mols = data.get("molecules") or []
        if not mols:
            return None
        m = mols[0]

        def g(k):
            return m.get(k)

        struct = m.get("molecule_structures") or {}
        attr = {}
        for k in ("molecular_formula", "molecular_weight", "alogp", "psa", "hbd",
                  "hba", "rotatable_bonds", "full_molformula", "pref_name", "molecule_chembl_id"):
            v = g(k)
            if v is not None:
                attr[k] = v
        try:
            records = self._chembl_activities(g("molecule_chembl_id"))
            if records:
                attr["activities"] = records[:10]
        except Exception:
            pass
        return CompoundRecord(
            name=g("pref_name") or name,
            smiles=struct.get("canonical_smiles"),
            inchi=struct.get("standard_inchi"),
            inchikey=struct.get("standard_inchi_key"),
            molecular_formula=g("full_molformula") or g("molecular_formula"),
            molecular_weight=g("molecular_weight"),
            synonyms=[name],
            source="chembl",
            source_id=g("molecule_chembl_id"),
            source_url=f"https://www.ebi.ac.uk/chembl/assay_report_card/{g('molecule_chembl_id')}/",
            attributes=attr,
        )

    def _chembl_activities(self, chembl_id: str, limit: int = 10) -> list[dict]:
        if not chembl_id:
            return []
        url = (f"{CHEMBL}/activity?molecule_chembl_id={chembl_id}"
               f"&target_pref_name__icontains=alopecia&limit={limit}")
        try:
            data = json.loads(_get(url))
            return [
                {
                    "target": a.get("target_pref_name"),
                    "type": a.get("standard_type"),
                    "value": a.get("standard_value"),
                    "units": a.get("standard_units"),
                    "relation": a.get("standard_relation"),
                    "pchembl": a.get("pchembl_value"),
                }
                for a in (data.get("activities") or [])
            ]
        except Exception:
            return []

    def target_search(self, name: str, limit: int = 8) -> list[dict]:
        """Predict-mechanism search: match a target name to a ChEMBL target id."""
        url = f"{CHEMBL}/target/search?q={urllib.parse.quote(name)}&limit={limit}"
        try:
            data = json.loads(_get(url))
            return [
                {
                    "id": t.get("target_chembl_id"),
                    "name": t.get("pref_name"),
                    "organism": t.get("organism"),
                    "target_type": t.get("target_type"),
                }
                for t in (data.get("targets") or [])
            ]
        except Exception:
            return []

    def compounds_for_target(self, target_chembl_id: str, min_pchembl: float = 4.0,
                             limit: int = 25) -> list[dict]:
        """Real measured activity records (molecule + potency) against a target.

        NOTE: ChEMBL's server-side `pchembl_value__gte` filter returns HTTP 500,
        so we fetch a window unfiltered and filter client-side (kept robust).
        """
        url = f"{CHEMBL}/activity?target_chembl_id={target_chembl_id}&limit=50"
        try:
            data = json.loads(_get(url))
        except Exception:
            return []
        out = []
        for a in (data.get("activities") or []):
            rec = {
                "molecule_chembl_id": a.get("molecule_chembl_id"),
                "name": a.get("molecule_pref_name"),
                "smiles": a.get("canonical_smiles"),
                "standard_type": a.get("standard_type"),
                "standard_value": a.get("standard_value"),
                "standard_units": a.get("standard_units"),
                "relation": a.get("standard_relation"),
                "pchembl": float(a["pchembl_value"]) if a.get("pchembl_value") is not None else None,
                "assay_type": a.get("assay_type"),
            }
            if rec["smiles"] and rec["pchembl"] and rec["pchembl"] >= min_pchembl:
                out.append(rec)
        # dedupe by molecule, keep strongest pChEMBL
        dedup: dict[str, dict] = {}
        for rec in out:
            cur = dedup.get(rec["molecule_chembl_id"])
            if cur is None or rec["pchembl"] > cur["pchembl"]:
                dedup[rec["molecule_chembl_id"]] = rec
        ranked = sorted(dedup.values(), key=lambda r: -r["pchembl"])
        return ranked[:limit]


def get_compound_provider(name: str | None = None):
    name = (name or "pubchem").lower()
    if name == "chembl":
        return ChEMBLProvider()
    return PubChemProvider()


def fetch_compound(name: str, providers=("pubchem", "chembl")) -> CompoundRecord | None:
    """Try providers in order, return the first record that carries a usable SMILES."""
    for p in providers:
        rec = get_compound_provider(p).fetch_by_name(name)
        if rec and rec.smiles:
            return rec
    # fall back to any record even without SMILES (reported as `incomplete`)
    for p in providers:
        rec = get_compound_provider(p).fetch_by_name(name)
        if rec:
            return rec
    return None