"""Structural biology providers (spec 36 ProteinStructureProvider seam).

Flow: UniProt accession -> PDB cross-refs (method/resolution) -> RCSB entry
metadata + ligands -> cif artifact -> sha256. AlphaFold covers targets with no
experimental structure; predicted models are labelled predictions, never facts.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request

log = logging.getLogger("dd.struct")

UNIPROT = "https://rest.uniprot.org/uniprotkb"  # /search for queries, /<acc>.json direct
RCSB = "https://data.rcsb.org/rest/v1/core"
AF = "https://alphafold.ebi.ac.uk/api"
FILES = "https://files.rcsb.org/download"
UA = {"User-Agent": "drug-discovery-platform/0.1 (research; keyless)"}


def _get(url, tries=3, backoff=2.0):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={**UA, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            code = getattr(e, "code", None)
            if code and 400 <= code < 500 and code != 429:
                return None
            if attempt < tries - 1:
                time.sleep(backoff * (attempt + 1))
    return None


def _get_bytes(url, tries=3, backoff=2.0):
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            code = getattr(e, "code", None)
            if code and 400 <= code < 500 and code != 429:
                return None
            if attempt < tries - 1:
                time.sleep(backoff * (attempt + 1))
    return None


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def uniprot_for_gene(gene, organism="9606"):
    q = f"gene_exact:{gene} AND organism_id:{organism} AND reviewed:true"
    d = _get(f"{UNIPROT}/search?query={urllib.parse.quote(q)}"
             "&fields=accession,gene_names,protein_name&format=json&size=1")
    if not d or not d.get("results"):
        # fallback: descriptive names (e.g. "Vascular endothelial growth factor-A")
        q2 = f"{gene} AND organism_id:{organism} AND reviewed:true"
        d = _get(f"{UNIPROT}/search?query={urllib.parse.quote(q2)}"
                 "&fields=accession,gene_names,protein_name&format=json&size=1")
    if not d or not d.get("results"):
        return None
    r = d["results"][0]
    genes = [g.get("geneName", {}).get("value") for g in r.get("genes", [])]
    name = (r.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", ""))
    return {"accession": r.get("primaryAccession", ""), "gene": genes[0] if genes else gene, "protein_name": name}


def pdb_cross_refs(accession):
    d = _get(f"{UNIPROT}/{accession}.json")
    if not d:
        return []
    out = []
    for x in d.get("uniProtKBCrossReferences", []):
        if x.get("database") != "PDB":
            continue
        props = {p["key"]: p["value"] for p in x.get("properties", [])}
        res = props.get("Resolution", "")
        try:
            res_v = float(res.split()[0])
        except (ValueError, IndexError):
            res_v = None
        out.append({"pdb_id": x["id"], "method": (props.get("Method", "") or "").lower(), "resolution": res_v})
    return out


def fetch_cif(pdb_id):
    return _get_bytes(f"{FILES}/{pdb_id.upper()}.cif")


def alphafold(accession):
    d = _get(f"{AF}/prediction/{accession}")
    if isinstance(d, dict):
        d = [d]
    if not d:
        return None
    e = d[0]
    return {"source_id": e.get("modelEntityId") or e.get("entryId") or f"AF-{accession}",
            "cif_url": e.get("cifUrl") or e.get("bcifUrl", ""),
            "confidence": e.get("fractionPlddtConfident"),
            "organism": e.get("organismScientificName", ""),
            "method": "predicted"}


def pdb_entry_meta(pdb_id):
    d = _get(f"{RCSB}/entry/{pdb_id.upper()}") or {}
    info = d.get("rcsb_entry_info", {})
    ex = (d.get("exptl") or [{}])[0]
    ligands = []
    for lid in (d.get("rcsb_entry_container_identifiers", {}).get("non_polymer_entity_ids") or [])[:12]:
        nd = _get(f"{RCSB}/nonpolymer_entity/{pdb_id.upper()}/{lid}") or {}
        cid = (nd.get("rcsb_nonpolymer_entity_container_identifiers", {}).get("nonpolymer_comp_id", ""))
        nm = (nd.get("entity_names") or [{}])[0].get("name", "")
        if cid:
            ligands.append({"comp_id": cid, "name": nm or cid})
    return {"title": d.get("struct", {}).get("title", ""),
            "method": (ex.get("experiment_type") or "").lower(),
            "resolution": (info.get("resolution_combined") or [None])[0],
            "organism": (info.get("major_species_of_polypeptide") or [""])[0],
            "ligands": ligands}
