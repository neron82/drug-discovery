"""Entity normalization — alias map for the initial disease domain (androgenetic alopecia),
with a generic fallback so the system generalises to other workspaces.

Maps surface tokens (as extracted by the LLM) to a canonical node name + kind so the
knowledge graph stays unified across papers that spell the same thing differently.
New disease workspaces extend/add their own term sets; the mechanism is agnostic.
"""
from __future__ import annotations

# alias (lowercased) -> (canonical_name, kind)
ALIASES: dict[str, tuple[str, str]] = {
    # disease / phenotype
    "androgenetic alopecia": ("Androgenetic alopecia (AGA)", "disease"),
    "male pattern baldness": ("Androgenetic alopecia (AGA)", "disease"),
    "pattern hair loss": ("Androgenetic alopecia (AGA)", "disease"),
    "mpb": ("Androgenetic alopecia (AGA)", "disease"),
    "aga": ("Androgenetic alopecia (AGA)", "disease"),
    "balding scalp": ("Androgenetic alopecia (AGA)", "disease"),
    "follicular miniaturization": ("Follicular miniaturization", "phenotype"),
    "hair follicle miniaturization": ("Follicular miniaturization", "phenotype"),
    "hair miniaturization": ("Follicular miniaturization", "phenotype"),
    "scalp hair loss": ("Hair loss (alopecia)", "phenotype"),
    # tissue / cells
    "scalp": ("Scalp", "tissue"),
    "occipital scalp": ("Scalp (occipital)", "tissue"),
    "balding scalp": ("Scalp (balding)", "tissue"),
    "hair follicle": ("Hair follicle", "tissue"),
    "hair follicle dermal papilla cell": ("Dermal papilla cell", "cell_type"),
    "dermal papilla cell": ("Dermal papilla cell", "cell_type"),
    "dermal papilla": ("Dermal papilla", "cell_type"),
    "dermal papilla cells": ("Dermal papilla cell", "cell_type"),
    "dpc": ("Dermal papilla cell", "cell_type"),
    "human dermal papilla cell": ("Dermal papilla cell", "cell_type"),
    "keratinocyte": ("Keratinocyte", "cell_type"),
    "hair follicle keratinocyte": ("Keratinocyte", "cell_type"),
    "outer root sheath keratinocyte": ("Outer root sheath keratinocyte", "cell_type"),
    "sebaceous gland": ("Sebaceous gland", "tissue"),
    # hormones / metabolites / ligands
    "dihydrotestosterone": ("DHT (dihydrotestosterone)", "hormone"),
    "dht": ("DHT (dihydrotestosterone)", "hormone"),
    "testosterone": ("Testosterone", "hormone"),
    "5α-dihydrotestosterone": ("DHT (dihydrotestosterone)", "hormone"),
    # androgen axis
    "androgen receptor": ("Androgen receptor (AR)", "receptor"),
    "ar": ("Androgen receptor (AR)", "receptor"),
    "androgen receptor signalling": ("Androgen receptor signalling", "pathway"),
    "androgen signaling": ("Androgen receptor signalling", "pathway"),
    "androgen signalling": ("Androgen receptor signalling", "pathway"),
    "androgen signaling pathway": ("Androgen receptor signalling", "pathway"),
    "androgen response": ("Androgen receptor signalling", "pathway"),
    "5-alpha-reductase": ("5-alpha-reductase (generic)", "enzyme"),
    "5-alpha-reductase type 2": ("5-alpha-reductase type 2 (SRD5A2)", "enzyme"),
    "5α-reductase type 2": ("5-alpha-reductase type 2 (SRD5A2)", "enzyme"),
    "srd5a2": ("5-alpha-reductase type 2 (SRD5A2)", "enzyme"),
    "5-alpha-reductase 2": ("5-alpha-reductase type 2 (SRD5A2)", "enzyme"),
    "5-alpha-reductase type 1": ("5-alpha-reductase type 1 (SRD5A1)", "enzyme"),
    "srd5a1": ("5-alpha-reductase type 1 (SRD5A1)", "enzyme"),
    "5-alpha-reductase 1": ("5-alpha-reductase type 1 (SRD5A1)", "enzyme"),
    # wnt / beta-catenin
    "wnt/beta-catenin signaling": ("Wnt/beta-catenin signalling", "pathway"),
    "wnt/beta-catenin signalling": ("Wnt/beta-catenin signalling", "pathway"),
    "wnt signalling": ("Wnt/beta-catenin signalling", "pathway"),
    "wnt signaling": ("Wnt/beta-catenin signalling", "pathway"),
    "wnt pathway": ("Wnt/beta-catenin signalling", "pathway"),
    "beta-catenin": ("beta-catenin (CTNNB1)", "protein"),
    "β-catenin": ("beta-catenin (CTNNB1)", "protein"),
    "ctnnb1": ("beta-catenin (CTNNB1)", "protein"),
    "wnt3a": ("WNT3A", "protein"),
    "wnt10b": ("WNT10B", "protein"),
    "wnt5a": ("WNT5A", "protein"),
    "dkk1": ("DKK1", "protein"),
    "dickkopf-1": ("DKK1", "protein"),
    # tgf-beta
    "tgf-beta signaling": ("TGF-beta signalling", "pathway"),
    "tgf-β signaling": ("TGF-beta signalling", "pathway"),
    "tgf-β signalling": ("TGF-beta signalling", "pathway"),
    "tgf-beta": ("TGF-beta (TGFB1)", "protein"),
    "tgf-β": ("TGF-beta (TGFB1)", "protein"),
    "tgfb1": ("TGF-beta (TGFB1)", "protein"),
    "tgf-β1": ("TGF-beta (TGFB1)", "protein"),
    "tgf-β2": ("TGF-beta 2 (TGFB2)", "protein"),
    # prostaglandins
    "prostaglandin d2": ("Prostaglandin D2 (PGD2)", "metabolite"),
    "pgd2": ("Prostaglandin D2 (PGD2)", "metabolite"),
    "prostaglandin d2 receptor": ("DP2 receptor (PTGDR2)", "receptor"),
    "dp2 receptor": ("DP2 receptor (PTGDR2)", "receptor"),
    "ptgdr2": ("DP2 receptor (PTGDR2)", "receptor"),
    "gpr44": ("DP2 receptor (PTGDR2)", "receptor"),
    "crdth2": ("DP2 receptor (PTGDR2)", "receptor"),
    "ptgds": ("PGD2 synthase (PTGDS)", "enzyme"),
    "prostaglandin f2α": ("Prostaglandin F2α (PGF2α)", "metabolite"),
    "prostaglandin f2alpha": ("Prostaglandin F2α (PGF2α)", "metabolite"),
    "prostaglandin f2 alpha": ("Prostaglandin F2α (PGF2α)", "metabolite"),
    "ptgfr": ("FP receptor (PTGFR)", "receptor"),
    "fp receptor": ("FP receptor (PTGFR)", "receptor"),
    "prostaglandin d2 synthase": ("PGD2 synthase (PTGDS)", "enzyme"),
    # growth factors / vascular
    "vegf": ("VEGF", "protein"),
    "vegf-a": ("VEGF", "protein"),
    "vascular endothelial growth factor": ("VEGF", "protein"),
    "fgf": ("FGF (fibroblast growth factor)", "protein"),
    "fgf7": ("FGF7 (KGF)", "protein"),
    "kgf": ("FGF7 (KGF)", "protein"),
    "fgf2": ("FGF2 (bFGF)", "protein"),
    "igf-1": ("IGF-1", "protein"),
    "igf1": ("IGF-1", "protein"),
    "gremlin": ("GREM1 (gremlin-1)", "protein"),
    "gremlin-1": ("GREM1 (gremlin-1)", "protein"),
    "grem1": ("GREM1 (gremlin-1)", "protein"),
    # hedgehog / other pathways
    "hedgehog signaling": ("Hedgehog signalling", "pathway"),
    "shh": ("Sonic hedgehog (SHH)", "protein"),
    "sonic hedgehog": ("Sonic hedgehog (SHH)", "protein"),
    "notch signaling": ("Notch signalling", "pathway"),
    "jak-stat": ("JAK-STAT signalling", "pathway"),
    "jak/stat signaling": ("JAK-STAT signalling", "pathway"),
    "pi3k/akt": ("PI3K/AKT signalling", "pathway"),
    "pten": ("PTEN", "protein"),
    # inflammation / oxidative stress / matrix
    "perifollicular inflammation": ("Perifollicular inflammation", "biological_process"),
    "microinflammation": ("Perifollicular inflammation", "biological_process"),
    "inflammatory infiltration": ("Perifollicular inflammation", "biological_process"),
    "oxidative stress": ("Oxidative stress", "biological_process"),
    "reactive oxygen species": ("Reactive oxygen species (ROS)", "metabolite"),
    "ros": ("Reactive oxygen species (ROS)", "metabolite"),
    "extracellular matrix": ("Extracellular matrix (ECM)", "biological_process"),
    "ecm remodelling": ("Extracellular matrix (ECM) remodelling", "biological_process"),
    "collagen": ("Collagen", "protein"),
    "fibrosis": ("Fibrosis", "biological_process"),
    "toll-like receptor": ("Toll-like receptor (TLR)", "receptor"),
    "nf-κb": ("NF-kB", "transcription_factor"),
    "nf-kb": ("NF-kB", "transcription_factor"),
    "nf-κb signalling": ("NF-kB signalling", "pathway"),
    "tumor necrosis factor-alpha": ("TNF-alpha", "protein"),
    "tnf-alpha": ("TNF-alpha", "protein"),
    "interleukin-6": ("IL-6", "protein"),
    "il-6": ("IL-6", "protein"),
    "interleukin-1β": ("IL-1beta", "protein"),
    "interferon-gamma": ("IFN-gamma", "protein"),
    # hair cycle processes
    "anagen": ("Anagen phase", "biological_process"),
    "anagen phase": ("Anagen phase", "biological_process"),
    "catagen": ("Catagen phase", "biological_process"),
    "telogen": ("Telogen phase", "biological_process"),
    "hair cycling": ("Hair cycle", "biological_process"),
    "hair growth": ("Hair growth", "biological_process"),
    "angiogenesis": ("Angiogenesis", "biological_process"),
    "cell proliferation": ("Cell proliferation", "biological_process"),
    "apoptosis": ("Apoptosis", "biological_process"),
    # treatments / drugs
    "minoxidil": ("Minoxidil", "drug"),
    "finasteride": ("Finasteride", "drug"),
    "dutasteride": ("Dutasteride", "drug"),
    "ketoconazole": ("Ketoconazole", "drug"),
    "spironolactone": ("Spironolactone", "drug"),
    "bimatoprost": ("Bimatoprost", "drug"),
    "latanoprost": ("Latanoprost", "drug"),
    "5α-reductase inhibitors": ("5-alpha-reductase inhibitors", "drug_class"),
    "5-alpha-reductase inhibitors": ("5-alpha-reductase inhibitors", "drug_class"),
}

# fallback kind guesses from the extractor's hint type
_HINT_KIND = {
    "disease": "disease", "phenotype": "phenotype", "cell_type": "cell_type",
    "tissue": "tissue", "biological_process": "biological_process", "pathway": "pathway",
    "gene": "gene", "protein": "protein", "receptor": "receptor", "enzyme": "enzyme",
    "transcription_factor": "transcription_factor", "hormone": "hormone",
    "metabolite": "metabolite", "compound": "compound", "drug": "drug",
    "drug_class": "drug_class", "ligand": "ligand",
}


def normalize_entity(raw: str, hint_type: str = "protein") -> dict:
    """Return {canonical_name, kind, mapped}."""
    if not raw:
        return {"canonical_name": "unknown", "kind": "other", "mapped": False}
    key = raw.strip().lower().replace("/", "/")
    if key in ALIASES:
        name, kind = ALIASES[key]
        return {"canonical_name": name, "kind": kind, "mapped": True}
    # try collapsing punctuation/spacing variants (e.g. "TGF beta" -> "TGF-beta")
    compact = _compact(key)
    by_canonical = {}
    for k, (name, kind) in ALIASES.items():
        by_canonical.setdefault(_compact(k), (name, kind))
    if compact in by_canonical:
        name, kind = by_canonical[compact]
        return {"canonical_name": name, "kind": kind, "mapped": True}
    kind = _HINT_KIND.get(hint_type, "protein") if hint_type else "protein"
    # gene/protein tokens often uppercase entity IDs; keep readable title-style
    name = raw.strip()
    return {"canonical_name": name, "kind": kind, "mapped": False}


def _compact(s: str) -> str:
    import re
    return re.sub(r"[\s\-_/αβγ]+", "", s).replace("β", "b").replace("α", "a").lower()
