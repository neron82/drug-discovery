"""Structure-based safety screening (spec 12): PAINS, reactive groups,
promoter-ish aggregation flags. Deterministic RDKit substructure screens.
All outputs are PREDICTIONS labelled as such — never measured toxicity."""
from __future__ import annotations

from . import parse_smiles

_PAINS_CAT = None


def _load_pains():
    global _PAINS_CAT
    if _PAINS_CAT is not None:
        return _PAINS_CAT
    try:
        from rdkit.Chem import FilterCatalog
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        _PAINS_CAT = FilterCatalog.FilterCatalog(params)
    except Exception:
        _PAINS_CAT = False
    return _PAINS_CAT


REACTIVE_SMARTS = {
    "acyl_halide": "[C](=[O])[F,Cl,Br,I]",
    "sulfonyl_halide": "S(=[O])(=[O])[F,Cl,Br,I]",
    "epoxide": "C1OC1",
    "aziridine": "C1NC1",
    "aldehyde": "[CX3H1](=O)",
    "michael_acceptor": "[CX3]=[CX3][C=X]",
    "quinone": "O=C1C=CC(=O)C=C1",
    "aryl_nitro": "[c][#7+](=O)[O-]",  # RDKit canonicalises nitro as N+/O-
    "nitroso": "[NX2]=O",          # excludes nitro (N valence 4)
    "diazo": "[N]=[N]",
    "mustard": "CClCCCl",
    "peroxide": "[OX2H2,OX2][OX2]",
    "acid_anhydride": "C(=O)OC(=O)",
    "isocyanate": "N=C=O",
    "thiol": "[SH]",
    "hydroxylamine": "[OX2H][NX3;!$(N(=O)=O)]",  # N-OH, not nitro O
    "hydrazine": "NN",
    "sulfinic_acid": "S(=O)[OH]",
    "beta_lactam": "O=C1NCC1",
}


def safety_alerts(smiles: str) -> dict:
    """Deterministic structure-only alerts: PAINS + reactive groups + rough
    aggregation flag. Returns dict with prediction=True; never invents
    experimental values."""
    from rdkit import Chem
    mol = parse_smiles(smiles)
    if mol is None:
        return {"prediction": True, "error": "unparseable SMILES"}
    out = {"prediction": True,
           "note": "Substructure-screen predictions — not measured toxicology."}
    pains = []
    cat = _load_pains()
    if cat:
        try:
            hits = cat.GetMatches(mol)
            pains = sorted({h.GetDescription() for h in hits})
        except Exception:
            pains = []
    out["pains"] = pains
    flagged = []
    for name, sma in REACTIVE_SMARTS.items():
        patt = Chem.MolFromSmarts(sma)
        if patt is not None and mol.HasSubstructMatch(patt):
            flagged.append(name)
    out["reactive_groups"] = flagged
    # rough aggregation/promiscuity flag: big, flat, lipophilic, many aromatic rings
    from rdkit.Chem import rdMolDescriptors, Descriptors
    n_arom = rdMolDescriptors.CalcNumAromaticRings(mol)
    logp = Descriptors.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    risk = "low"
    if n_arom >= 5 and logp > 4.5:
        risk = "elevated"
    elif n_arom >= 4 and logp > 4:
        risk = "moderate"
    if pains:
        risk = "elevated" if risk != "elevated" else risk
    out["aggregation_risk"] = risk
    out["liability_score"] = round(min(1.0, 0.4*len(pains) + 0.3*len(flagged)
                                       + (0.3 if risk == "elevated" else 0.1 if risk == "moderate" else 0)), 3)
    return out
