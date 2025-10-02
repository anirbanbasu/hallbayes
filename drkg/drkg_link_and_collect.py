#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DRKG entity linking + PyTorch Geometric accelerated multi-hop evidence collector.
Now uses DrugBank vocabulary file for drug name resolution instead of Wikidata.
"""

import os
import re
import json
import time
import math
import gzip
import argparse
import random
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Iterable

import requests
import pandas as pd
from tqdm import tqdm
from rapidfuzz import fuzz
from lxml import etree

# PyTorch / PyTorch Geometric
import torch
from torch import Tensor
from torch_geometric.utils import to_undirected, k_hop_subgraph, degree, coalesce

# --------------------------------- Globals ---------------------------------

CACHE_DIR = "./cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Path to DrugBank vocabulary CSV
DRUGBANK_VOCAB_PATH = "./drugbank-vocabulary.csv"  # Adjust this path as needed

NCBI_GENE_INFO_URL = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz"
DOID_OBO_URL = "https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/main/src/ontology/doid.obo"

MESH_XML_BASE = "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh"
MESH_PROBE_YEARS = [time.localtime().tm_year, time.localtime().tm_year - 1, time.localtime().tm_year - 2, 2024, 2023]

USER_AGENT = {"User-Agent": "drkg-linker/pyg-1.4 (+https://example.org)"}

# Relation hints and typed transition bonuses
RELATION_HINTS_DEFAULT = [
    "cause","causes","caused","causing","induce","induces","induced","inducing",
    "associated","association","associate","associates","risk","increase","decrease",
    "treat","treated","treats","therapeutic","side effect","adverse","toxicity","safety",
    "interact","upregulat","downregulat","bind","pathway","regulat",
]
TYPE_BONUS_DEFAULT = {
    ("Compound","Gene"):2.0, ("Gene","Pathway"):2.0, ("Pathway","Disease"):2.0,
    ("Gene","Disease"):1.5, ("Compound","Pathway"):1.0, ("Compound","Disease"):1.0,
    # reverse directions
    ("Gene","Compound"):2.0, ("Pathway","Gene"):2.0, ("Disease","Pathway"):2.0,
    ("Disease","Gene"):1.5, ("Pathway","Compound"):1.0, ("Disease","Compound"):1.0,
}

# --------------------------------- Utils ------------------------------------

def _save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[\u2010-\u2015]", "-", s)
    s = re.sub(r"[^a-z0-9\-\s\+\.,/:()]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def _unique(seq):
    seen=set(); out=[]
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

# --------------------------------- DRKG IO ----------------------------------

def load_drkg_edges(drkg_tsv: str) -> pd.DataFrame:
    df = pd.read_csv(drkg_tsv, sep="\t", header=None, names=["h","r","t"], dtype=str)
    for col in ["h","r","t"]:
        df[col] = df[col].fillna("").astype(str)
    return df

def split_type_id(e: str) -> Tuple[str, str]:
    if "::" not in e: return ("Unknown", e)
    typ, rest = e.split("::", 1)
    return typ, rest

def collect_entities(df: pd.DataFrame) -> List[str]:
    return sorted(set(df["h"].tolist()) | set(df["t"].tolist()))

# --------------------------- DrugBank Vocabulary ---------------------------

def load_drugbank_vocabulary(path: str = None) -> Dict[str, Dict]:
    """Load DrugBank vocabulary and create DBID -> info mapping"""
    if path is None:
        path = DRUGBANK_VOCAB_PATH
    
    if not os.path.exists(path):
        print(f"Warning: DrugBank vocabulary file not found at {path}")
        return {}
    
    df = pd.read_csv(path)
    dbid_info = {}
    
    for _, row in df.iterrows():
        dbid = row['DrugBank ID']
        info = {
            'label': row['Common name'] if pd.notna(row['Common name']) else dbid,
            'synonyms': []
        }
        if pd.notna(row['Synonyms']):
            info['synonyms'] = [s.strip() for s in row['Synonyms'].split(' | ')]
        dbid_info[dbid] = info
    
    return dbid_info

def build_drugbank_name_lookup(dbid_info: Dict[str, Dict]) -> Dict[str, str]:
    """Build reverse lookup: drug name -> DrugBank ID"""
    name_to_dbid = {}
    
    for dbid, info in dbid_info.items():
        # Add common name
        if info['label'] and info['label'] != dbid:
            name_to_dbid[info['label'].lower()] = dbid
        
        # Add all synonyms
        for syn in info['synonyms']:
            if syn:
                name_to_dbid[syn.lower()] = dbid
    
    return name_to_dbid

# ----------------------------- Enrichment: Genes ----------------------------

def ensure_ncbi_gene_info() -> str:
    gz = os.path.join(CACHE_DIR, "Homo_sapiens.gene_info.gz")
    if not os.path.exists(gz):
        r = requests.get(NCBI_GENE_INFO_URL, headers=USER_AGENT, timeout=180)
        r.raise_for_status()
        with open(gz, "wb") as f: f.write(r.content)
    return gz

def load_gene_info() -> pd.DataFrame:
    gz = ensure_ncbi_gene_info()
    with gzip.open(gz, "rt", encoding="utf-8", errors="ignore") as f:
        cols = ["#tax_id","GeneID","Symbol","LocusTag","Synonyms","dbXrefs","chromosome","map_location","description",
                "type_of_gene","Symbol_from_nomenclature_authority","Full_name_from_nomenclature_authority",
                "Nomenclature_status","Other_designations","Modification_date","Feature_type"]
        df = pd.read_csv(f, sep="\t", names=cols, comment="#", dtype=str)
    df = df[df["#tax_id"]=="9606"].copy()
    for c in ["GeneID","Symbol","Synonyms","Full_name_from_nomenclature_authority","Other_designations"]:
        df[c] = df[c].fillna("")
    return df

# ------------------------------ Enrichment: DOID ----------------------------

def ensure_doid_obo() -> str:
    path = os.path.join(CACHE_DIR, "doid.obo")
    if not os.path.exists(path):
        r = requests.get(DOID_OBO_URL, headers=USER_AGENT, timeout=180)
        r.raise_for_status()
        with open(path, "wb") as f: f.write(r.content)
    return path

def parse_doid_obo() -> Dict[str, Dict]:
    out = {}
    cur = None
    path = ensure_doid_obo()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line=line.strip()
            if line == "[Term]":
                cur = {}
                continue
            if not line:
                if cur and "id" in cur: out[cur["id"]] = cur
                cur = None; continue
            if cur is None: continue
            if line.startswith("id: "):
                cur["id"] = line[4:]; cur.setdefault("synonyms", []); cur.setdefault("xrefs", {})
            elif line.startswith("name: "):
                cur["label"] = line[6:]
            elif line.startswith("synonym: "):
                m = re.search(r'^synonym:\s+"(.+?)"\s+(\w+)', line)
                if m: cur["synonyms"].append(m.group(1))
            elif line.startswith("xref: "):
                m = re.search(r'^xref:\s+(\w+):(.+)$', line)
                if m:
                    ns, val = m.group(1), m.group(2)
                    cur["xrefs"][ns] = cur["xrefs"].get(ns, []) + [val]
    return out

# ------------------------------ Enrichment: MeSH ----------------------------

def ensure_mesh_desc_xml() -> str:
    cache_path = os.path.join(CACHE_DIR, "mesh_desc.xml")
    if os.path.exists(cache_path): return cache_path
    last_exc = None
    for y in _unique([y for y in MESH_PROBE_YEARS if isinstance(y, int) and 1990 < y < 2100]):
        url = f"{MESH_XML_BASE}/desc{y}.xml"
        try:
            r = requests.get(url, headers=USER_AGENT, timeout=240)
            if r.status_code == 200 and r.content:
                with open(cache_path, "wb") as f: f.write(r.content)
                return cache_path
        except Exception as e:
            last_exc = e
            continue
    raise RuntimeError("Could not download MeSH descriptor XML (descYYYY.xml). "
                       "Place a recent file as ./cache/mesh_desc.xml manually.") from last_exc

def parse_mesh_desc_xml(path: str) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    context = etree.iterparse(path, events=("end",), tag="DescriptorRecord")
    for _, rec in context:
        try:
            ui = rec.findtext("DescriptorUI")
            if not ui or not ui.startswith("D"): rec.clear(); continue
            label = rec.findtext("DescriptorName/String")
            syns_set = set()
            for concept in rec.findall("ConceptList/Concept"):
                for term in concept.findall("TermList/Term/String"):
                    if term is not None and term.text:
                        syns_set.add(term.text.strip())
            if label: syns_set.add(label)
            out[ui] = {"label": label or ui, "synonyms": _unique([s for s in syns_set if s])}
        except Exception:
            pass
        finally:
            rec.clear()
    return out

# ------------------------------ Alias index --------------------------

def build_alias_index(df_edges: pd.DataFrame, refresh: bool = False, drugbank_path: str = None) -> pd.DataFrame:
    """
    Build alias index using DrugBank vocabulary instead of Wikidata
    """
    cache_path = os.path.join(CACHE_DIR, "alias_index.parquet")
    csv_fallback = os.path.join(CACHE_DIR, "alias_index.csv")

    if os.path.exists(cache_path) and not refresh:
        try: return pd.read_parquet(cache_path)
        except Exception: pass
    if os.path.exists(csv_fallback) and not refresh:
        try: return pd.read_csv(csv_fallback)
        except Exception: pass

    print("Loading DrugBank vocabulary...")
    dbid_info = load_drugbank_vocabulary(drugbank_path)
    
    entities = collect_entities(df_edges)
    buckets = defaultdict(list)
    for e in entities:
        t, rest = split_type_id(e)
        buckets[t].append((e, rest))

    rows = []

    # Genes
    print("Loading gene information...")
    gene_info = load_gene_info()
    gene_dict = {str(gid): row for gid, row in gene_info.set_index("GeneID").iterrows()}

    # DOID
    print("Loading disease ontology...")
    doid = parse_doid_obo()
    
    # MeSH
    print("Loading MeSH descriptors...")
    mesh_desc_xml_path = ensure_mesh_desc_xml()
    mesh_dict = parse_mesh_desc_xml(mesh_desc_xml_path)

    print("Building alias index...")
    for e in entities:
        typ, rest = split_type_id(e)
        aliases = []
        canon = None
        external = {}
        
        if typ == "Compound":
            if rest.startswith("DB"):
                # Use DrugBank vocabulary
                if rest in dbid_info:
                    info = dbid_info[rest]
                    canon = info['label']
                    aliases = info['synonyms']
                else:
                    canon = rest  # Fallback to ID if not found
            elif rest.startswith("pubchem:"):
                cid = rest.split(":",1)[1]
                canon = f"PubChem CID {cid}"
                external["pubchem"] = [cid]
        
        elif typ == "Gene":
            row = gene_dict.get(rest)
            if row is not None:
                canon = row["Symbol"] or rest
                syns = []
                syns += [row["Symbol"]]
                syns += row["Synonyms"].split("|") if row["Synonyms"] else []
                syns += [row["Full_name_from_nomenclature_authority"]] if row["Full_name_from_nomenclature_authority"] else []
                syns += row["Other_designations"].split("|") if row["Other_designations"] else []
                aliases = _unique([s for s in syns if s and s != "-"])
        
        elif typ == "Disease":
            if rest.startswith("DOID:"):
                info = doid.get(rest)
                if info:
                    canon = info.get("label") or rest
                    aliases = info.get("synonyms", [])
                else:
                    canon = rest
            elif rest.startswith("MESH:"):
                ui = rest.split(":", 1)[1]
                mi = mesh_dict.get(ui)
                if mi:
                    canon = mi.get("label") or rest
                    aliases = mi.get("synonyms", [])
                    external["mesh_ui"] = ui
                else:
                    canon = rest
            else:
                canon = rest
        else:
            canon = rest

        aliases_norm = _unique([_norm(a) for a in [canon, *aliases] if a])

        rows.append({
            "drkg_id": e,
            "type": typ,
            "source_id": rest,
            "canonical_label": canon or rest,
            "aliases": aliases,
            "aliases_norm": aliases_norm,
            "external": json.dumps(external, ensure_ascii=False),
        })

    alias_df = pd.DataFrame(rows)
    try:
        alias_df.to_parquet(cache_path, index=False)
    except Exception:
        alias_df.to_csv(csv_fallback, index=False)
    
    return alias_df

# --------------------------------- Linking ----------------------------------

def extract_intent(prompt: str) -> Dict:
    p = prompt.lower()
    r = {"expects": None, "relation_bias": []}
    if any(w in p for w in ["cause","causes","induce","induces","risk","associated"]):
        r["expects"] = ("Compound","Disease")
        r["relation_bias"] = ["cause","risk","associated"]
    if any(w in p for w in ["treat","therapy","therapeutic","indication"]):
        r["expects"] = r["expects"] or ("Compound","Disease")
        r["relation_bias"].append("treat")
    if any(w in p for w in ["gene","polymorphism","mutation"]):
        if r["expects"] is None: r["expects"] = ("Gene", None)
    return r

def candidate_strings(prompt: str) -> List[str]:
    out = []
    out += re.findall(r'[""](.+?)[""]', prompt)
    out += re.findall(r"\b([A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+)*)\b", prompt)
    out += re.findall(r"[A-Za-z0-9\-\+]{3,}", prompt)
    out = [s.strip() for s in out if s.strip()]
    return _unique(out)

def link_mentions(prompt: str, alias_df: pd.DataFrame, topk: int = 5) -> List[Dict]:
    intent = extract_intent(prompt)
    cands = []
    inv = defaultdict(list)
    for i, row in alias_df.iterrows():
        for a in row["aliases_norm"]:
            inv[a].append(i)
    for span in candidate_strings(prompt):
        n = _norm(span)
        hit_rows = inv.get(n, [])
        if not hit_rows:
            for key in inv.keys():
                if len(key) < 3: continue
                score = fuzz.token_set_ratio(n, key)
                if score >= 90:
                    hit_rows += inv[key]
        hit_rows = _unique(hit_rows)
        for idx in hit_rows[:topk]:
            row = alias_df.iloc[idx]
            type_bonus = 0.0
            if intent["expects"]:
                expected = [t for t in intent["expects"] if t]
                if row["type"] in expected: type_bonus = 10.0
            base = 100.0 if n in row["aliases_norm"] else 85.0
            score = base + type_bonus
            cands.append({
                "mention": span,
                "norm": n,
                "drkg_id": row["drkg_id"],
                "type": row["type"],
                "canonical_label": row["canonical_label"],
                "score": score
            })
    cands.sort(key=lambda x: (-x["score"], x["drkg_id"]))
    uniq = {}
    for c in cands:
        key = (c["mention"].lower(), c["type"])
        uniq.setdefault(key, c)
    return list(uniq.values())[:50]

# ------------------------- PyG graph build & subgraph -----------------------

def build_pyg_graph(df: pd.DataFrame, allowed_types: Iterable[str], undirected: bool = True):
    """Return (edge_index[E2], node2idx, idx2node, node_types[idx], rel_list[E2])"""
    allowed = set(t.strip() for t in allowed_types if t.strip())
    def _ok(e: str) -> bool: return split_type_id(e)[0] in allowed
    sub = df[df["h"].apply(_ok) & df["t"].apply(_ok)].copy()

    nodes = _unique(list(sub["h"]) + list(sub["t"]))
    node2idx = {n:i for i,n in enumerate(nodes)}
    idx2node = {i:n for n,i in node2idx.items()}
    node_types = [split_type_id(idx2node[i])[0] for i in range(len(nodes))]

    hs = [node2idx[x] for x in sub["h"].tolist()]
    ts = [node2idx[x] for x in sub["t"].tolist()]
    rels = sub["r"].tolist()

    # Build edge_index
    ei = torch.tensor([hs, ts], dtype=torch.long)
    rel_list = list(rels)
    if undirected:
        hs_rev = ts[:]
        ts_rev = hs[:]
        ei_rev = torch.tensor([hs_rev, ts_rev], dtype=torch.long)
        ei = torch.cat([ei, ei_rev], dim=1)
        rel_list = rel_list + rel_list

    # Coalesce duplicates
    ei, _ = coalesce(ei, None, len(nodes), len(nodes))
    seen_rel = {}
    for h, t, r in zip(hs + hs_rev if undirected else hs, ts + ts_rev if undirected else ts, rels + rels if undirected else rels):
        seen_rel[(h,t)] = seen_rel.get((h,t), r)
    rel_list_aligned = [seen_rel[(int(ei[0,i]), int(ei[1,i]))] for i in range(ei.size(1))]

    return ei, node2idx, idx2node, node_types, rel_list_aligned

def khop_neighborhood(edge_index: Tensor, sources_idx: List[int], max_hops: int, num_nodes: int):
    """Return (subset_nodes, sub_ei, mapping, edge_mask) as in PyG."""
    src = torch.tensor(sources_idx, dtype=torch.long)
    subset, sub_ei, mapping, edge_mask = k_hop_subgraph(src, max_hops, edge_index, relabel_nodes=True, num_nodes=num_nodes)
    return subset, sub_ei, mapping, edge_mask

# ------------------------- Edge scoring & path search -----------------------

def relation_bias_vector(rel_list: List[str], hints: List[str]) -> Tensor:
    low_hints = [h.lower() for h in hints]
    scores = []
    for r in rel_list:
        rl = r.lower()
        s = sum(1 for k in low_hints if k in rl)
        scores.append(float(s))
    return torch.tensor(scores, dtype=torch.float32)

def type_transition_bonus_vector(edge_index: Tensor, node_types: List[str], bonus_map: Dict[Tuple[str,str], float]) -> Tensor:
    u = edge_index[0].tolist(); v = edge_index[1].tolist()
    vals = []
    for ui,vi in zip(u,v):
        tu = node_types[ui]; tv = node_types[vi]
        vals.append(float(bonus_map.get((tu,tv), 0.0)))
    return torch.tensor(vals, dtype=torch.float32)

def degree_penalty_vector(edge_index: Tensor, num_nodes: int, alpha: float = 0.02) -> Tensor:
    deg = degree(edge_index[0], num_nodes=num_nodes)
    deg = deg.clamp_min_(0)
    u = edge_index[0]; v = edge_index[1]
    pu = torch.log1p(deg[u]); pv = torch.log1p(deg[v])
    return alpha * (pu * pv)

def build_edge_scores(edge_index: Tensor,
                      rel_list: List[str],
                      node_types: List[str],
                      relation_hints: List[str],
                      type_bonus_map: Dict[Tuple[str,str], float],
                      degree_alpha: float = 0.02) -> Tensor:
    rb = relation_bias_vector(rel_list, relation_hints)
    tb = type_transition_bonus_vector(edge_index, node_types, type_bonus_map)
    dp = degree_penalty_vector(edge_index, num_nodes=len(node_types), alpha=degree_alpha)
    return 3.0*rb + tb - dp

class Path:
    __slots__ = ("nodes", "edges", "score")
    def __init__(self, nodes: List[int], edges: List[int], score: float):
        self.nodes = nodes
        self.edges = edges
        self.score = float(score)

def build_subgraph_adjacency(sub_ei: Tensor) -> Dict[int, List[Tuple[int,int]]]:
    """Return adjacency: node -> list of (neighbor_node, edge_id)."""
    adj = defaultdict(list)
    E = sub_ei.size(1)
    for eid in range(E):
        u = int(sub_ei[0, eid]); v = int(sub_ei[1, eid])
        adj[u].append((v, eid))
    return adj

def beam_search_paths_subgraph(sub_ei: Tensor,
                               edge_scores: Tensor,
                               subset_nodes: Tensor,
                               node_types_sub: List[str],
                               sources_local: List[int],
                               targets_local: List[int],
                               max_hops: int = 4,
                               beam_width: int = 500,
                               max_paths: int = 25) -> List[Path]:
    """Beam-search over the PyG subgraph (local node ids)."""
    adj = build_subgraph_adjacency(sub_ei)
    results: List[Path] = []
    frontier: List[Path] = [Path([s], [], 0.0) for s in sources_local]

    for depth in range(1, max_hops+1):
        new_frontier: List[Path] = []
        for p in frontier:
            u = p.nodes[-1]
            prev = p.nodes[-2] if len(p.nodes) >= 2 else None
            for (v, eid) in adj.get(u, []):
                if prev is not None and v == prev:
                    continue  # no immediate backtrack
                sc = p.score + float(edge_scores[eid])
                new_p = Path(p.nodes + [v], p.edges + [eid], sc)
                if v in targets_local:
                    results.append(new_p)
                else:
                    new_frontier.append(new_p)
        # Beam prune
        new_frontier.sort(key=lambda x: x.score, reverse=True)
        frontier = new_frontier[:beam_width]
        if len(results) >= max_paths and depth >= 2:
            break

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:max_paths]

# ----------------------- Evidence construction & permutations ----------------

def rank_edges_by_path_support(paths: List[Path]) -> List[Tuple[int, float, int]]:
    count = Counter(); agg = defaultdict(float)
    for p in paths:
        used = set()
        for eid in p.edges:
            if eid in used: continue
            used.add(eid)
            count[eid] += 1
            agg[eid] += p.score
    scored = [(eid, agg[eid] + 10.0*count[eid], count[eid]) for eid in agg.keys()]
    scored.sort(key=lambda x: (-x[1], -x[2], x[0]))
    return scored

def verbalize_edge_local(eid: int,
                         sub_ei: Tensor,
                         subset_nodes: Tensor,
                         idx2node_global: Dict[int,str],
                         alias_df: pd.DataFrame,
                         rel_list: List[str]) -> str:
    u = int(sub_ei[0, eid]); v = int(sub_ei[1, eid])
    g_u = int(subset_nodes[u]); g_v = int(subset_nodes[v])
    drkg_u = idx2node_global[g_u]; drkg_v = idx2node_global[g_v]
    relation = rel_list[eid] if eid < len(rel_list) else "relates_to"
    
    def label(drkg_id: str) -> str:
        rows = alias_df[alias_df["drkg_id"]==drkg_id]
        return rows.iloc[0]["canonical_label"] if not rows.empty else drkg_id
    return f"- Fact: {label(drkg_u)} — {relation} — {label(drkg_v)}. [{drkg_u} ; {drkg_v}]"

def permute_evidence(lines: List[str], m: int, seeds: Optional[List[int]] = None) -> List[List[str]]:
    if not lines: lines = ["- (no relevant DRKG facts found)"]
    seeds = seeds if seeds is not None else list(range(m))
    perms = []
    for s in seeds[:m]:
        rng = random.Random(int(s))
        order = list(lines); rng.shuffle(order)
        perms.append(order)
    return perms

# ------------------------------------ CLI -----------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drkg", required=True, help="Path to drkg.tsv")
    ap.add_argument("--prompt", required=True, help="User prompt")
    ap.add_argument("--drugbank-vocab", default=DRUGBANK_VOCAB_PATH, help="Path to DrugBank vocabulary CSV")
    ap.add_argument("--allowed-types", default="Compound,Gene,Pathway,Disease",
                    help="Comma-separated node types to keep")
    ap.add_argument("--max-hops", type=int, default=4)
    ap.add_argument("--beam-width", type=int, default=500)
    ap.add_argument("--max-paths", type=int, default=15)
    ap.add_argument("--max-evidence", type=int, default=20)
    ap.add_argument("--m", type=int, default=6)
    ap.add_argument("--refresh-index", action="store_true", help="Rebuild alias cache")
    ap.add_argument("--undirected", action="store_true", default=True, help="Undirected traversal")
    ap.add_argument("--include-1hop", action="store_true", help="Also include top 1-hop edges")
    args = ap.parse_args()

    # 1) Load DRKG + alias index
    print("Loading DRKG…")
    df = load_drkg_edges(args.drkg)
    print("Building/Loading alias index…")
    alias_df = build_alias_index(df, refresh=args.refresh_index, drugbank_path=args.drugbank_vocab)

    # 2) Link mentions
    links = link_mentions(args.prompt, alias_df)
    by_type = defaultdict(list)
    for c in links: by_type[c["type"]].append(c)

    chosen = []
    for t in ["Compound","Disease","Gene","Pathway"]:
        if by_type[t]: chosen.append(max(by_type[t], key=lambda x: x["score"]))
    chosen_ids = [x["drkg_id"] for x in chosen]

    print("\n=== Linking ===")
    if chosen:
        for c in chosen:
            print(f"{c['mention']}  ->  {c['drkg_id']}  [{c['type']}]  label={c['canonical_label']}  score={c['score']:.1f}")
    else:
        print("(no confident links)")

    intent = extract_intent(args.prompt)
    relation_bias = intent["relation_bias"] if intent["relation_bias"] else RELATION_HINTS_DEFAULT

    # Identify sources/targets
    comp_ids = [c["drkg_id"] for c in chosen if c["type"]=="Compound"]
    dis_ids  = [c["drkg_id"] for c in chosen if c["type"]=="Disease"]
    gene_ids = [c["drkg_id"] for c in chosen if c["type"]=="Gene"]

    if comp_ids and dis_ids:
        sources_ids = comp_ids
        targets_ids = dis_ids
    elif comp_ids and gene_ids:
        sources_ids = comp_ids
        targets_ids = gene_ids
    elif gene_ids and dis_ids:
        sources_ids = gene_ids
        targets_ids = dis_ids
    elif comp_ids:
        sources_ids = comp_ids
        targets_ids = dis_ids or []
    else:
        if len(chosen_ids) >= 2:
            sources_ids = [chosen_ids[0]]; targets_ids = [chosen_ids[1]]
        elif len(chosen_ids) == 1:
            sources_ids = [chosen_ids[0]]; targets_ids = []
        else:
            sources_ids = []; targets_ids = []

    # 3) Build PyG graph
    allowed_types = [t.strip() for t in args.allowed_types.split(",") if t.strip()]
    ei, node2idx, idx2node, node_types, rel_list = build_pyg_graph(df, allowed_types, undirected=args.undirected)
    N = len(node_types)

    if not sources_ids:
        print("\nNo sources linked; cannot traverse.")
        return

    # Map global node ids
    sources_local_global_idx = [node2idx[s] for s in sources_ids if s in node2idx]
    if not sources_local_global_idx:
        print("\nLinked sources not present in the typed graph.")
        return

    subset, sub_ei, mapping, edge_mask = khop_neighborhood(ei, sources_local_global_idx, args.max_hops, N)
    print(f"\nSubgraph size: nodes={subset.numel()} edges={sub_ei.size(1)} (K={args.max_hops})")

    # Build subgraph node types
    node_types_sub = [node_types[int(g)] for g in subset.tolist()]
    # Build aligned relation strings
    rel_full = [rel_list[i] for i,keep in enumerate(edge_mask.tolist()) if keep]
    # Score edges
    edge_scores = build_edge_scores(sub_ei, rel_full, node_types_sub, relation_bias, TYPE_BONUS_DEFAULT, degree_alpha=0.02)

    # Compute local ids for sources/targets
    g2local = {int(g):i for i,g in enumerate(subset.tolist())}
    sources_local = [g2local[g] for g in sources_local_global_idx if g in g2local]
    targets_local = []
    if targets_ids:
        targets_global = [node2idx[t] for t in targets_ids if t in node2idx]
        targets_local = [g2local[g] for g in targets_global if g in g2local]

    # 4) Search paths
    paths = []
    if sources_local:
        paths = beam_search_paths_subgraph(
            sub_ei=sub_ei,
            edge_scores=edge_scores,
            subset_nodes=subset,
            node_types_sub=node_types_sub,
            sources_local=sources_local,
            targets_local=set(targets_local) if targets_local else set(),
            max_hops=args.max_hops,
            beam_width=args.beam_width,
            max_paths=args.max_paths
        )

    if paths:
        print(f"\n=== Top {len(paths)} multi-hop path(s) ===")
        for i, p in enumerate(paths, 1):
            labels = []
            for loc in p.nodes:
                g = int(subset[int(loc)])
                drkg_id = idx2node[g]
                rows = alias_df[alias_df["drkg_id"]==drkg_id]
                label = rows.iloc[0]["canonical_label"] if not rows.empty else drkg_id
                labels.append(label)
            print(f"Path {i} (score={p.score:.2f}, hops={len(p.nodes)-1}): " + " → ".join(labels))
    else:
        print("\n(no multi-hop path found)")

    # 5) Collect evidence
    evidence_lines: List[str] = []
    if paths:
        ranked = rank_edges_by_path_support(paths)
        for (eid, _score, _cnt) in ranked[:args.max_evidence]:
            evidence_lines.append(verbalize_edge_local(eid, sub_ei, subset, idx2node, alias_df, rel_full))

    if args.include_1hop and chosen_ids:
        id_set = set(chosen_ids)
        sub1 = df[(df["h"].isin(id_set)) | (df["t"].isin(id_set))].head(max(0, args.max_evidence - len(evidence_lines)))
        for _, row in sub1.iterrows():
            h, t = row["h"], row["t"]
            def label(x):
                rows = alias_df[alias_df["drkg_id"]==x]
                return rows.iloc[0]["canonical_label"] if not rows.empty else x
            line = f"- Fact: {label(h)} — {row['r']} — {label(t)}. [{h} ; {t}]"
            if line not in evidence_lines:
                evidence_lines.append(line)
                if len(evidence_lines) >= args.max_evidence: break

    print("\n=== Evidence ({} line(s)) ===".format(len(evidence_lines)))
    if evidence_lines:
        for ln in evidence_lines: print(ln)
    else:
        print("- (no relevant DRKG facts found)")

    # 6) Produce permutations
    perms = permute_evidence(evidence_lines, m=args.m)
    print("\n=== Example Evidence Permutation (1 of {}) ===".format(args.m))
    print("Evidence:")
    print("\n".join(perms[0]))

if __name__ == "__main__":
    main()