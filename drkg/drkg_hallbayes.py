#!/usr/bin/env python3
"""
End-to-end DRKG → Evidence Collection → Hallucination Assessment
Automatically downloads DRKG if needed, uses DrugBank vocabulary for drug name resolution.
"""

import os
import sys
import argparse
import tarfile
import urllib.request
from collections import defaultdict

# Ensure repo root (parent of this file) is importable for sibling packages
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Check for required imports before starting
try:
    from drkg_link_and_collect import (
        load_drkg_edges,
        build_alias_index,
        link_mentions,
        extract_intent,
        build_pyg_graph,
        khop_neighborhood,
        beam_search_paths_subgraph,
        build_edge_scores,
        rank_edges_by_path_support,
        verbalize_edge_local,
        RELATION_HINTS_DEFAULT,
        TYPE_BONUS_DEFAULT
    )
except ImportError:
    print("Error: drkg_pyg_fixed.py not found in current directory")
    sys.exit(1)

try:
    from hallbayes.hallucination_toolkit import (
        OpenAIBackend,
        OpenAIPlanner,
        OpenAIItem,
        generate_answer_if_allowed
    )
except ImportError:
    print("Error: hallbayes package not found (install with `pip install .` from repo root)")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("Error: PyTorch not installed. Run: pip install torch")
    sys.exit(1)



def check_drugbank_vocab(path):
    """Check if DrugBank vocabulary exists."""
    if os.path.exists(path):
        return True
    
    print(f"\n⚠️  DrugBank vocabulary not found at: {path}")
    print("Please download it from: https://go.drugbank.com/releases/latest#open-data")
    print("Save it as: ./drugbank-vocabulary.csv")
    print("\nNote: DrugBank vocabulary improves drug name recognition but is optional.")
    response = input("Continue without DrugBank vocabulary? (y/n): ").strip().lower()
    return response == 'y'


def main():
    parser = argparse.ArgumentParser(description="DRKG-based hallucination assessment for medical queries")
    parser.add_argument('--drkg', default='./data/drkg/drkg.tsv', help='Path to drkg.tsv')
    parser.add_argument('--drugbank-vocab', default='./drugbank-vocabulary.csv', help='Path to DrugBank vocabulary CSV')
    parser.add_argument('--prompt', required=True, help='Medical query to evaluate')
    parser.add_argument('--max-evidence', type=int, default=20, help='Maximum evidence facts to collect')
    parser.add_argument('--max-hops', type=int, default=3, help='Maximum hops in knowledge graph')
    parser.add_argument('--beam-width', type=int, default=500, help='Beam width for path search')
    parser.add_argument('--max-paths', type=int, default=15, help='Maximum paths to find')
    parser.add_argument('--h-star', type=float, default=0.05, help='Target hallucination rate')
    parser.add_argument('--isr-threshold', type=float, default=1.0, help='ISR threshold for answering')
    parser.add_argument('--model', default='gpt-4o-mini', help='OpenAI model to use')
    args = parser.parse_args()


    # Check DrugBank vocabulary (optional)
    if not check_drugbank_vocab(args.drugbank_vocab):
        args.drugbank_vocab = None

    # Step 1: Load DRKG and build alias index
    print("\nLoading DRKG...")
    df_edges = load_drkg_edges(args.drkg)
    
    print("Building alias index...")
    if args.drugbank_vocab:
        print("  Using DrugBank vocabulary for improved drug recognition")
        alias_df = build_alias_index(df_edges, refresh=False, drugbank_path=args.drugbank_vocab)
    else:
        print("  Using basic aliasing (no DrugBank vocabulary)")
        alias_df = build_alias_index(df_edges, refresh=False, drugbank_path=None)

    # Step 2: Link entities in prompt
    print(f"\nQuery: {args.prompt}")
    print("Linking entities...")
    
    links = link_mentions(args.prompt, alias_df)
    by_type = defaultdict(list)
    for c in links:
        by_type[c["type"]].append(c)
    
    # Select best entity per type
    chosen = []
    for t in ["Compound", "Disease", "Gene", "Pathway"]:
        if by_type[t]:
            chosen.append(max(by_type[t], key=lambda x: x["score"]))
    chosen_ids = [x["drkg_id"] for x in chosen]

    if not chosen:
        print("\n⚠️ No entities recognized in the query")
        print("Try rephrasing with specific drug or disease names")
        sys.exit(1)

    print("\nLinked entities:")
    for c in chosen:
        print(f"  {c['mention']} → {c['drkg_id']} ({c['canonical_label']})")

    # Step 3: Determine sources and targets
    intent = extract_intent(args.prompt)
    relation_bias = intent.get("relation_bias", RELATION_HINTS_DEFAULT)
    
    comp_ids = [c["drkg_id"] for c in chosen if c["type"]=="Compound"]
    dis_ids = [c["drkg_id"] for c in chosen if c["type"]=="Disease"]
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
            sources_ids = [chosen_ids[0]]
            targets_ids = [chosen_ids[1]]
        else:
            sources_ids = chosen_ids[:1] if chosen_ids else []
            targets_ids = []

    # Step 4: Build graph and find paths
    print("\nSearching for evidence paths...")
    allowed_types = ["Compound", "Gene", "Pathway", "Disease"]
    ei, node2idx, idx2node, node_types, rel_list = build_pyg_graph(
        df_edges, allowed_types, undirected=True
    )
    
    # Get evidence from paths
    evidence_lines = []
    if sources_ids and sources_ids[0] in node2idx:
        sources_global_idx = [node2idx[s] for s in sources_ids if s in node2idx]
        
        # Extract k-hop subgraph
        subset, sub_ei, mapping, edge_mask = khop_neighborhood(
            ei, sources_global_idx, args.max_hops, len(node_types)
        )
        
        print(f"  Subgraph: {subset.numel()} nodes, {sub_ei.size(1)} edges")
        
        # Build subgraph components
        node_types_sub = [node_types[int(g)] for g in subset.tolist()]
        rel_full = [rel_list[i] for i,keep in enumerate(edge_mask.tolist()) if keep]
        
        # Score edges
        edge_scores = build_edge_scores(
            sub_ei, rel_full, node_types_sub, relation_bias, TYPE_BONUS_DEFAULT
        )
        
        # Map to local indices
        g2local = {int(g):i for i,g in enumerate(subset.tolist())}
        sources_local = [g2local[g] for g in sources_global_idx if g in g2local]
        targets_local = []
        if targets_ids:
            targets_global = [node2idx[t] for t in targets_ids if t in node2idx]
            targets_local = [g2local[g] for g in targets_global if g in g2local]
        
        # Search paths
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
        
        print(f"  Found {len(paths)} paths")
        
        # Collect evidence from paths
        if paths:
            ranked = rank_edges_by_path_support(paths)
            for (eid, _score, _cnt) in ranked[:args.max_evidence]:
                evidence_lines.append(verbalize_edge_local(
                    eid, sub_ei, subset, idx2node, alias_df, rel_full
                ))
    
    print(f"\nFound {len(evidence_lines)} evidence facts")
    if evidence_lines:
        print("Sample evidence:")
        for line in evidence_lines[:3]:
            print(f"  {line}")

    # Step 5: Build prompt with evidence
    evidence_block = "\n".join(evidence_lines) if evidence_lines else "- (no relevant facts found)"
    
    full_prompt = f"""Question: {args.prompt}

Evidence:
{evidence_block}

Based on the evidence from the Drug Repurposing Knowledge Graph, please answer the question accurately. If the evidence is insufficient, state that clearly."""

    # Step 6: Run hallucination assessment
    print("\nRunning hallucination risk assessment...")
    
    try:
        backend = OpenAIBackend(model=args.model)
    except Exception as e:
        print(f"Error initializing OpenAI: {e}")
        print("Make sure OPENAI_API_KEY is set in your environment")
        sys.exit(1)
    
    planner = OpenAIPlanner(backend, temperature=0.5, q_floor=0.1)
    
    item = OpenAIItem(
        prompt=full_prompt,
        n_samples=3,
        m=6,
        skeleton_policy="evidence_erase",
        fields_to_erase=["Evidence"]
    )
    
    try:
        metrics = planner.evaluate_item(
            idx=0,
            item=item,
            h_star=args.h_star,
            isr_threshold=args.isr_threshold,
            B_clip=12.0,
            clip_mode="one-sided"
        )
    except Exception as e:
        print(f"Error during assessment: {e}")
        sys.exit(1)

    # Step 7: Report results
    print(f"\n{'='*80}")
    print(f"DECISION: {'ANSWER' if metrics.decision_answer else 'REFUSE'}")
    print(f"{'='*80}")
    print(f"\nSafety Metrics:")
    print(f"  Δ̄ (information gain): {metrics.delta_bar:.4f} nats")
    print(f"  B2T (bits to trust):   {metrics.b2t:.4f} nats")
    print(f"  ISR:                   {metrics.isr:.3f} (threshold: {args.isr_threshold})")
    print(f"  RoH bound:             {metrics.roh_bound:.3f} (target: {args.h_star})")
    print(f"\n{metrics.rationale}")

    # Step 8: Generate answer if safe
    if metrics.decision_answer:
        print(f"\n{'='*80}")
        print("ANSWER:")
        print(f"{'='*80}")
        answer = generate_answer_if_allowed(backend, item, metrics)
        if answer:
            print(answer)
    else:
        print("\nNo answer generated (safety gate triggered)")
        print("The model cannot reliably answer based on the available evidence.")


if __name__ == '__main__':
    main()
