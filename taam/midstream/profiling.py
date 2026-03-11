from __future__ import annotations

from typing import Dict, List, Set, Tuple

from ..types import NodeId, TAAMGraph, make_empty_adj


def _bfs_shortest(src: NodeId, out_edges: Dict[NodeId, Set[NodeId]]) -> Tuple[Dict[NodeId, int], Dict[NodeId, int]]:
    queue = [src]
    dist = {src: 0}
    count = {src: 1}
    i = 0
    while i < len(queue):
        cur = queue[i]
        i += 1
        for nxt in out_edges[cur]:
            nd = dist[cur] + 1
            if nxt not in dist:
                dist[nxt] = nd
                count[nxt] = count[cur]
                queue.append(nxt)
            elif dist[nxt] == nd:
                count[nxt] += count[cur]
    return dist, count


class TopologicalProfiler:
    """
    Phase 2:
    Compute C_bet(v) for lemma nodes and rank as mask candidates.
    """

    @staticmethod
    def lemma_betweenness(graph: TAAMGraph) -> Dict[NodeId, float]:
        target = graph.target()
        rev_edges = make_empty_adj(graph.nodes.keys())
        for src, dsts in graph.out_edges.items():
            for dst in dsts:
                rev_edges[dst].add(src)

        dist_to_t, cnt_to_t = _bfs_shortest(target, rev_edges)
        scores = {lid: 0.0 for lid in graph.lemmas()}

        for premise in graph.premises():
            dist_s, cnt_s = _bfs_shortest(premise, graph.out_edges)
            if target not in dist_s or cnt_s[target] == 0:
                continue
            st_dist = dist_s[target]
            sigma_st = cnt_s[target]
            for lemma in scores.keys():
                if lemma not in dist_s or lemma not in dist_to_t:
                    continue
                if dist_s[lemma] + dist_to_t[lemma] != st_dist:
                    continue
                sigma_through = cnt_s[lemma] * cnt_to_t[lemma]
                scores[lemma] += sigma_through / sigma_st
        return scores

    @staticmethod
    def rank_candidates(graph: TAAMGraph) -> List[Tuple[NodeId, float]]:
        scores = TopologicalProfiler.lemma_betweenness(graph)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
