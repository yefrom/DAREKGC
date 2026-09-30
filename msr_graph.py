import json
import math
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from logger_config import logger


@dataclass(frozen=True)
class PathEdge:
    relation: str
    target: str


class MultiHopPathGraph:
    """Bounded multi-hop path evidence graph with deterministic caching."""

    def __init__(self, train_path: str, max_cache_size: int = 50000):
        logger.info('Start to build multi-hop path graph from {}'.format(train_path))
        self.train_path = train_path
        self.max_cache_size = max_cache_size
        self.adjacency: Dict[str, List[PathEdge]] = defaultdict(list)
        self._edge_set = set()
        self.path_cache = OrderedDict()

        examples = json.load(open(train_path, 'r', encoding='utf-8'))
        for ex in examples:
            head_id, relation, tail_id = ex['head_id'], ex['relation'], ex['tail_id']
            self._add_edge(head_id, relation, tail_id)
            self._add_edge(tail_id, self._inverse_relation(relation), head_id)

        for entity_id in list(self.adjacency.keys()):
            self.adjacency[entity_id] = sorted(self.adjacency[entity_id],
                                               key=lambda edge: (edge.relation, edge.target))

        logger.info('Done build multi-hop path graph with {} nodes and {} directed edges'
                    .format(len(self.adjacency), len(self._edge_set)))

    @staticmethod
    def _inverse_relation(relation: str) -> str:
        return relation[len('inverse '):] if relation.startswith('inverse ') else 'inverse {}'.format(relation)

    @staticmethod
    def _base_relation(relation: str) -> str:
        return relation[len('inverse '):] if relation.startswith('inverse ') else relation

    def _add_edge(self, source: str, relation: str, target: str):
        edge_key = (source, relation, target)
        if edge_key in self._edge_set:
            return
        self._edge_set.add(edge_key)
        self.adjacency[source].append(PathEdge(relation=relation, target=target))

    def _cache_get(self, key):
        if key not in self.path_cache:
            return None
        value = self.path_cache[key]
        self.path_cache.move_to_end(key)
        return value

    def _cache_put(self, key, value):
        self.path_cache[key] = value
        self.path_cache.move_to_end(key)
        while len(self.path_cache) > self.max_cache_size:
            self.path_cache.popitem(last=False)

    def get_paths(self,
                  entity_id: str,
                  query_relation: str,
                  path_hops: Iterable[int],
                  max_path_hop: int,
                  max_paths_per_entity: int,
                  path_topk: int,
                  path_score_threshold: float,
                  path_mode: str) -> List[dict]:
        if not entity_id:
            return []

        hops = tuple(sorted(set(int(h) for h in path_hops if int(h) > 0)))
        if not hops:
            return []

        max_path_hop = min(int(max_path_hop), max(hops))
        hops = tuple(h for h in hops if h <= max_path_hop)
        if not hops:
            return []

        cache_key = (entity_id, query_relation, hops, max_path_hop,
                     max_paths_per_entity, path_topk, path_score_threshold, path_mode)
        cached_paths = self._cache_get(cache_key)
        if cached_paths is not None:
            return cached_paths

        paths = self._bounded_search(entity_id=entity_id,
                                     query_relation=query_relation,
                                     path_hops=hops,
                                     max_path_hop=max_path_hop,
                                     max_paths_per_entity=max_paths_per_entity)
        paths = self._select_paths(paths=paths,
                                   path_mode=path_mode,
                                   path_topk=path_topk,
                                   path_score_threshold=path_score_threshold,
                                   max_paths_per_entity=max_paths_per_entity)
        self._cache_put(cache_key, paths)
        return paths

    def _bounded_search(self,
                        entity_id: str,
                        query_relation: str,
                        path_hops: Tuple[int, ...],
                        max_path_hop: int,
                        max_paths_per_entity: int) -> List[dict]:
        hop_set = set(path_hops)
        per_hop_limit = max(1, math.ceil(max_paths_per_entity / max(1, len(hop_set))))
        frontier_cap = max(32, max_paths_per_entity * 4)

        paths = []
        counts_by_hop = defaultdict(int)
        frontier = [(entity_id, [], [], frozenset([entity_id]))]

        for depth in range(1, max_path_hop + 1):
            next_frontier = []
            for current_entity, relation_seq, node_seq, visited in frontier:
                for edge in self.adjacency.get(current_entity, []):
                    if edge.target in visited:
                        continue

                    next_relations = relation_seq + [edge.relation]
                    next_nodes = node_seq + [edge.target]
                    next_visited = visited | frozenset([edge.target])

                    if depth in hop_set and counts_by_hop[depth] < per_hop_limit:
                        paths.append(self._make_path(entity_id,
                                                     query_relation,
                                                     next_relations,
                                                     next_nodes,
                                                     depth))
                        counts_by_hop[depth] += 1

                    if depth < max_path_hop:
                        next_frontier.append((edge.target, next_relations, next_nodes, next_visited))

                    if len(next_frontier) >= frontier_cap and depth < max_path_hop:
                        break
                if len(next_frontier) >= frontier_cap and depth < max_path_hop:
                    break

            if not next_frontier:
                break

            next_frontier.sort(key=lambda item: self._partial_path_key(item, query_relation))
            frontier = next_frontier[:frontier_cap]

        return paths[:max_paths_per_entity]

    def _make_path(self,
                   start_entity: str,
                   query_relation: str,
                   relation_seq: List[str],
                   node_seq: List[str],
                   hop: int) -> dict:
        return {
            'start_entity': start_entity,
            'relations': relation_seq,
            'relation_ids': relation_seq,
            'intermediate_entities': node_seq[:-1],
            'end_entity': node_seq[-1],
            'hop': hop,
            'path_score': self._score_path(relation_seq, node_seq[-1], query_relation),
        }

    def _partial_path_key(self, item, query_relation: str):
        current_entity, relation_seq, node_seq, _ = item
        score = self._score_path(relation_seq, current_entity, query_relation)
        return (-score, len(relation_seq), tuple(relation_seq), tuple(node_seq))

    def _score_path(self, relation_seq: List[str], end_entity: str, query_relation: str) -> float:
        if not relation_seq:
            return 0.0

        query_base = self._base_relation(query_relation or '')
        relation_bases = [self._base_relation(rel) for rel in relation_seq]
        exact_match = sum(1 for rel in relation_seq if rel == query_relation)
        base_match = sum(1 for rel in relation_bases if rel == query_base and query_base)
        inverse_match = sum(1 for rel in relation_seq if rel == self._inverse_relation(query_relation or ''))
        hop_penalty = 1.0 / max(1, len(relation_seq))
        degree_penalty = 1.0 / (1.0 + math.log1p(len(self.adjacency.get(end_entity, []))))

        return round(0.5 * hop_penalty +
                     0.35 * exact_match +
                     0.2 * base_match +
                     0.1 * inverse_match +
                     0.15 * degree_penalty, 6)

    def _select_paths(self,
                      paths: List[dict],
                      path_mode: str,
                      path_topk: int,
                      path_score_threshold: float,
                      max_paths_per_entity: int) -> List[dict]:
        if not paths:
            return []

        if path_mode == 'selected':
            selected = [path for path in paths if path['path_score'] >= path_score_threshold]
            selected.sort(key=lambda path: (-path['path_score'],
                                            path['hop'],
                                            tuple(path['relations']),
                                            path['end_entity']))
            return selected[:path_topk]

        paths.sort(key=lambda path: (path['hop'],
                                     tuple(path['relations']),
                                     path['end_entity']))
        return paths[:max_paths_per_entity]
