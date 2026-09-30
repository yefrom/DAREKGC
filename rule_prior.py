import json
from collections import Counter, defaultdict, deque
from typing import Dict, List, Tuple

from logger_config import logger


class RelationRulePrior:
    """Relation-level closed-path sampler inspired by DRAM.

    The sampler extracts relation paths from the training graph only and stores
    top paths per target relation. It never stores candidate tail entities, so it
    can be used as a lightweight rule prior for a bi-encoder.
    """

    def __init__(self,
                 train_path: str,
                 max_hop: int = 2,
                 topk: int = 8,
                 max_anchors_per_relation: int = 64,
                 max_paths_per_anchor: int = 8):
        self.train_path = train_path
        self.max_hop = max(2, int(max_hop))
        self.topk = max(1, int(topk))
        self.max_anchors_per_relation = max(1, int(max_anchors_per_relation))
        self.max_paths_per_anchor = max(1, int(max_paths_per_anchor))
        self.forward_graph = defaultdict(list)
        self.reverse_graph = defaultdict(list)
        self.paths_by_relation: Dict[str, List[dict]] = {}

        self._build()

    @staticmethod
    def inverse_relation(relation: str) -> str:
        return relation[len('inverse '):] if relation.startswith('inverse ') else 'inverse {}'.format(relation)

    def get_paths(self, relation: str) -> List[dict]:
        return self.paths_by_relation.get(str(relation), [])

    def _add_edge(self, head_id: str, relation: str, tail_id: str):
        self.forward_graph[head_id].append((relation, tail_id))
        self.reverse_graph[tail_id].append((relation, head_id))

    def _build(self):
        logger.info('Start to build relation rule prior from {}'.format(self.train_path))
        examples = json.load(open(self.train_path, 'r', encoding='utf-8'))
        anchors_by_relation = defaultdict(list)

        for ex in examples:
            head_id, relation, tail_id = ex['head_id'], ex['relation'], ex['tail_id']
            inv_relation = self.inverse_relation(relation)
            self._add_edge(head_id, relation, tail_id)
            self._add_edge(tail_id, inv_relation, head_id)
            anchors_by_relation[relation].append((head_id, relation, tail_id))
            anchors_by_relation[inv_relation].append((tail_id, inv_relation, head_id))

        for entity_id in list(self.forward_graph.keys()):
            self.forward_graph[entity_id] = sorted(self.forward_graph[entity_id],
                                                   key=lambda item: (item[0], item[1]))
        for entity_id in list(self.reverse_graph.keys()):
            self.reverse_graph[entity_id] = sorted(self.reverse_graph[entity_id],
                                                   key=lambda item: (item[0], item[1]))

        for relation, anchors in anchors_by_relation.items():
            counter = Counter()
            for head_id, rel, tail_id in anchors[:self.max_anchors_per_relation]:
                for relation_path in self._closed_paths(head_id, rel, tail_id):
                    counter[tuple(relation_path)] += 1
            total = sum(counter.values())
            if total == 0:
                self.paths_by_relation[relation] = []
                continue
            ranked_paths = []
            for relation_path, support in counter.most_common(self.topk):
                ranked_paths.append({
                    'relations': list(relation_path),
                    'support': support,
                    'score': float(support) / float(total),
                })
            self.paths_by_relation[relation] = ranked_paths

        non_empty = sum(1 for paths in self.paths_by_relation.values() if paths)
        logger.info('Done build relation rule prior: {} relations, {} with rules'
                    .format(len(self.paths_by_relation), non_empty))

    def _closed_paths(self, head_id: str, target_relation: str, tail_id: str) -> List[Tuple[str, ...]]:
        left_depth = max(1, self.max_hop // 2)
        right_depth = max(1, self.max_hop - left_depth)

        left_paths = self._paths_from_head(head_id, left_depth)
        right_paths = self._paths_to_tail(tail_id, right_depth)

        results = []
        for meet_entity in sorted(set(left_paths.keys()) & set(right_paths.keys())):
            for left_relations in left_paths[meet_entity]:
                for right_relations in right_paths[meet_entity]:
                    relation_path = tuple(left_relations + right_relations)
                    if len(relation_path) < 2:
                        continue
                    if len(relation_path) > self.max_hop:
                        continue
                    if relation_path == (target_relation,):
                        continue
                    results.append(relation_path)
                    if len(results) >= self.max_paths_per_anchor:
                        return results
        return results

    def _paths_from_head(self, head_id: str, max_depth: int):
        paths = defaultdict(list)
        queue = deque([(head_id, [], frozenset([head_id]))])
        while queue:
            entity_id, relation_seq, visited = queue.popleft()
            if relation_seq:
                paths[entity_id].append(relation_seq)
            if len(relation_seq) >= max_depth:
                continue
            for relation, next_entity in self.forward_graph.get(entity_id, []):
                if next_entity in visited:
                    continue
                queue.append((next_entity,
                              relation_seq + [relation],
                              visited | frozenset([next_entity])))
        return paths

    def _paths_to_tail(self, tail_id: str, max_depth: int):
        paths = defaultdict(list)
        queue = deque([(tail_id, [], frozenset([tail_id]))])
        while queue:
            entity_id, relation_seq_to_tail, visited = queue.popleft()
            paths[entity_id].append(relation_seq_to_tail)
            if len(relation_seq_to_tail) >= max_depth:
                continue
            for relation, prev_entity in self.reverse_graph.get(entity_id, []):
                if prev_entity in visited:
                    continue
                queue.append((prev_entity,
                              [relation] + relation_seq_to_tail,
                              visited | frozenset([prev_entity])))
        return paths
