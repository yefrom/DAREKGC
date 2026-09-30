from abc import ABC
from copy import deepcopy
import json
import os
from collections import defaultdict

import torch
import torch.nn as nn
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

from transformers import AutoModel, AutoConfig
from dict_hub import get_cotail_graph, get_dynamic_cache, get_cotail_graph_valid
from rule_prior import RelationRulePrior
from triplet_mask import construct_mask
import torch.nn.functional as F


def build_model(args) -> nn.Module:
    if args.model_name in ['msr-progkgc', 'msr_progkgc']:
        return MSRProgKGC(args)
    if (args.use_st_fusion or args.use_relation_aware_gnn or
            args.use_dynamic_relation_memory or args.use_rule_path_prior):
        return ResearchProgKGC(args)
    return CustomBertModel(args)


@dataclass
class ModelOutput:
    logits: torch.tensor
    labels: torch.tensor
    inv_t: torch.tensor
    hr_vector: torch.tensor
    tail_vector: torch.tensor
    msc_loss: torch.tensor = None
    contrastive_loss: torch.tensor = None
    total_loss: torch.tensor = None
    msr_stats: dict = None


class GNNLayer(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
        )

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.guide_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):

        for layer in [self.attention, self.ffn]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
        nn.init.xavier_uniform_(self.guide_proj.weight)
        nn.init.constant_(self.guide_proj.bias, 0.0)

    def forward(self,
                query: torch.Tensor,
                neighbors: torch.Tensor,
                guide: torch.Tensor = None) -> torch.Tensor:

        if neighbors.size(0) == 0:
            return query

        if guide is not None:
            attn_query = torch.tanh(self.guide_proj(torch.cat([query, guide], dim=-1)))
        else:
            attn_query = query

        query_3d = query.view(1, 1, -1)
        attn_query_3d = attn_query.view(1, 1, -1)
        neighbors_3d = neighbors.unsqueeze(1)  # [K, 1, D]

        attn_output, _ = self.attention(
            query=attn_query_3d,  # [1, 1, D]
            key=neighbors_3d,  # [K, 1, D]
            value=neighbors_3d,  # [K, 1, D]
            need_weights=False
        )

        query_3d = self.norm1(query_3d + self.dropout(attn_output))  # [1, 1, D]

        ffn_output = self.ffn(query_3d)
        output = self.norm2(query_3d + self.dropout(ffn_output))  # [1, 1, D]

        return output.squeeze(0).squeeze(0)  # [D]


class CacheProfiler:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cache_hits = 0
        self.cache_misses = 0
        self.cache_lookup_time = 0.0
        self.cache_store_time = 0.0
        self.gnn_computation_time = 0.0
        self.neighbor_retrieval_time = 0.0
        self.bert_single_call_times = []
        self.bert_batch_total_time = 0.0
        self.cache_memory_usage = []  # 记录每轮缓存的内存使用情况

    def record_cache_memory(self, memory_usage):
        """记录当前缓存的内存使用情况"""
        self.cache_memory_usage.append(memory_usage)

    def record_cache_hit(self):
        self.cache_hits += 1

    def record_cache_miss(self):
        self.cache_misses += 1

    def record_lookup_time(self, time_taken):
        self.cache_lookup_time += time_taken

    def record_store_time(self, time_taken):
        self.cache_store_time += time_taken

    def record_gnn_time(self, time_taken):
        self.gnn_computation_time += time_taken

    def record_neighbor_time(self, time_taken):
        self.neighbor_retrieval_time += time_taken

    def record_bert_single_call(self, time_taken):
        """记录单次BERT调用时间"""
        self.bert_single_call_times.append(time_taken)

    def record_bert_batch_time(self, time_taken):
        """记录一个批次的BERT总时间"""
        self.bert_batch_total_time += time_taken

    def get_stats(self):
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total_requests if total_requests > 0 else 0

        # 计算平均内存使用
        total_memory = sum(usage['total_memory'] for usage in self.cache_memory_usage)
        avg_memory = total_memory / len(self.cache_memory_usage) if self.cache_memory_usage else 0

        # 获取最后一次记录的内存使用（当前状态）
        current_memory = self.cache_memory_usage[-1] if self.cache_memory_usage else {
            'hr_cache_memory': 0,
            'tail_cache_memory': 0,
            'total_memory': 0
        }

        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_lookup_time': self.cache_lookup_time,
            'cache_store_time': self.cache_store_time,
            'gnn_computation_time': self.gnn_computation_time,
            'neighbor_retrieval_time': self.neighbor_retrieval_time,
            'total_cache_time': self.cache_lookup_time + self.cache_store_time,
            'bert_single_call_times': self.bert_single_call_times,
            'bert_batch_total_time': self.bert_batch_total_time,
            'bert_avg_single_time': sum(self.bert_single_call_times) / len(
                self.bert_single_call_times) if self.bert_single_call_times else 0,
            'bert_call_count': len(self.bert_single_call_times),
            'cache_memory_usage': {
                'current': current_memory,
                'average_total': avg_memory,
                'all_entries': self.cache_memory_usage
            }
        }


# 全局profiler实例
cache_profiler = CacheProfiler()


class CustomBertModel(nn.Module, ABC):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained("model/bert-base-uncased")
        self.log_inv_t = torch.nn.Parameter(torch.tensor(1.0 / args.t).log(), requires_grad=args.finetune_t)
        self.add_margin = args.additive_margin
        self.batch_size = args.batch_size
        self.pre_batch = args.pre_batch
        num_pre_batch_vectors = max(1, self.pre_batch) * self.batch_size
        random_vector = torch.randn(num_pre_batch_vectors, self.config.hidden_size)
        self.register_buffer("pre_batch_vectors",
                             nn.functional.normalize(random_vector, dim=1),
                             persistent=False)
        self.offset = 0
        self.pre_batch_exs = [None for _ in range(num_pre_batch_vectors)]

        self.hr_bert = AutoModel.from_pretrained("model/bert-base-uncased")
        self.tail_bert = deepcopy(self.hr_bert)
        self.hr_gnn = GNNLayer(self.config.hidden_size)
        self.tail_gnn = GNNLayer(self.config.hidden_size)

        self.use_relation_features = (
            args.use_relation_aware_gnn or
            args.use_dynamic_relation_memory or
            args.use_rule_path_prior
        )
        self.use_relation_vocab = self.use_relation_features or args.use_brnq
        self.relation_to_idx = {}
        self.unk_relation_idx = 0
        if self.use_relation_vocab:
            self.relation_to_idx = self._build_relation_vocab()
            self.unk_relation_idx = self.relation_to_idx.get('<unk>', 0)
        if self.use_relation_features:
            self.relation_embeddings = nn.Embedding(len(self.relation_to_idx),
                                                    self.config.hidden_size)

        if args.use_brnq:
            self.register_buffer(
                "brnq_vectors",
                torch.zeros(args.brnq_size, self.config.hidden_size),
                persistent=False,
            )
            self.register_buffer(
                "brnq_relation_ids",
                torch.full((args.brnq_size,), -1, dtype=torch.long),
                persistent=False,
            )
            self.brnq_offset = 0
            self.brnq_filled = 0
            self.brnq_exs = [None for _ in range(args.brnq_size)]

        if args.use_dynamic_relation_memory:
            self.relation_memory_attention = nn.MultiheadAttention(
                embed_dim=self.config.hidden_size,
                num_heads=args.relation_memory_heads,
                dropout=args.dropout,
                batch_first=True,
            )
            self.relation_memory_ffn = nn.Sequential(
                nn.Linear(self.config.hidden_size, 4 * self.config.hidden_size),
                nn.GELU(),
                nn.Dropout(args.dropout),
                nn.Linear(4 * self.config.hidden_size, self.config.hidden_size),
            )
            self.relation_memory_norm1 = nn.LayerNorm(self.config.hidden_size)
            self.relation_memory_norm2 = nn.LayerNorm(self.config.hidden_size)
            self.relation_memory_dropout = nn.Dropout(args.dropout)

        if args.use_st_fusion:
            fusion_input_dim = self.config.hidden_size * 4
            self.st_fusion_mlp = nn.Sequential(
                nn.Linear(fusion_input_dim, self.config.hidden_size),
                nn.GELU(),
                nn.Dropout(args.dropout),
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
            )
            self.st_gate = nn.Sequential(
                nn.Linear(fusion_input_dim, self.config.hidden_size),
                nn.GELU(),
                nn.Linear(self.config.hidden_size, 1),
            )
            self.st_difference = nn.Linear(self.config.hidden_size, self.config.hidden_size)
            self.st_norm = nn.LayerNorm(self.config.hidden_size)
            nn.init.zeros_(self.st_fusion_mlp[-1].weight)
            nn.init.zeros_(self.st_fusion_mlp[-1].bias)
            nn.init.zeros_(self.st_gate[-1].weight)
            nn.init.constant_(self.st_gate[-1].bias, -2.0)
            nn.init.zeros_(self.st_difference.weight)
            nn.init.zeros_(self.st_difference.bias)

        self.rule_prior = None
        if args.use_rule_path_prior:
            self.rule_prior = RelationRulePrior(
                train_path=args.train_path,
                max_hop=args.rule_max_hop,
                topk=args.rule_topk,
                max_anchors_per_relation=args.rule_max_anchors_per_relation,
                max_paths_per_anchor=args.rule_paths_per_anchor,
            )
            self.rule_prior_proj = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.GELU(),
                nn.Dropout(args.dropout),
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
            )

        # 添加性能监控
        self.cache_profiler = cache_profiler

    def _build_relation_vocab(self) -> Dict[str, int]:
        relations = {'<unk>'}
        for path in [self.args.train_path, self.args.valid_path]:
            if not path or not os.path.exists(path):
                continue
            for ex in json.load(open(path, 'r', encoding='utf-8')):
                relation = str(ex['relation'])
                relations.add(relation)
                relations.add(self._inverse_relation(relation))
        return {relation: idx for idx, relation in enumerate(sorted(relations))}

    @staticmethod
    def _inverse_relation(relation: str) -> str:
        return relation[len('inverse '):] if relation.startswith('inverse ') else 'inverse {}'.format(relation)

    @staticmethod
    def _current_batch_items(full_items, current_batch_size):
        if not full_items:
            return []
        if torch.cuda.device_count() > 1 and len(full_items) > current_batch_size:
            device_idx = torch.cuda.current_device()
            start_idx = device_idx * current_batch_size
            end_idx = start_idx + current_batch_size
            return full_items[start_idx:end_idx]
        return full_items[:current_batch_size]

    def _relation_memory(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        relation_vectors = self.relation_embeddings.weight.to(device=device, dtype=dtype)
        if not self.args.use_dynamic_relation_memory:
            return relation_vectors

        memory_input = relation_vectors.unsqueeze(0)
        memory_output, _ = self.relation_memory_attention(memory_input,
                                                          memory_input,
                                                          memory_input,
                                                          need_weights=False)
        memory = self.relation_memory_norm1(
            memory_input + self.relation_memory_dropout(memory_output))
        memory_ffn = self.relation_memory_ffn(memory)
        memory = self.relation_memory_norm2(
            memory + self.relation_memory_dropout(memory_ffn))
        return memory.squeeze(0)

    def _relation_ids_batch(self,
                            batch_data: List,
                            current_batch_size: int,
                            device: torch.device) -> torch.Tensor:
        if not self.use_relation_vocab or not batch_data:
            return torch.full((current_batch_size,),
                              self.unk_relation_idx,
                              device=device,
                              dtype=torch.long)

        relation_indices = [
            self.relation_to_idx.get(str(ex.relation), self.unk_relation_idx)
            for ex in batch_data[:current_batch_size]
        ]
        while len(relation_indices) < current_batch_size:
            relation_indices.append(self.unk_relation_idx)
        return torch.LongTensor(relation_indices[:current_batch_size]).to(device)

    def _relation_context_batch(self,
                                batch_data: List,
                                current_batch_size: int,
                                device: torch.device,
                                dtype: torch.dtype) -> torch.Tensor:
        if not self.use_relation_features or not batch_data:
            return torch.zeros(current_batch_size, self.config.hidden_size, device=device, dtype=dtype)

        relation_memory = self._relation_memory(device=device, dtype=dtype)
        relation_tensor = self._relation_ids_batch(batch_data, current_batch_size, device)
        return relation_memory.index_select(0, relation_tensor).to(dtype=dtype)

    def _module_mask_batch(self,
                           batch_data: List,
                           current_batch_size: int,
                           device: torch.device,
                           dtype: torch.dtype) -> torch.Tensor:
        if self.args.research_module_direction == 'all':
            return torch.ones(current_batch_size, 1, device=device, dtype=dtype)

        values = []
        for ex in batch_data[:current_batch_size]:
            is_backward = str(ex.relation).startswith('inverse ')
            if self.args.research_module_direction == 'backward':
                values.append(1.0 if is_backward else 0.0)
            else:
                values.append(0.0 if is_backward else 1.0)
        while len(values) < current_batch_size:
            values.append(0.0)
        return torch.tensor(values, device=device, dtype=dtype).view(current_batch_size, 1)

    def _rule_prior_batch(self,
                          batch_data: List,
                          current_batch_size: int,
                          device: torch.device,
                          dtype: torch.dtype) -> Tuple[torch.Tensor, dict]:
        zero_vec = torch.zeros(current_batch_size, self.config.hidden_size, device=device, dtype=dtype)
        zero_stats = {
            'rule_count': torch.zeros(current_batch_size, device=device, dtype=dtype),
            'rule_score': torch.zeros(current_batch_size, device=device, dtype=dtype),
        }
        if not self.args.use_rule_path_prior or self.rule_prior is None or not batch_data:
            return zero_vec, zero_stats

        relation_memory = self._relation_memory(device=device, dtype=dtype)
        prior_vecs, rule_counts, rule_scores = [], [], []
        for ex in batch_data[:current_batch_size]:
            paths = self.rule_prior.get_paths(ex.relation)[:self.args.rule_topk]
            if not paths:
                prior_vecs.append(torch.zeros(self.config.hidden_size, device=device, dtype=dtype))
                rule_counts.append(0.0)
                rule_scores.append(0.0)
                continue

            path_vecs, path_weights = [], []
            for path in paths:
                relation_indices = [
                    self.relation_to_idx.get(str(relation), self.unk_relation_idx)
                    for relation in path.get('relations', [])
                ]
                if not relation_indices:
                    continue
                relation_tensor = torch.LongTensor(relation_indices).to(device)
                path_vecs.append(relation_memory.index_select(0, relation_tensor).mean(dim=0))
                path_weights.append(float(path.get('score', 0.0)))

            if not path_vecs:
                prior_vecs.append(torch.zeros(self.config.hidden_size, device=device, dtype=dtype))
                rule_counts.append(0.0)
                rule_scores.append(0.0)
                continue

            path_tensor = torch.stack(path_vecs, dim=0)
            weight_tensor = torch.tensor(path_weights, device=device, dtype=dtype)
            weight_tensor = weight_tensor / weight_tensor.sum().clamp_min(1e-6)
            prior_vec = torch.sum(weight_tensor.unsqueeze(-1) * path_tensor, dim=0)
            prior_vecs.append(prior_vec)
            rule_counts.append(float(len(path_vecs)))
            rule_scores.append(float(sum(path_weights) / max(1, len(path_weights))))

        while len(prior_vecs) < current_batch_size:
            prior_vecs.append(torch.zeros(self.config.hidden_size, device=device, dtype=dtype))
            rule_counts.append(0.0)
            rule_scores.append(0.0)

        prior_tensor = self.rule_prior_proj(torch.stack(prior_vecs[:current_batch_size], dim=0))
        prior_tensor = F.normalize(prior_tensor, p=2, dim=1)
        stats = {
            'rule_count': torch.tensor(rule_counts[:current_batch_size], device=device, dtype=dtype),
            'rule_score': torch.tensor(rule_scores[:current_batch_size], device=device, dtype=dtype),
        }
        return prior_tensor, stats

    def _fuse_text_structure(self,
                             text_vec: torch.Tensor,
                             structure_vec: torch.Tensor,
                             module_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.args.use_st_fusion:
            gate = text_vec.new_ones(text_vec.size(0), 1)
            return F.normalize(structure_vec, p=2, dim=1), gate

        fusion_input = torch.cat([
            text_vec,
            structure_vec,
            text_vec * structure_vec,
            torch.abs(text_vec - structure_vec),
        ], dim=-1)
        interaction = self.st_fusion_mlp(fusion_input)
        gate = torch.sigmoid(self.st_gate(fusion_input))
        difference = self.st_difference(structure_vec - text_vec)
        fused = structure_vec + self.args.st_fusion_lambda * gate * interaction
        fused = fused + (1.0 - self.args.st_fusion_lambda) * difference
        fused = F.normalize(fused, p=2, dim=1)
        base = F.normalize(structure_vec, p=2, dim=1)
        if module_mask is not None:
            fused = torch.where(module_mask.bool(), fused, base)
            gate = gate * module_mask
        return fused, gate.detach()

    def _encode(self, encoder, token_ids, mask, token_type_ids):
        """带BERT时间监控的编码"""
        start_time = time.time()

        outputs = encoder(input_ids=token_ids,
                          attention_mask=mask,
                          token_type_ids=token_type_ids,
                          return_dict=True)

        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0, :]
        cls_output = _pool_output(self.args.pooling, cls_output, mask, last_hidden_state)

        # 记录单次BERT调用时间
        encoding_time = time.time() - start_time
        self.cache_profiler.record_bert_single_call(encoding_time)

        return cls_output

    def _get_cached_vectors_with_profiling(self, cache, entity_ids, vector_type):
        """带性能监控的缓存向量获取"""
        start_time = time.time()

        if vector_type == 'hr':
            vectors = [vec for vec in cache.get_hr_vectors(entity_ids) if vec is not None]
        else:  # tail
            vectors = [vec for vec in cache.get_tail_vectors(entity_ids) if vec is not None]

        lookup_time = time.time() - start_time
        self.cache_profiler.record_lookup_time(lookup_time)

        # 记录缓存命中/未命中
        hit_count = len(vectors)
        miss_count = len(entity_ids) - hit_count

        for _ in range(hit_count):
            self.cache_profiler.record_cache_hit()
        for _ in range(miss_count):
            self.cache_profiler.record_cache_miss()

        return vectors

    def _get_neighbors_with_profiling(self, graph, entity_id):
        """带性能监控的邻居获取"""
        start_time = time.time()
        neighbors = graph.get_cotail_neighbors(entity_id)
        neighbor_time = time.time() - start_time
        self.cache_profiler.record_neighbor_time(neighbor_time)
        return neighbors

    def _apply_gnn_with_profiling(self, gnn_layer, target_vector, neighbor_tensor, guide_vector=None):
        """带性能监控的GNN计算"""
        start_time = time.time()
        result = gnn_layer(target_vector, neighbor_tensor, guide=guide_vector)
        gnn_time = time.time() - start_time
        self.cache_profiler.record_gnn_time(gnn_time)
        return result

    def forward(self, hr_token_ids, hr_mask, hr_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,
                use_gnn=True,
                use_multi_hop_path=True,
                use_head_gnn=True,
                use_tail_gnn=True,
                only_ent_embedding=False, **kwargs) -> dict:

        def get_current_batch_data(full_batch_data, current_batch_size):
            """处理多GPU下的批次数据分割"""
            if not full_batch_data:
                return []

            # 自动检测是否在多GPU环境中
            if torch.is_tensor(hr_token_ids):
                device_count = torch.cuda.device_count()
                if device_count > 1 and len(full_batch_data) > current_batch_size:
                    # 获取当前GPU的索引
                    device_idx = torch.cuda.current_device()
                    # 计算子批次的起始位置
                    start_idx = device_idx * current_batch_size
                    end_idx = start_idx + current_batch_size
                    return full_batch_data[start_idx:end_idx]
            return full_batch_data[:current_batch_size]

        # 记录批次BERT开始时间
        batch_bert_start = time.time()

        if only_ent_embedding:
            result = self.predict_ent_embedding(tail_token_ids=tail_token_ids,
                                                tail_mask=tail_mask,
                                                tail_token_type_ids=tail_token_type_ids)
            # 记录批次BERT总时间
            batch_bert_time = time.time() - batch_bert_start
            self.cache_profiler.record_bert_batch_time(batch_bert_time)
            return result

        # 三次BERT编码调用
        hr_vector = self._encode(self.hr_bert,
                                 token_ids=hr_token_ids,
                                 mask=hr_mask,
                                 token_type_ids=hr_token_type_ids)

        tail_vector = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)

        head_vector = self._encode(self.tail_bert,
                                   token_ids=head_token_ids,
                                   mask=head_mask,
                                   token_type_ids=head_token_type_ids)

        # 记录批次BERT总时间
        batch_bert_time = time.time() - batch_bert_start
        self.cache_profiler.record_bert_batch_time(batch_bert_time)

        if not use_gnn:
            return {'hr_vector': hr_vector,
                    'tail_vector': tail_vector,
                    'head_vector': head_vector}

        # GNN处理 - 添加性能监控
        current_batch_size = hr_vector.size(0)  # 当前GPU的实际批次大小

        if use_gnn and not self.training:
            cache = get_dynamic_cache()
            full_batch_data = kwargs.get('batch_data', [])
            batch_data = get_current_batch_data(full_batch_data, current_batch_size)
            if use_head_gnn:
                updated_hr = []
                for i, ex in enumerate(batch_data):
                    cotail_heads = self._get_neighbors_with_profiling(
                        get_cotail_graph_valid(), ex.head_id)

                    neighbor_vectors = self._get_cached_vectors_with_profiling(
                        cache, cotail_heads, 'hr')

                    if neighbor_vectors:
                        neighbor_tensor = torch.stack(neighbor_vectors).to(hr_vector.device)
                        updated = self._apply_gnn_with_profiling(
                            self.hr_gnn, hr_vector[i], neighbor_tensor)
                        updated_hr.append(updated)
                    else:
                        updated_hr.append(hr_vector[i])
                hr_vector = torch.stack(updated_hr)

            if use_tail_gnn:
                updated_tail = []
                for i, ex in enumerate(batch_data):
                    cotail_tails = self._get_neighbors_with_profiling(
                        get_cotail_graph_valid(), ex.tail_id)

                    neighbor_vectors = self._get_cached_vectors_with_profiling(
                        cache, cotail_tails, 'tail')

                    if neighbor_vectors:
                        neighbor_tensor = torch.stack(neighbor_vectors).to(tail_vector.device)
                        updated = self._apply_gnn_with_profiling(
                            self.tail_gnn, tail_vector[i], neighbor_tensor)
                        updated_tail.append(updated)
                    else:
                        updated_tail.append(tail_vector[i])
                tail_vector = torch.stack(updated_tail)

        if use_gnn and self.training:
            cache = get_dynamic_cache()
            full_batch_data = kwargs.get('batch_data', [])
            batch_data = get_current_batch_data(full_batch_data, current_batch_size)
            if use_head_gnn:
                updated_hr = []
                for i, ex in enumerate(batch_data):
                    cotail_heads = self._get_neighbors_with_profiling(
                        get_cotail_graph(), ex.head_id)

                    neighbor_vectors = self._get_cached_vectors_with_profiling(
                        cache, cotail_heads, 'hr')

                    if neighbor_vectors:
                        neighbor_tensor = torch.stack(neighbor_vectors).to(hr_vector.device)
                        updated = self._apply_gnn_with_profiling(
                            self.hr_gnn, hr_vector[i], neighbor_tensor)
                        updated_hr.append(updated)
                    else:
                        updated_hr.append(hr_vector[i])
                hr_vector = torch.stack(updated_hr)

            if use_tail_gnn:
                updated_tail = []
                for i, ex in enumerate(batch_data):
                    cotail_tails = self._get_neighbors_with_profiling(
                        get_cotail_graph(), ex.tail_id)

                    neighbor_vectors = self._get_cached_vectors_with_profiling(
                        cache, cotail_tails, 'tail')

                    if neighbor_vectors:
                        neighbor_tensor = torch.stack(neighbor_vectors).to(tail_vector.device)
                        updated = self._apply_gnn_with_profiling(
                            self.tail_gnn, tail_vector[i], neighbor_tensor)
                        updated_tail.append(updated)
                    else:
                        updated_tail.append(tail_vector[i])
                tail_vector = torch.stack(updated_tail)

        hr_vector = F.normalize(hr_vector, p=2, dim=1)
        tail_vector = F.normalize(tail_vector, p=2, dim=1)

        return {'hr_vector': hr_vector,
                'tail_vector': tail_vector,
                'head_vector': head_vector}

    def get_cache_stats(self):
        """获取缓存统计信息"""
        return self.cache_profiler.get_stats()

    def reset_cache_stats(self):
        """重置缓存统计"""
        self.cache_profiler.reset()

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> dict:
        hr_vector, tail_vector = output_dict['hr_vector'], output_dict['tail_vector']
        batch_size = hr_vector.size(0)
        labels = torch.arange(batch_size).to(hr_vector.device)

        logits = hr_vector.mm(tail_vector.t())
        if self.training:
            logits -= torch.zeros(logits.size()).fill_diagonal_(self.add_margin).to(logits.device)
        logits *= self.log_inv_t.exp()

        triplet_mask = batch_dict.get('triplet_mask', None)
        if triplet_mask is not None:
            logits.masked_fill_(~triplet_mask, -1e4)

        if self.pre_batch > 0 and self.training:
            pre_batch_logits = self._compute_pre_batch_logits(hr_vector, tail_vector, batch_dict)
            logits = torch.cat([logits, pre_batch_logits], dim=-1)

        if self.args.use_self_negative and self.training:
            head_vector = output_dict['head_vector']
            self_neg_logits = torch.sum(hr_vector * head_vector, dim=1) * self.log_inv_t.exp()
            self_negative_mask = batch_dict['self_negative_mask']
            self_neg_logits.masked_fill_(~self_negative_mask, -1e4)
            logits = torch.cat([logits, self_neg_logits.unsqueeze(1)], dim=-1)
        if self.args.use_rs_negative and self.training:
            rs_neg_ids = batch_dict['rs_neg_token_ids']
            if rs_neg_ids.size(0) > 0:
                rs_neg_vec = self._encode(
                    self.tail_bert,
                    rs_neg_ids,
                    batch_dict['rs_neg_mask'],
                    batch_dict['rs_neg_token_type_ids']
                )
                K = rs_neg_vec.size(0) // batch_size
                rs_neg_vec = rs_neg_vec.view(batch_size, K, -1)
                rs_logits = torch.bmm(
                    hr_vector.unsqueeze(1),  # (B,1,D)
                    rs_neg_vec.transpose(1, 2)  # (B,D,K)
                ).squeeze(1) * self.log_inv_t.exp()
                logits = torch.cat([logits, rs_logits], dim=-1)

        if self.args.use_brnq and self.training:
            brnq_logits = self._compute_brnq_logits(hr_vector, tail_vector, batch_dict)
            if brnq_logits is not None:
                logits = torch.cat([logits, brnq_logits], dim=-1)

        result = {'logits': logits,
                  'labels': labels,
                  'inv_t': self.log_inv_t.detach().exp(),
                  'hr_vector': hr_vector.detach(),
                  'tail_vector': tail_vector.detach()}
        if 'msc_loss' in output_dict:
            result['msc_loss'] = output_dict['msc_loss']
        if 'msr_stats' in output_dict:
            result['msr_stats'] = output_dict['msr_stats']
        return result

    def _compute_pre_batch_logits(self, hr_vector: torch.tensor,
                                  tail_vector: torch.tensor,
                                  batch_dict: dict) -> torch.tensor:
        assert tail_vector.size(0) == self.batch_size
        batch_exs = batch_dict['batch_data']
        # batch_size x num_neg
        pre_batch_logits = hr_vector.mm(self.pre_batch_vectors.clone().t())
        pre_batch_logits *= self.log_inv_t.exp() * self.args.pre_batch_weight
        if self.pre_batch_exs[-1] is not None:
            pre_triplet_mask = construct_mask(batch_exs, self.pre_batch_exs).to(hr_vector.device)
            pre_batch_logits.masked_fill_(~pre_triplet_mask, -1e4)

        self.pre_batch_vectors[self.offset:(self.offset + self.batch_size)] = tail_vector.data.clone()
        self.pre_batch_exs[self.offset:(self.offset + self.batch_size)] = batch_exs
        self.offset = (self.offset + self.batch_size) % len(self.pre_batch_exs)

        return pre_batch_logits

    def _compute_brnq_logits(self,
                             hr_vector: torch.tensor,
                             tail_vector: torch.tensor,
                             batch_dict: dict):
        batch_exs = batch_dict.get('batch_data', [])
        if not batch_exs:
            return None

        current_batch_size = hr_vector.size(0)
        brnq_logits = None
        if self.brnq_filled > 0:
            queue_vectors = self.brnq_vectors[:self.brnq_filled].detach().clone().to(
                device=hr_vector.device,
                dtype=hr_vector.dtype,
            )
            queue_exs = self.brnq_exs[:self.brnq_filled]
            queue_relation_ids = self.brnq_relation_ids[:self.brnq_filled].to(hr_vector.device)
            row_relation_ids = self._relation_ids_batch(batch_exs,
                                                        current_batch_size,
                                                        hr_vector.device)

            brnq_logits = hr_vector.mm(queue_vectors.t())
            brnq_logits *= self.log_inv_t.exp() * self.args.brnq_weight

            direction_mask = self._module_mask_batch(batch_exs,
                                                     current_batch_size,
                                                     hr_vector.device,
                                                     torch.float32).bool()
            relation_mask = row_relation_ids.unsqueeze(1).eq(queue_relation_ids.unsqueeze(0))
            valid_mask = relation_mask & direction_mask
            if queue_exs and queue_exs[-1] is not None:
                triplet_mask = construct_mask(batch_exs, queue_exs).to(hr_vector.device)
                valid_mask = valid_mask & triplet_mask

            brnq_logits = brnq_logits.masked_fill(~valid_mask, -1e4)
            topk = min(self.args.brnq_topk, self.brnq_filled)
            if topk > 0:
                brnq_logits = torch.topk(brnq_logits, k=topk, dim=-1).values
            else:
                brnq_logits = None

        self._update_brnq_queue(tail_vector, batch_exs)
        return brnq_logits

    @torch.no_grad()
    def _update_brnq_queue(self, tail_vector: torch.tensor, batch_exs: List):
        if not batch_exs:
            return

        current_batch_size = min(tail_vector.size(0), len(batch_exs))
        direction_mask = self._module_mask_batch(batch_exs,
                                                 current_batch_size,
                                                 tail_vector.device,
                                                 torch.float32).view(-1).bool()
        relation_ids = self._relation_ids_batch(batch_exs,
                                                current_batch_size,
                                                self.brnq_relation_ids.device)

        for idx in range(current_batch_size):
            if not direction_mask[idx].item():
                continue
            self.brnq_vectors[self.brnq_offset] = tail_vector[idx].detach().to(
                device=self.brnq_vectors.device,
                dtype=self.brnq_vectors.dtype,
            )
            self.brnq_relation_ids[self.brnq_offset] = relation_ids[idx]
            self.brnq_exs[self.brnq_offset] = batch_exs[idx]
            self.brnq_offset = (self.brnq_offset + 1) % self.args.brnq_size
            self.brnq_filled = min(self.brnq_filled + 1, self.args.brnq_size)

    @torch.no_grad()
    def predict_ent_embedding(self, tail_token_ids, tail_mask, tail_token_type_ids, **kwargs) -> dict:
        ent_vectors = self._encode(self.tail_bert,
                                   token_ids=tail_token_ids,
                                   mask=tail_mask,
                                   token_type_ids=tail_token_type_ids)
        return {'ent_vectors': ent_vectors.detach()}


class ResearchProgKGC(CustomBertModel):
    """ProgKGC with independently switchable research modules."""

    def _relation_guide(self,
                        index: int,
                        hr_text_vector: torch.Tensor,
                        relation_context: torch.Tensor,
                        module_mask: torch.Tensor):
        if not self.args.use_relation_aware_gnn:
            return None
        if module_mask[index].item() <= 0.0:
            return None
        guide = hr_text_vector[index]
        if self.use_relation_features:
            guide = F.normalize(guide + relation_context[index], p=2, dim=0)
        return guide

    def _apply_batch_gnn(self,
                         base_vector: torch.Tensor,
                         hr_text_vector: torch.Tensor,
                         relation_context: torch.Tensor,
                         batch_data: List,
                         graph,
                         cache,
                         vector_type: str,
                         gnn_layer: nn.Module,
                         module_mask: torch.Tensor) -> torch.Tensor:
        updated_vectors = []
        for i, ex in enumerate(batch_data):
            entity_id = ex.head_id if vector_type == 'hr' else ex.tail_id
            neighbors = self._get_neighbors_with_profiling(graph, entity_id)
            neighbor_vectors = self._get_cached_vectors_with_profiling(cache, neighbors, vector_type)
            if neighbor_vectors:
                neighbor_tensor = torch.stack(neighbor_vectors).to(base_vector.device)
                updated = self._apply_gnn_with_profiling(
                    gnn_layer,
                    base_vector[i],
                    neighbor_tensor,
                    guide_vector=self._relation_guide(i, hr_text_vector, relation_context, module_mask))
                updated_vectors.append(updated)
            else:
                updated_vectors.append(base_vector[i])

        while len(updated_vectors) < base_vector.size(0):
            updated_vectors.append(base_vector[len(updated_vectors)])
        return torch.stack(updated_vectors[:base_vector.size(0)], dim=0)

    def forward(self, hr_token_ids, hr_mask, hr_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,
                use_gnn=True,
                use_multi_hop_path=True,
                use_head_gnn=True,
                use_tail_gnn=True,
                only_ent_embedding=False, **kwargs) -> dict:
        batch_bert_start = time.time()

        if only_ent_embedding:
            result = self.predict_ent_embedding(tail_token_ids=tail_token_ids,
                                                tail_mask=tail_mask,
                                                tail_token_type_ids=tail_token_type_ids)
            self.cache_profiler.record_bert_batch_time(time.time() - batch_bert_start)
            return result

        hr_text_vector = self._encode(self.hr_bert,
                                      token_ids=hr_token_ids,
                                      mask=hr_mask,
                                      token_type_ids=hr_token_type_ids)
        tail_text_vector = self._encode(self.tail_bert,
                                        token_ids=tail_token_ids,
                                        mask=tail_mask,
                                        token_type_ids=tail_token_type_ids)
        head_vector = self._encode(self.tail_bert,
                                   token_ids=head_token_ids,
                                   mask=head_mask,
                                   token_type_ids=head_token_type_ids)
        self.cache_profiler.record_bert_batch_time(time.time() - batch_bert_start)

        if not use_gnn:
            return {'hr_vector': hr_text_vector,
                    'tail_vector': tail_text_vector,
                    'head_vector': head_vector}

        current_batch_size = hr_text_vector.size(0)
        batch_data = self._current_batch_items(kwargs.get('batch_data', []), current_batch_size)
        device, dtype = hr_text_vector.device, hr_text_vector.dtype
        module_mask = self._module_mask_batch(batch_data,
                                              current_batch_size,
                                              device,
                                              dtype)

        relation_context = self._relation_context_batch(batch_data,
                                                        current_batch_size,
                                                        device,
                                                        dtype)
        rule_prior_vec, rule_stats = self._rule_prior_batch(batch_data,
                                                            current_batch_size,
                                                            device,
                                                            dtype)

        if self.args.use_dynamic_relation_memory and batch_data:
            hr_text_vector = F.normalize(
                hr_text_vector + module_mask * self.args.relation_memory_weight * relation_context,
                p=2,
                dim=1)

        if self.args.use_rule_path_prior and batch_data:
            hr_text_vector = F.normalize(
                hr_text_vector + module_mask * self.args.rule_path_prior_weight * rule_prior_vec,
                p=2,
                dim=1)
            relation_context = F.normalize(relation_context + module_mask * rule_prior_vec, p=2, dim=1)

        graph = get_cotail_graph() if self.training else get_cotail_graph_valid()
        cache = get_dynamic_cache()
        hr_one_hop_vector = hr_text_vector
        tail_one_hop_vector = tail_text_vector

        if batch_data and use_head_gnn:
            hr_one_hop_vector = self._apply_batch_gnn(base_vector=hr_text_vector,
                                                      hr_text_vector=hr_text_vector,
                                                      relation_context=relation_context,
                                                      batch_data=batch_data,
                                                      graph=graph,
                                                      cache=cache,
                                                      vector_type='hr',
                                                      gnn_layer=self.hr_gnn,
                                                      module_mask=module_mask)
        if batch_data and use_tail_gnn:
            tail_one_hop_vector = self._apply_batch_gnn(base_vector=tail_text_vector,
                                                        hr_text_vector=hr_text_vector,
                                                        relation_context=relation_context,
                                                        batch_data=batch_data,
                                                        graph=graph,
                                                        cache=cache,
                                                        vector_type='tail',
                                                        gnn_layer=self.tail_gnn,
                                                        module_mask=module_mask)

        hr_vector, hr_gate = self._fuse_text_structure(hr_text_vector,
                                                       hr_one_hop_vector,
                                                       module_mask=module_mask)
        tail_vector, tail_gate = self._fuse_text_structure(tail_text_vector,
                                                           tail_one_hop_vector,
                                                           module_mask=module_mask)

        output = {'hr_vector': hr_vector,
                  'tail_vector': tail_vector,
                  'head_vector': head_vector}

        active = module_mask.view(-1)
        active_count = active.sum().clamp_min(1.0)
        avg_rule_count = (rule_stats['rule_count'].detach() * active).sum() / active_count
        avg_rule_score = (rule_stats['rule_score'].detach() * active).sum() / active_count
        output['msr_stats'] = {
            'avg_selected_path_count': avg_rule_count,
            'avg_selected_path_score': avg_rule_score,
            'lambda_text': hr_text_vector.new_tensor(0.0),
            'lambda_1hop': hr_text_vector.new_tensor(1.0),
            'lambda_path': hr_text_vector.new_tensor(
                self.args.rule_path_prior_weight if self.args.use_rule_path_prior else 0.0),
        }
        return output


class MSRProgKGC(CustomBertModel):
    """Multi-hop Structure Reliability-guided Progressive KGC."""

    def __init__(self, args):
        super().__init__(args)
        self.relation_to_idx = self._build_relation_vocab()
        self.unk_relation_idx = self.relation_to_idx['<unk>']
        self.path_relation_embeddings = nn.Embedding(len(self.relation_to_idx), self.config.hidden_size)

        self.path_mlp = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
        )
        self.hop_attention = nn.Linear(self.config.hidden_size, 1)
        self.hop_reliability_gate = nn.Sequential(
            nn.Linear(self.config.hidden_size * 2 + 3, self.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.config.hidden_size, 1),
        )
        self.scale_concat = nn.Linear(self.config.hidden_size * 3, self.config.hidden_size)
        self.scale_attention = nn.Linear(self.config.hidden_size, 1)
        self.scale_reliability_gate = nn.Sequential(
            nn.Linear(self.config.hidden_size + 5, self.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.config.hidden_size, 1),
        )

    def _build_relation_vocab(self) -> Dict[str, int]:
        relations = {'<unk>'}
        data_dir = os.path.dirname(self.args.train_path)
        relation_path = os.path.join(data_dir, 'relations.json')
        if os.path.exists(relation_path):
            relation_obj = json.load(open(relation_path, 'r', encoding='utf-8'))
            relations.update(str(rel) for rel in relation_obj.values())

        for path in [self.args.train_path, self.args.valid_path]:
            if not path or not os.path.exists(path):
                continue
            for ex in json.load(open(path, 'r', encoding='utf-8')):
                relation = str(ex['relation'])
                relations.add(relation)
                relations.add(self._inverse_relation(relation))

        return {relation: idx for idx, relation in enumerate(sorted(relations))}

    @staticmethod
    def _inverse_relation(relation: str) -> str:
        return relation[len('inverse '):] if relation.startswith('inverse ') else 'inverse {}'.format(relation)

    def _relation_index(self, relation: str, device: torch.device) -> torch.Tensor:
        idx = self.relation_to_idx.get(str(relation), self.unk_relation_idx)
        return torch.LongTensor([idx]).to(device)

    def _relation_embedding(self, relation: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # [1] -> [D]
        rel_idx = self._relation_index(relation, device)
        return self.path_relation_embeddings(rel_idx).squeeze(0).to(dtype=dtype)

    def _path_relation_embeddings(self, relation_seq: List[str], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not relation_seq:
            return torch.zeros(1, self.config.hidden_size, device=device, dtype=dtype)
        rel_indices = [self.relation_to_idx.get(str(rel), self.unk_relation_idx) for rel in relation_seq]
        # [K] -> [K, D]
        rel_tensor = torch.LongTensor(rel_indices).to(device)
        return self.path_relation_embeddings(rel_tensor).to(dtype=dtype)

    def _encode_single_path(self,
                            path: dict,
                            target_relation: str,
                            device: torch.device,
                            dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        relation_seq = path.get('relation_ids') or path.get('relations') or []
        # [K, D] -> [D]
        rel_emb = self._path_relation_embeddings(relation_seq, device, dtype)
        path_vec = rel_emb.mean(dim=0)
        if self.args.path_encoder_type == 'mlp':
            path_vec = self.path_mlp(path_vec)
        path_vec = F.normalize(path_vec, p=2, dim=0)

        target_vec = self._relation_embedding(target_relation, device, dtype)
        target_vec = F.normalize(target_vec, p=2, dim=0)
        # [D], [D] -> scalar
        path_score = F.cosine_similarity(path_vec.unsqueeze(0), target_vec.unsqueeze(0), dim=-1).squeeze(0)
        return path_vec, path_score

    def _encode_paths_for_example(self,
                                  paths: List[dict],
                                  target_relation: str,
                                  device: torch.device,
                                  dtype: torch.dtype) -> Tuple[torch.Tensor, dict]:
        zero = torch.zeros(self.config.hidden_size, device=device, dtype=dtype)
        if not paths:
            return zero, {'path_count': 0.0, 'avg_path_score': 0.0, 'path_mask': 0.0}

        path_vecs, path_scores, path_hops = [], [], []
        for path in paths:
            path_vec, path_score = self._encode_single_path(path, target_relation, device, dtype)
            path_vecs.append(path_vec)
            path_scores.append(path_score)
            path_hops.append(int(path.get('hop', len(path.get('relations', [])))))

        # [P, D], [P]
        path_vec_tensor = torch.stack(path_vecs, dim=0)
        path_score_tensor = torch.stack(path_scores, dim=0)
        path_hop_tensor = torch.LongTensor(path_hops).to(device)

        if self.args.path_mode == 'selected':
            valid_mask = path_score_tensor >= self.args.path_score_threshold
            if valid_mask.any():
                valid_indices = torch.nonzero(valid_mask, as_tuple=False).view(-1)
                topk = min(self.args.path_topk, valid_indices.numel())
                _, topk_offsets = torch.topk(path_score_tensor[valid_indices], k=topk)
                keep_indices = valid_indices[topk_offsets]
                path_vec_tensor = path_vec_tensor[keep_indices]
                path_score_tensor = path_score_tensor[keep_indices]
                path_hop_tensor = path_hop_tensor[keep_indices]
            else:
                return zero, {'path_count': 0.0, 'avg_path_score': 0.0, 'path_mask': 0.0}

        hop_vecs, hop_scores, hop_counts, hop_masks = [], [], [], []
        for hop in self.args.path_hops:
            hop_mask = path_hop_tensor == hop
            if hop_mask.any():
                # [P_h, D] -> [D]
                hop_vecs.append(path_vec_tensor[hop_mask].mean(dim=0))
                hop_scores.append(path_score_tensor[hop_mask].mean())
                hop_counts.append(float(hop_mask.long().sum().item()))
                hop_masks.append(1.0)
            else:
                hop_vecs.append(zero)
                hop_scores.append(torch.tensor(0.0, device=device, dtype=dtype))
                hop_counts.append(0.0)
                hop_masks.append(0.0)

        # [H, D], [H]
        hop_vec_tensor = torch.stack(hop_vecs, dim=0)
        hop_score_tensor = torch.stack(hop_scores, dim=0)
        hop_count_tensor = torch.tensor(hop_counts, device=device, dtype=dtype)
        hop_mask_tensor = torch.tensor(hop_masks, device=device, dtype=dtype)
        valid_hop_mask = hop_mask_tensor > 0

        if not valid_hop_mask.any():
            return zero, {'path_count': 0.0, 'avg_path_score': 0.0, 'path_mask': 0.0}

        if self.args.hop_fusion_type == 'attention':
            # [H, D] -> [H]
            hop_logits = self.hop_attention(hop_vec_tensor).squeeze(-1)
            hop_logits = hop_logits.masked_fill(~valid_hop_mask, -1e4)
            hop_weights = torch.softmax(hop_logits, dim=0)
            # [H], [H, D] -> [D]
            multi_hop_vec = torch.sum(hop_weights.unsqueeze(-1) * hop_vec_tensor, dim=0)
        elif self.args.hop_fusion_type == 'reliability_gate':
            target_vec = self._relation_embedding(target_relation, device, dtype)
            target_vecs = target_vec.unsqueeze(0).expand(hop_vec_tensor.size(0), -1)
            hop_features = torch.stack([hop_count_tensor,
                                        hop_score_tensor,
                                        hop_mask_tensor], dim=-1)
            # [H, 2D + 3] -> [H]
            gate_input = torch.cat([hop_vec_tensor, target_vecs, hop_features], dim=-1)
            hop_logits = self.hop_reliability_gate(gate_input).squeeze(-1)
            hop_logits = hop_logits.masked_fill(~valid_hop_mask, -1e4)
            hop_weights = torch.softmax(hop_logits, dim=0)
            multi_hop_vec = torch.sum(hop_weights.unsqueeze(-1) * hop_vec_tensor, dim=0)
        else:
            multi_hop_vec = hop_vec_tensor[valid_hop_mask].mean(dim=0)

        stats = {
            'path_count': float(path_vec_tensor.size(0)),
            'avg_path_score': float(path_score_tensor.detach().mean().item()),
            'path_mask': 1.0,
        }
        return F.normalize(multi_hop_vec, p=2, dim=0), stats

    def _encode_path_batch(self,
                           paths_batch: List[List[dict]],
                           batch_data: List,
                           text_vector: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        device, dtype = text_vector.device, text_vector.dtype
        path_vecs, path_counts, avg_scores, path_masks = [], [], [], []
        for paths, ex in zip(paths_batch, batch_data):
            # Each example returns [D] plus scalar metadata.
            path_vec, stats = self._encode_paths_for_example(paths,
                                                             ex.relation,
                                                             device,
                                                             dtype)
            path_vecs.append(path_vec)
            path_counts.append(stats['path_count'])
            avg_scores.append(stats['avg_path_score'])
            path_masks.append(stats['path_mask'])

        batch_size = text_vector.size(0)
        while len(path_vecs) < batch_size:
            path_vecs.append(torch.zeros(self.config.hidden_size, device=device, dtype=dtype))
            path_counts.append(0.0)
            avg_scores.append(0.0)
            path_masks.append(0.0)

        # list([D]) -> [B, D]
        path_tensor = torch.stack(path_vecs[:batch_size], dim=0)
        stats = {
            'path_count': torch.tensor(path_counts[:batch_size], device=device, dtype=dtype),
            'avg_path_score': torch.tensor(avg_scores[:batch_size], device=device, dtype=dtype),
            'path_mask': torch.tensor(path_masks[:batch_size], device=device, dtype=dtype),
        }
        return path_tensor, stats

    def _fuse_scales(self,
                     text_vec: torch.Tensor,
                     one_hop_vec: torch.Tensor,
                     path_vec: torch.Tensor,
                     neighbor_counts: List[int],
                     path_stats: dict) -> Tuple[torch.Tensor, dict]:
        # text_vec / one_hop_vec / path_vec: [B, D]
        if self.args.fusion_type == 'text_only':
            stats = {
                'lambda_text': text_vec.new_tensor(1.0),
                'lambda_1hop': text_vec.new_tensor(0.0),
                'lambda_path': text_vec.new_tensor(0.0),
            }
            return text_vec, stats

        batch_size = text_vec.size(0)
        device, dtype = text_vec.device, text_vec.dtype
        path_mask = path_stats['path_mask'].view(batch_size, 1)
        safe_path_vec = torch.where(path_mask.bool(), path_vec, torch.zeros_like(path_vec))

        if self.args.fusion_type == 'average':
            # [B, 3, D]
            stacked = torch.stack([text_vec, one_hop_vec, safe_path_vec], dim=1)
            scale_mask = torch.cat([
                torch.ones(batch_size, 2, device=device, dtype=dtype),
                path_mask.to(dtype=dtype)
            ], dim=1)
            scale_weights = scale_mask / scale_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            fused = (stacked * scale_weights.unsqueeze(-1)).sum(dim=1)
            stats = self._scale_weight_stats(scale_weights)
            return F.normalize(fused, p=2, dim=1), stats

        if self.args.fusion_type == 'concat':
            # [B, 3D] -> [B, D]
            concat_vec = torch.cat([text_vec, one_hop_vec, safe_path_vec], dim=-1)
            stats = {
                'lambda_text': text_vec.new_tensor(1.0 / 3.0),
                'lambda_1hop': text_vec.new_tensor(1.0 / 3.0),
                'lambda_path': path_mask.to(dtype=dtype).mean().detach() / 3.0,
            }
            return F.normalize(self.scale_concat(concat_vec), p=2, dim=1), stats

        scale_vecs = torch.stack([text_vec, one_hop_vec, safe_path_vec], dim=1)  # [B, 3, D]
        scale_mask = torch.cat([
            torch.ones(batch_size, 2, device=device, dtype=torch.bool),
            path_mask.bool()
        ], dim=1)  # [B, 3]

        if self.args.fusion_type == 'attention':
            scale_logits = self.scale_attention(scale_vecs).squeeze(-1)  # [B, 3]
            scale_logits = scale_logits.masked_fill(~scale_mask, -1e4)
        else:
            neighbor_tensor = torch.tensor(neighbor_counts[:batch_size], device=device, dtype=dtype)
            while neighbor_tensor.numel() < batch_size:
                neighbor_tensor = torch.cat([neighbor_tensor,
                                             torch.zeros(1, device=device, dtype=dtype)], dim=0)
            path_count = path_stats['path_count'].view(batch_size)
            avg_path_score = path_stats['avg_path_score'].view(batch_size)
            path_mask_flat = path_stats['path_mask'].view(batch_size)

            # [B, 3, 5]
            text_features = torch.stack([
                torch.zeros(batch_size, device=device, dtype=dtype),
                torch.zeros(batch_size, device=device, dtype=dtype),
                path_count,
                avg_path_score,
                path_mask_flat,
            ], dim=-1)
            hop_features = torch.stack([
                neighbor_tensor,
                neighbor_tensor,
                path_count,
                avg_path_score,
                path_mask_flat,
            ], dim=-1)
            path_features = torch.stack([
                neighbor_tensor,
                neighbor_tensor,
                path_count,
                avg_path_score,
                path_mask_flat,
            ], dim=-1)
            stat_features = torch.stack([text_features, hop_features, path_features], dim=1)
            gate_input = torch.cat([scale_vecs, stat_features], dim=-1)  # [B, 3, D + 5]
            scale_logits = self.scale_reliability_gate(gate_input).squeeze(-1)  # [B, 3]
            scale_logits = scale_logits.masked_fill(~scale_mask, -1e4)

        scale_weights = torch.softmax(scale_logits, dim=1)  # [B, 3]
        fused = torch.sum(scale_weights.unsqueeze(-1) * scale_vecs, dim=1)  # [B, D]
        return F.normalize(fused, p=2, dim=1), self._scale_weight_stats(scale_weights)

    @staticmethod
    def _scale_weight_stats(scale_weights: torch.Tensor) -> dict:
        # scale_weights: [B, 3] for text, 1-hop and path scales.
        detached = scale_weights.detach()
        return {
            'lambda_text': detached[:, 0].mean(),
            'lambda_1hop': detached[:, 1].mean(),
            'lambda_path': detached[:, 2].mean(),
        }

    @staticmethod
    def _current_batch_items(full_items, current_batch_size):
        if not full_items:
            return []
        if torch.cuda.device_count() > 1 and len(full_items) > current_batch_size:
            device_idx = torch.cuda.current_device()
            start_idx = device_idx * current_batch_size
            end_idx = start_idx + current_batch_size
            return full_items[start_idx:end_idx]
        return full_items[:current_batch_size]

    def _apply_one_hop_gnn(self,
                           base_vector: torch.Tensor,
                           batch_data: List,
                           graph,
                           cache,
                           vector_type: str,
                           gnn_layer: nn.Module) -> Tuple[torch.Tensor, List[int]]:
        updated_vectors, neighbor_counts = [], []
        for i, ex in enumerate(batch_data):
            entity_id = ex.head_id if vector_type == 'hr' else ex.tail_id
            neighbors = self._get_neighbors_with_profiling(graph, entity_id)
            neighbor_counts.append(len(neighbors))
            neighbor_vectors = self._get_cached_vectors_with_profiling(cache, neighbors, vector_type)

            if neighbor_vectors:
                # [K, D]
                neighbor_tensor = torch.stack(neighbor_vectors).to(base_vector.device)
                updated = self._apply_gnn_with_profiling(gnn_layer, base_vector[i], neighbor_tensor)
                updated_vectors.append(updated)
            else:
                updated_vectors.append(base_vector[i])

        while len(updated_vectors) < base_vector.size(0):
            updated_vectors.append(base_vector[len(updated_vectors)])
            neighbor_counts.append(0)

        # list([D]) -> [B, D]
        return torch.stack(updated_vectors[:base_vector.size(0)], dim=0), neighbor_counts

    def _compute_msc_loss(self,
                          one_hop_vec: torch.Tensor,
                          path_vec: torch.Tensor,
                          path_stats: dict) -> torch.Tensor:
        if not (self.args.use_multi_hop_path and self.args.use_msc_loss):
            return one_hop_vec.new_tensor(0.0)
        mask = path_stats['path_mask'].bool()  # [B]
        if not mask.any():
            return one_hop_vec.new_tensor(0.0)
        # [B_valid, D], [B_valid, D] -> scalar
        cosine = F.cosine_similarity(one_hop_vec[mask], path_vec[mask], dim=-1)
        return 1.0 - cosine.mean()

    def forward(self, hr_token_ids, hr_mask, hr_token_type_ids,
                tail_token_ids, tail_mask, tail_token_type_ids,
                head_token_ids, head_mask, head_token_type_ids,
                use_gnn=True,
                use_multi_hop_path=True,
                use_head_gnn=True,
                use_tail_gnn=True,
                only_ent_embedding=False, **kwargs) -> dict:
        batch_bert_start = time.time()

        if only_ent_embedding:
            result = self.predict_ent_embedding(tail_token_ids=tail_token_ids,
                                                tail_mask=tail_mask,
                                                tail_token_type_ids=tail_token_type_ids)
            self.cache_profiler.record_bert_batch_time(time.time() - batch_bert_start)
            return result

        # [B, D]
        hr_text_vector = self._encode(self.hr_bert,
                                      token_ids=hr_token_ids,
                                      mask=hr_mask,
                                      token_type_ids=hr_token_type_ids)
        tail_text_vector = self._encode(self.tail_bert,
                                        token_ids=tail_token_ids,
                                        mask=tail_mask,
                                        token_type_ids=tail_token_type_ids)
        head_vector = self._encode(self.tail_bert,
                                   token_ids=head_token_ids,
                                   mask=head_mask,
                                   token_type_ids=head_token_type_ids)
        self.cache_profiler.record_bert_batch_time(time.time() - batch_bert_start)

        hr_one_hop_vector = hr_text_vector
        tail_one_hop_vector = tail_text_vector
        hr_neighbor_counts = [0 for _ in range(hr_text_vector.size(0))]
        tail_neighbor_counts = [0 for _ in range(tail_text_vector.size(0))]

        current_batch_size = hr_text_vector.size(0)
        full_batch_data = kwargs.get('batch_data', [])
        batch_data = self._current_batch_items(full_batch_data, current_batch_size)

        if use_gnn:
            cache = get_dynamic_cache()
            graph = get_cotail_graph_valid() if not self.training else get_cotail_graph()
            if use_head_gnn:
                hr_one_hop_vector, hr_neighbor_counts = self._apply_one_hop_gnn(
                    hr_text_vector, batch_data, graph, cache, 'hr', self.hr_gnn)
            if use_tail_gnn:
                tail_one_hop_vector, tail_neighbor_counts = self._apply_one_hop_gnn(
                    tail_text_vector, batch_data, graph, cache, 'tail', self.tail_gnn)

        hr_final_vector = hr_one_hop_vector
        tail_final_vector = tail_one_hop_vector
        msc_losses = []
        msr_stats = {
            'avg_selected_path_count': hr_text_vector.new_tensor(0.0),
            'avg_selected_path_score': hr_text_vector.new_tensor(0.0),
            'lambda_text': hr_text_vector.new_tensor(0.0 if use_gnn else 1.0),
            'lambda_1hop': hr_text_vector.new_tensor(1.0 if use_gnn else 0.0),
            'lambda_path': hr_text_vector.new_tensor(0.0),
        }

        if self.args.use_multi_hop_path and use_multi_hop_path and batch_data:
            head_paths = self._current_batch_items(kwargs.get('head_multi_hop_paths', []), current_batch_size)
            tail_paths = self._current_batch_items(kwargs.get('tail_multi_hop_paths', []), current_batch_size)

            # [B, D], scalar stats [B]
            hr_path_vector, hr_path_stats = self._encode_path_batch(head_paths, batch_data, hr_text_vector)
            tail_path_vector, tail_path_stats = self._encode_path_batch(tail_paths, batch_data, tail_text_vector)

            hr_final_vector, hr_gate_stats = self._fuse_scales(hr_text_vector,
                                                               hr_one_hop_vector,
                                                               hr_path_vector,
                                                               hr_neighbor_counts,
                                                               hr_path_stats)
            tail_final_vector, tail_gate_stats = self._fuse_scales(tail_text_vector,
                                                                   tail_one_hop_vector,
                                                                   tail_path_vector,
                                                                   tail_neighbor_counts,
                                                                   tail_path_stats)
            msc_losses.append(self._compute_msc_loss(hr_one_hop_vector, hr_path_vector, hr_path_stats))
            msc_losses.append(self._compute_msc_loss(tail_one_hop_vector, tail_path_vector, tail_path_stats))
            avg_count = torch.cat([hr_path_stats['path_count'],
                                   tail_path_stats['path_count']], dim=0).detach().mean()
            avg_score = torch.cat([hr_path_stats['avg_path_score'],
                                   tail_path_stats['avg_path_score']], dim=0).detach().mean()
            msr_stats = {
                'avg_selected_path_count': avg_count,
                'avg_selected_path_score': avg_score,
                'lambda_text': (hr_gate_stats['lambda_text'] + tail_gate_stats['lambda_text']) / 2.0,
                'lambda_1hop': (hr_gate_stats['lambda_1hop'] + tail_gate_stats['lambda_1hop']) / 2.0,
                'lambda_path': (hr_gate_stats['lambda_path'] + tail_gate_stats['lambda_path']) / 2.0,
            }

        hr_final_vector = F.normalize(hr_final_vector, p=2, dim=1)
        tail_final_vector = F.normalize(tail_final_vector, p=2, dim=1)
        output = {
            'hr_vector': hr_final_vector,
            'tail_vector': tail_final_vector,
            'head_vector': head_vector,
            'msr_stats': msr_stats,
        }
        if msc_losses:
            output['msc_loss'] = torch.stack(msc_losses).mean()
        return output


def _pool_output(pooling: str,
                 cls_output: torch.tensor,
                 mask: torch.tensor,
                 last_hidden_state: torch.tensor) -> torch.tensor:
    if pooling == 'cls':
        output_vector = cls_output
    elif pooling == 'max':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).long()
        last_hidden_state[input_mask_expanded == 0] = -1e4
        output_vector = torch.max(last_hidden_state, 1)[0]
    elif pooling == 'mean':
        input_mask_expanded = mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-4)
        output_vector = sum_embeddings / sum_mask
    else:
        assert False, 'Unknown pooling mode: {}'.format(pooling)

    output_vector = nn.functional.normalize(output_vector, dim=1)
    return output_vector
