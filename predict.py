import os
import json
import tqdm
import torch
import torch.utils.data

from typing import List
from collections import OrderedDict

from doc import collate, Example, Dataset
from config import args
from models import build_model
from utils import AttrDict, move_to_cuda
from dict_hub import build_tokenizer, get_dynamic_cache
from logger_config import logger


class BertPredictor:

    def __init__(self):
        self.model = None
        self.train_args = AttrDict()
        self.use_cuda = False
        self.device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    def load(self, ckt_path, use_data_parallel=True):  # 默认开启数据并行
        assert os.path.exists(ckt_path)
        ckt_dict = torch.load(ckt_path, map_location=lambda storage, loc: storage)
        self.train_args.__dict__ = ckt_dict['args']
        self._setup_args()
        build_tokenizer(self.train_args)
        self.model = build_model(self.train_args)

        # DataParallel will introduce 'module.' prefix
        state_dict = ckt_dict['state_dict']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                k = k[len('module.'):]
            new_state_dict[k] = v
        self.model.load_state_dict(new_state_dict, strict=True)
        self.model.eval()

        # 优先使用多GPU
        if torch.cuda.is_available():
            if use_data_parallel and self.device_count > 1:
                logger.info(f'使用 {self.device_count} 张GPU进行数据并行处理')
                self.model = torch.nn.DataParallel(self.model)
                self.model.cuda()
                self.use_cuda = True
            else:
                logger.info('使用单GPU')
                self.model.cuda()
                self.use_cuda = True
        logger.info('Load model from {} successfully'.format(ckt_path))

    def _setup_args(self):
        for k, v in args.__dict__.items():
            if k not in self.train_args.__dict__:
                logger.info('Set default attribute: {}={}'.format(k, v))
                self.train_args.__dict__[k] = v
        logger.info(
            'Args used in training: {}'.format(json.dumps(self.train_args.__dict__, ensure_ascii=False, indent=4)))
        args.model_name = self.train_args.model_name
        args.use_link_graph = self.train_args.use_link_graph
        args.structure_start_epoch = self.train_args.structure_start_epoch
        args.multi_hop_start_epoch = self.train_args.multi_hop_start_epoch
        args.use_multi_hop_path = self.train_args.use_multi_hop_path
        args.path_hops = self.train_args.path_hops
        args.max_path_hop = self.train_args.max_path_hop
        args.max_paths_per_entity = self.train_args.max_paths_per_entity
        args.path_topk = self.train_args.path_topk
        args.path_score_threshold = self.train_args.path_score_threshold
        args.path_mode = self.train_args.path_mode
        args.path_encoder_type = self.train_args.path_encoder_type
        args.hop_fusion_type = self.train_args.hop_fusion_type
        args.fusion_type = self.train_args.fusion_type
        args.use_msc_loss = self.train_args.use_msc_loss
        args.alpha_msc = self.train_args.alpha_msc
        args.is_test = True

    def _get_optimal_batch_size(self, base_batch_size):
        """根据GPU数量调整batch size"""
        if self.device_count > 1:
            # 多GPU时可以使用更大的batch size
            return base_batch_size * self.device_count
        return base_batch_size

    @torch.no_grad()
    def predict_by_examples(self, examples: List[Example]):
        # 根据GPU数量调整batch size
        base_batch_size = 64  # 单GPU的基础batch size
        inference_batch_size = self._get_optimal_batch_size(base_batch_size)

        logger.info(f"使用batch_size={inference_batch_size} (GPU数量: {self.device_count})")

        data_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=examples, task=args.task, args=args),
            num_workers=min(8, self.device_count * 2),  # 根据GPU数量调整worker数
            batch_size=inference_batch_size,
            collate_fn=collate,
            shuffle=False,
            pin_memory=True)  # 启用pin_memory加速数据传输

        hr_tensor_list, tail_tensor_list = [], []
        for idx, batch_dict in enumerate(data_loader):
            if self.use_cuda:
                batch_dict = move_to_cuda(batch_dict)

            # 添加内存监控
            if idx % 10 == 0:
                self._print_gpu_memory(f"predict_by_examples Batch {idx}")

            try:
                outputs = self.model(**batch_dict, use_gnn=True)
                hr_tensor_list.append(outputs['hr_vector'].cpu())  # 立即移到CPU释放GPU内存
                tail_tensor_list.append(outputs['tail_vector'].cpu())

                # 清理中间变量
                del outputs

            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"内存不足在batch {idx}, 当前batch size: {len(batch_dict['batch_data'])}")
                self._print_gpu_memory("OOM时")
                torch.cuda.empty_cache()
                raise e

            # 定期清理内存
            if idx % 10 == 0:
                torch.cuda.empty_cache()

        # 将结果tensor移回GPU（如果需要的话）
        hr_result = torch.cat(hr_tensor_list, dim=0)
        tail_result = torch.cat(tail_tensor_list, dim=0)

        if self.use_cuda:
            hr_result = hr_result.cuda()
            tail_result = tail_result.cuda()

        return hr_result, tail_result

    @torch.no_grad()
    def predict_by_entities(self, entity_exs) -> torch.tensor:
        examples = []
        for entity_ex in entity_exs:
            examples.append(Example(head_id='', relation='',
                                    tail_id=entity_ex.entity_id))

        # 根据GPU数量调整batch size
        base_batch_size = 128  # 单GPU的基础batch size
        entity_batch_size = self._get_optimal_batch_size(base_batch_size)

        logger.info(f"Entity预测使用batch_size={entity_batch_size}")

        data_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=examples, task=args.task, args=args),
            num_workers=min(8, self.device_count * 2),
            batch_size=entity_batch_size,
            collate_fn=collate,
            shuffle=False,
            pin_memory=True)

        ent_tensor_list = []
        for idx, batch_dict in enumerate(tqdm.tqdm(data_loader, desc="预测实体embeddings")):
            batch_dict['only_ent_embedding'] = True
            if self.use_cuda:
                batch_dict = move_to_cuda(batch_dict)

            try:
                outputs = self.model(**batch_dict, use_gnn=True)
                ent_tensor_list.append(outputs['ent_vectors'].cpu())  # 立即移到CPU

                # 清理中间变量
                del outputs

            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"Entity预测内存不足在batch {idx}")
                self._print_gpu_memory("Entity OOM时")
                torch.cuda.empty_cache()
                raise e

            # 定期清理内存
            if idx % 10 == 0:
                torch.cuda.empty_cache()

        # 将结果移回GPU
        result = torch.cat(ent_tensor_list, dim=0)
        if self.use_cuda:
            result = result.cuda()

        return result

    def init_Dynamic_cache(self, examples: List[Example]):
        # 根据GPU数量调整batch size
        base_batch_size = 32  # 缓存初始化使用较小的基础batch size
        cache_batch_size = self._get_optimal_batch_size(base_batch_size)

        data_loader = torch.utils.data.DataLoader(
            Dataset(path='', examples=examples, task=args.task, args=args),
            num_workers=min(4, self.device_count),
            batch_size=cache_batch_size,
            collate_fn=collate,
            shuffle=False,
            pin_memory=True)

        logger.info(f"初始化动态缓存，共{len(examples)}个样本，batch_size={cache_batch_size}")

        for idx, batch_dict in enumerate(tqdm.tqdm(data_loader, desc="初始化动态缓存")):
            if self.use_cuda:
                batch_dict = move_to_cuda(batch_dict)

            # 添加内存监控
            if idx % 5 == 0:
                self._print_gpu_memory(f"缓存初始化 Batch {idx}/{len(data_loader)}")

            try:
                outputs = self.model(**batch_dict, use_gnn=False)
                batch_data = batch_dict['batch_data']
                head_ids = [ex.head_id for ex in batch_data]
                tail_ids = [ex.tail_id for ex in batch_data]
                hr_vectors = outputs['hr_vector']
                tail_vectors = outputs['tail_vector']

                cache = get_dynamic_cache()
                cache.update_hr(head_ids, hr_vectors)
                cache.update_tail(tail_ids, tail_vectors)

                # 清理中间变量
                del outputs

            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"缓存初始化内存不足在batch {idx}")
                logger.error(f"当前batch大小: {len(batch_dict['batch_data'])}")
                self._print_gpu_memory("缓存初始化OOM时")
                torch.cuda.empty_cache()
                raise e

            # 每处理几个batch就清理一次内存
            if idx % 5 == 0:
                torch.cuda.empty_cache()

    def _print_gpu_memory(self, stage=""):
        """打印所有GPU的内存使用情况"""
        if torch.cuda.is_available():
            for i in range(self.device_count):
                allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
                reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
                logger.info(f"{stage} - GPU {i}: 已分配={allocated:.2f}GB, 已缓存={reserved:.2f}GB")
