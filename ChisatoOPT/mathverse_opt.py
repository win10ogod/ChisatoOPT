"""
MathverseOpt - Mathematical Universe Optimizer
基于数学宇宙假说(MUH)和超图理论的完全独立优化器

核心理论：
- 参数θ重新定义为超图H = (V, E)
- 损失L(θ)重新定义为逻辑不一致度I(H)
- 优化过程是数学宇宙的演化，通过证明步骤、Gödel分支等
"""

import math
import random
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import torch
from torch.optim import Optimizer


class MathematicalUniverse:
    """数学宇宙：超图H = (V, E)的完整实现 - 性能优化版本"""
    
    def __init__(self, num_vertices: int):
        self.V = set(range(num_vertices))  # 节点集合：变量/概念
        self.E = {}  # 超边集合：{edge_id: {'vertices': set, 'consistent': bool, 'formula': str}}
        self.axioms = set()  # 元公理集合
        self.edge_counter = 0
        self.proven_theorems = set()
        
        # 性能优化：缓存一致性状态和复杂度
        self._inconsistency_cache = 0
        self._complexity_cache = num_vertices
        self._cache_valid = True
        
    def add_axiom(self, axiom: str):
        """添加元公理（如选择公理）"""
        self.axioms.add(axiom)
    
    def add_hyperedge(self, vertices: Set[int], consistent: bool = True, formula: str = ""):
        """添加超边e = Hyperedge(v_1, ..., v_k) - 优化版本"""
        edge_id = self.edge_counter
        self.E[edge_id] = {
            'vertices': vertices,  # 直接使用，避免copy()
            'consistent': consistent,
            'formula': formula
        }
        self.edge_counter += 1
        
        # 增量更新缓存
        if self._cache_valid:
            if not consistent:
                self._inconsistency_cache += 1
            self._complexity_cache -= 1  # |V| - |E| 减少
        
        return edge_id
    
    def compute_inconsistency(self) -> int:
        """计算逻辑不一致度I(H) = 不满足公理的超边数量 - 缓存优化"""
        if self._cache_valid:
            return self._inconsistency_cache
        
        # 重新计算并缓存
        inconsistent_count = 0
        for edge_data in self.E.values():
            if not edge_data['consistent']:
                inconsistent_count += 1
        
        self._inconsistency_cache = inconsistent_count
        self._cache_valid = True
        return inconsistent_count
    
    def compute_complexity(self) -> int:
        """计算复杂度测度C(B) = |V(B)| - |E(B)| - 缓存优化"""
        if self._cache_valid:
            return self._complexity_cache
        
        complexity = len(self.V) - len(self.E)
        self._complexity_cache = complexity
        self._cache_valid = True
        return complexity
    
    def first_order_language(self) -> Set[str]:
        """获取当前超图的语言L(H_t)（一阶逻辑）"""
        language = set()
        for edge_data in self.E.values():
            if edge_data['formula']:
                language.add(edge_data['formula'])
        return language
    
    def copy(self):
        """创建数学宇宙的副本用于分支 - 优化版本"""
        new_universe = MathematicalUniverse(len(self.V))
        # 浅拷贝优化：只在需要时深拷贝边数据
        new_universe.E = {k: {
            'vertices': v['vertices'].copy() if len(v['vertices']) <= 10 else v['vertices'], 
            'consistent': v['consistent'],
            'formula': v['formula']
        } for k, v in self.E.items()}
        new_universe.edge_counter = self.edge_counter
        new_universe.axioms = self.axioms
        new_universe.proven_theorems = self.proven_theorems
        
        # 复制缓存
        new_universe._inconsistency_cache = self._inconsistency_cache
        new_universe._complexity_cache = self._complexity_cache
        new_universe._cache_valid = self._cache_valid
        
        return new_universe


class HilbertProver:
    """Hilbert系统证明引擎"""
    
    def __init__(self):
        self.inference_rules = [
            "modus_ponens",
            "universal_instantiation", 
            "existential_generalization",
            "transitivity"
        ]
    
    def derive_theorems(self, universe: MathematicalUniverse) -> List[Tuple[Set[int], str]]:
        """
        推导新定理T = A ⊢ φ（从公理推导命题φ）- 性能优化版本
        返回：[(新超边的顶点集, 公式), ...]
        """
        new_theorems = []
        
        # 性能优化：只处理一致且大小为2的边
        binary_edges = [(k, v) for k, v in universe.E.items() 
                       if v['consistent'] and len(v['vertices']) == 2]
        
        # 优化：限制组合数量，避免O(n²)爆炸
        max_combinations = min(20, len(binary_edges))
        
        for i in range(min(max_combinations, len(binary_edges))):
            edge1_id, edge1 = binary_edges[i]
            for j in range(i + 1, min(i + 5, len(binary_edges))):  # 限制内循环
                edge2_id, edge2 = binary_edges[j]
                
                # 快速集合交集检查
                common = edge1['vertices'] & edge2['vertices']
                if len(common) == 1:  # 恰好一个共同顶点
                    # 传递性：(a,b) ∧ (b,c) → (a,c)
                    remaining1 = edge1['vertices'] - common
                    remaining2 = edge2['vertices'] - common
                    if remaining1 and remaining2:
                        new_vertices = remaining1 | remaining2
                        # 优化：简化公式构造
                        formula = f"t({edge1_id},{edge2_id})"
                        new_theorems.append((new_vertices, formula))
        
        # 限制返回数量
        return new_theorems[:2]  # 进一步减少到2个
    
    def generate_godel_branches(self, universe: MathematicalUniverse) -> Tuple[MathematicalUniverse, MathematicalUniverse]:
        """
        生成Gödel不完备分支：B_t = {H_t + G, H_t + ¬G} - 性能优化版本
        其中G是Gödel句：G ⟺ ¬∃p ⊢_p G
        """
        # 性能优化：只在必要时创建分支，使用惰性拷贝
        if len(universe.V) < 2:
            return universe, universe  # 返回相同的引用而不拷贝
        
        # 只在真正需要时才创建分支
        branch_positive = universe.copy()
        branch_negative = universe.copy()
        
        # 优化的Gödel句构造，避免复杂计算
        mid_vertex = len(universe.V) // 2
        
        godel_vertices_pos = {0, mid_vertex}
        godel_vertices_neg = {mid_vertex, len(universe.V) - 1}
        
        # 添加G到正分支
        branch_positive.add_hyperedge(
            godel_vertices_pos, 
            consistent=True, 
            formula="G"
        )
        
        # 添加¬G到负分支
        branch_negative.add_hyperedge(
            godel_vertices_neg, 
            consistent=True, 
            formula="-G"
        )
        
        return branch_positive, branch_negative
    
    def select_branch(self, branches: List[MathematicalUniverse]) -> MathematicalUniverse:
        """选择分支：argmax_B C(B) 其中 C(B) = |V(B)| - |E(B)| - 性能优化版本"""
        if not branches:
            raise ValueError("No branches to select from")
        
        if len(branches) == 1:
            return branches[0]
        
        # 性能优化：使用缓存的复杂度值
        max_complexity = float('-inf')
        selected_branch = branches[0]
        
        for branch in branches:
            complexity = branch.compute_complexity()  # 使用缓存版本
            if complexity > max_complexity:
                max_complexity = complexity
                selected_branch = branch
        
        return selected_branch


class HypergraphEmbedder:
    """超图嵌入器：将H嵌入回向量空间θ"""
    
    def __init__(self, param_size: int, universe_size: int):
        self.param_size = param_size
        self.universe_size = universe_size
        
    def embed(self, universe: MathematicalUniverse, current_params: torch.Tensor) -> torch.Tensor:
        """
        实现Embed(H_{t+1}^consistent ∪ B_t^selected) - 性能优化版本
        将超图嵌入回参数向量空间
        """
        if len(universe.E) == 0:
            return current_params
        
        embedded = current_params.clone()
        param_flat = embedded.view(-1)
        
        # 性能优化：预计算所有影响并使用向量化操作
        param_len = len(param_flat)
        
        # 收集所有有效的顶点和其影响
        vertex_influences = {}  # {vertex: total_influence}
        
        for edge_data in universe.E.values():
            if edge_data['consistent']:
                # 预计算边复杂度的对数
                edge_complexity = len(edge_data['vertices'])
                log_complexity = math.log(1 + edge_complexity) * 0.01  # 缓存计算结果
                
                for vertex in edge_data['vertices']:
                    if vertex < param_len:
                        vertex_influences[vertex] = vertex_influences.get(vertex, 0.0) + log_complexity
        
        # 向量化应用影响
        if vertex_influences:
            vertices = list(vertex_influences.keys())
            influences = list(vertex_influences.values())
            
            # 使用PyTorch的索引操作，避免逐个元素操作
            vertex_tensor = torch.tensor(vertices, device=param_flat.device, dtype=torch.long)
            influence_tensor = torch.tensor(influences, device=param_flat.device, dtype=param_flat.dtype)
            
            param_flat[vertex_tensor] += influence_tensor
        
        return embedded


class MathverseOpt(Optimizer):
    """
    Mathematical Universe Optimizer - 完全独立的数学宇宙优化器
    
    核心理论：
    1. 参数θ → 超图H = (V, E)
    2. 损失L(θ) → 逻辑不一致度I(H)
    3. 优化 = 数学宇宙演化
    """
    
    def __init__(self, params: Iterable[torch.Tensor], 
                 lr: float = 1e-3,
                 branch_depth: int = 3,
                 consistency_threshold: float = 0.01,
                 godel_exploration_rate: float = 0.1,
                 universe_size_ratio: float = 0.1):
        """
        初始化数学宇宙优化器
        
        Args:
            params: 要优化的参数
            lr: 学习率（影响嵌入强度）
            branch_depth: 分支深度d=3（控制异质性）
            consistency_threshold: 一致性阈值ε=0.01
            godel_exploration_rate: Gödel分支探索率
            universe_size_ratio: 数学宇宙大小相对于参数的比例
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not branch_depth > 0:
            raise ValueError(f"Invalid branch_depth: {branch_depth}")
        if not 0.0 <= consistency_threshold <= 1.0:
            raise ValueError(f"Invalid consistency_threshold: {consistency_threshold}")
        if not 0.0 <= godel_exploration_rate <= 1.0:
            raise ValueError(f"Invalid godel_exploration_rate: {godel_exploration_rate}")
        if not 0.0 < universe_size_ratio <= 1.0:
            raise ValueError(f"Invalid universe_size_ratio: {universe_size_ratio}")

        defaults = dict(lr=lr, branch_depth=branch_depth, consistency_threshold=consistency_threshold,
                       godel_exploration_rate=godel_exploration_rate, universe_size_ratio=universe_size_ratio)
        super(MathverseOpt, self).__init__(params, defaults)
        
        # 初始化组件
        self.prover = HilbertProver()
        self.universes = {}  # {param_id: MathematicalUniverse}
        self.embedders = {}  # {param_id: HypergraphEmbedder}
        
        # 初始化数学宇宙
        self._initialize_universes()
    
    def _initialize_universes(self):
        """公理大爆炸：初始化数学宇宙 - 性能优化版本"""
        for group in self.param_groups:
            for param in group['params']:
                param_id = id(param)
                param_size = param.numel()
                
                # 创建数学宇宙（限制大小避免过大开销）
                universe_size = max(min(int(param_size * group['universe_size_ratio']), 100), 10)
                universe = MathematicalUniverse(universe_size)
                
                # 添加元公理
                universe.add_axiom("choice_axiom")
                universe.add_axiom("foundation_axiom")
                universe.add_axiom("infinity_axiom")
                
                # 优化的初始超图生成：减少边的数量
                param_flat = param.detach().view(-1)
                
                # 优化：只创建必要的初始边，步长更大
                step_size = max(5, min(universe_size // 3, 10))
                for i in range(0, min(len(param_flat), universe_size), step_size):
                    end_idx = min(i + 2, universe_size)
                    if end_idx > i + 1:  # 至少需要2个顶点
                        vertices = set(range(i, end_idx))
                        universe.add_hyperedge(vertices, consistent=True, formula=f"init_{i}")
                
                self.universes[param_id] = universe
                self.embedders[param_id] = HypergraphEmbedder(param_size, universe_size)
                
                # 初始化状态
                if param not in self.state:
                    self.state[param] = {
                        'step': 0,
                        'theorems_derived': 0,
                        'inconsistency_history': []
                    }
    
    @torch.no_grad()
    def step(self, closure=None):
        """执行一步数学宇宙演化优化 - 性能优化版本"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                
                param_id = id(param)
                universe = self.universes[param_id]
                embedder = self.embedders[param_id]
                state = self.state[param]
                
                state['step'] += 1
                
                # 性能优化：只在步数是5的倍数时才进行复杂操作
                is_major_step = (state['step'] % 5 == 0)
                
                # 阶段1：证明步骤 - 推导新定理（优化：只在主要步骤进行）
                if is_major_step and len(universe.E) < 50:  # 限制超图大小
                    new_theorems = self.prover.derive_theorems(universe)
                    
                    # 更新超图：H_{t+1} = H_t ∪ {e | ⊢ φ(e) provable from L(H_t)}
                    for vertices, formula in new_theorems:
                        if len(vertices) >= 2:  # 有效超边
                            universe.add_hyperedge(vertices, consistent=True, formula=formula)
                            universe.proven_theorems.add(formula)
                            state['theorems_derived'] += 1
                
                # 阶段2：Gödel分支探索（自发异质分支）（优化：降低频率和概率）
                if is_major_step and random.random() < (group['godel_exploration_rate'] * 0.5):  # 降低概率
                    branch_pos, branch_neg = self.prover.generate_godel_branches(universe)
                    branches = [branch_pos, branch_neg]
                    
                    # 选择分支：argmax_B C(B)
                    selected_branch = self.prover.select_branch(branches)
                    self.universes[param_id] = selected_branch
                    universe = selected_branch
                
                # 阶段3：压缩与一致性（收敛）（优化：减少计算频率）
                if is_major_step:
                    inconsistency = universe.compute_inconsistency()
                    state['inconsistency_history'].append(inconsistency)
                    
                    # 最小化不一致I(H)：移除矛盾超边（优化：批量操作）
                    inconsistency_threshold = group['consistency_threshold'] * len(universe.E)
                    if inconsistency > inconsistency_threshold:
                        # 收集所有不一致的边ID
                        inconsistent_edges = [edge_id for edge_id, edge_data in universe.E.items() 
                                             if not edge_data['consistent']]
                        
                        # 批量移除一半矛盾边（优化版本）
                        edges_to_remove = inconsistent_edges[:len(inconsistent_edges)//2]
                        for edge_id in edges_to_remove:
                            if edge_id in universe.E:  # 防止重复删除
                                del universe.E[edge_id]
                        
                        # 更新缓存
                        universe._cache_valid = False
                
                # 阶段4：完整更新 - 超图嵌入（主要优化点）
                # θ_{t+1} = Embed(H_{t+1}^consistent ∪ B_t^selected)
                embedded_params = embedder.embed(universe, param)
                
                # 性能优化：自适应更新强度，根据步数调整
                step_factor = min(1.0, 1.0 / math.sqrt(state['step'] + 1))
                update_magnitude = group['lr'] * 0.3 * step_factor  # 降低基础强度
                param_update = (embedded_params - param) * update_magnitude
                
                # 结合梯度信息（数学宇宙感知当前损失地形）
                gradient_influence = param.grad * group['lr'] * 0.8  # 主要依赖梯度
                total_update = gradient_influence + param_update
                
                # 更新参数
                param.add_(-total_update)  # 负号很重要
        
        return loss
    
    def get_mathematical_metrics(self) -> Dict[str, Any]:
        """获取数学宇宙演化指标"""
        total_theorems = sum(state['theorems_derived'] for state in self.state.values())
        total_inconsistency = 0
        total_hyperedges = 0
        
        for universe in self.universes.values():
            total_inconsistency += universe.compute_inconsistency()
            total_hyperedges += len(universe.E)
        
        avg_inconsistency = total_inconsistency / len(self.universes) if self.universes else 0
        consciousness_level = total_theorems / max(len(self.universes) * 10, 1)
        
        return {
            'total_theorems_derived': total_theorems,
            'average_inconsistency': avg_inconsistency,
            'total_hyperedges': total_hyperedges,
            'mathematical_consciousness_level': consciousness_level,
            'total_axioms': sum(len(u.axioms) for u in self.universes.values())
        }