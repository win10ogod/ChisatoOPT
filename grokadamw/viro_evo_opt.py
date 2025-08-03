"""
ViroEvoOpt - Viral Evolutionary Optimizer
基于病毒学、流行病学和进化动力学的完全独立优化器

核心理论：
- 参数θ视为病毒种群P(θ)，分为S(易感)、I(感染)、R(恢复)三类
- 使用SIR微分方程组描述种群动态
- Gillespie算法模拟随机变异和传播事件
- Moran过程进行进化选择
- 参数更新：θ_I ← θ_I - η∇L(θ_I) + ξ（变异项）
"""

import torch
import math
import numpy as np
import random
from typing import Iterable, Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ViralPopulation:
    """病毒种群：实现SIR模型的参数分类"""
    
    def __init__(self, total_params: int, beta: float, gamma: float, mu: float, alpha: float):
        """
        初始化病毒种群
        
        Args:
            total_params: 总参数数量N
            beta: 感染率β (e.g., 0.1)
            gamma: 恢复率γ (e.g., 0.05)  
            mu: 变异率μ (e.g., 1e-4, HIV突变率)
            alpha: 损失到死亡率的缩放α
        """
        self.N = total_params  # 总种群N = S + I + R
        self.beta = beta
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha
        
        # 种群初始化：θ_0分类（设计：S_0 = |θ|, I_0 = 0, R_0 = 0）
        self.S = float(total_params)  # Susceptible 易感参数
        self.I = 0.0                  # Infected 感染参数  
        self.R = 0.0                  # Recovered 恢复参数
        
        # 参数索引分类
        self.susceptible_indices = set(range(total_params))
        self.infected_indices = set()
        self.recovered_indices = set()
    
    def compute_death_rate(self, loss_value: float) -> float:
        """计算死亡率δ = αL(θ)"""
        return self.alpha * loss_value
    
    def basic_reproduction_number(self, loss_value: float) -> float:
        """计算基础繁殖数R_0 = β / (γ + δ)"""
        delta = self.compute_death_rate(loss_value)
        return self.beta / (self.gamma + delta + 1e-8)
    
    def sir_dynamics(self, dt: float, loss_value: float) -> Tuple[float, float, float]:
        """
        SIR微分方程组（Kermack-McKendrick 1927）：
        dS/dt = -β(SI/N) + μR
        dI/dt = β(SI/N) - γI - δI  
        dR/dt = γI - μR
        """
        delta = self.compute_death_rate(loss_value)
        
        if self.N <= 0:
            return 0.0, 0.0, 0.0
        
        # SIR微分方程组
        dS_dt = -self.beta * (self.S * self.I) / self.N + self.mu * self.R
        dI_dt = self.beta * (self.S * self.I) / self.N - self.gamma * self.I - delta * self.I
        dR_dt = self.gamma * self.I - self.mu * self.R
        
        # 应用时间步长
        dS = dS_dt * dt
        dI = dI_dt * dt
        dR = dR_dt * dt
        
        # 更新种群（保持非负性）
        self.S = max(0.0, self.S + dS)
        self.I = max(0.0, self.I + dI)
        self.R = max(0.0, self.R + dR)
        
        # 归一化保持总数
        total = self.S + self.I + self.R
        if total > 0:
            self.S = self.S * self.N / total
            self.I = self.I * self.N / total
            self.R = self.R * self.N / total
        
        return dS, dI, dR
    
    def update_parameter_classification(self, param_gradients: torch.Tensor):
        """根据梯度活跃度更新参数分类"""
        param_flat = param_gradients.view(-1)
        
        # 根据梯度幅度重新分类参数
        gradient_magnitude = torch.abs(param_flat)
        threshold_high = torch.quantile(gradient_magnitude, 0.8)
        threshold_low = torch.quantile(gradient_magnitude, 0.3)
        
        # 更新分类
        self.infected_indices = set()
        self.susceptible_indices = set()
        self.recovered_indices = set()
        
        for i, grad_mag in enumerate(gradient_magnitude):
            if grad_mag > threshold_high:
                self.infected_indices.add(i)
            elif grad_mag < threshold_low:
                self.recovered_indices.add(i)
            else:
                self.susceptible_indices.add(i)
        
        # 更新种群数量
        self.I = float(len(self.infected_indices))
        self.S = float(len(self.susceptible_indices))
        self.R = float(len(self.recovered_indices))


class GillespieSimulator:
    """Gillespie算法：模拟随机病毒事件"""
    
    def __init__(self):
        pass
    
    def simulate_viral_events(self, population: ViralPopulation, num_steps: int = 20) -> List[str]:
        """
        模拟病毒事件：
        - 事件率：λ_infect = βSI/N, λ_mut = μI
        - 时间步：τ ~ Exp(1/Σλ)
        """
        events = []
        
        for _ in range(num_steps):
            # 计算事件率
            lambda_infect = population.beta * population.S * population.I / population.N if population.N > 0 else 0.0
            lambda_mut = population.mu * population.I
            
            total_rate = lambda_infect + lambda_mut
            
            if total_rate <= 0:
                break
            
            # 采样时间到下一事件：τ ~ Exp(1/Σλ)
            tau = np.random.exponential(1.0 / total_rate)
            
            # 采样事件类型
            rand_event = np.random.uniform(0, total_rate)
            
            if rand_event < lambda_infect:
                # 感染事件：S → I
                if population.S > 0:
                    population.S -= 1
                    population.I += 1
                    events.append('infection')
            else:
                # 变异事件：I → S（病毒变异回到易感）
                if population.I > 0:
                    population.I -= 1
                    population.S += 1
                    events.append('mutation')
        
        return events
    
    def generate_viral_mutation(self, param_shape: torch.Size, mu_rate: float, device: torch.device) -> torch.Tensor:
        """
        生成病毒变异：ξ ~ Poisson(μ) · N(0, σ²)
        模拟病毒基因漂移
        """
        # 泊松变异计数
        mutation_counts = np.random.poisson(mu_rate, param_shape)
        mutation_tensor = torch.tensor(mutation_counts, dtype=torch.float32, device=device)
        
        # 高斯漂移
        sigma = 1e-4  # 变异幅度
        gaussian_drift = torch.randn(param_shape, device=device) * sigma
        
        # 病毒变异：ξ ~ Poisson(μ) · N(0, σ²)
        viral_mutation = mutation_tensor * gaussian_drift
        
        return viral_mutation


class MoranProcess:
    """Moran过程：进化游戏论选择"""
    
    def __init__(self, temperature: float = 1.0):
        self.T = temperature  # Boltzmann选择温度
    
    def fitness_ratio(self, loss_old: float, loss_new: float) -> float:
        """
        计算适应度比：r = exp(-L_new/T) / exp(-L_old/T) = exp((L_old - L_new)/T)
        """
        return math.exp((loss_old - loss_new) / self.T)
    
    def fixation_probability(self, fitness_ratio: float, k: int, N: int) -> float:
        """
        固定概率：p_fix = (1 - (1/r)^k) / (1 - (1/r)^N)
        """
        if fitness_ratio <= 1.0:
            return 1.0 / N  # 中性/有害变异
        
        r = fitness_ratio
        
        try:
            numerator = 1.0 - (1.0 / r) ** k
            denominator = 1.0 - (1.0 / r) ** N
            
            if abs(denominator) < 1e-10:
                return 1.0  # 强选择极限
            
            p_fix = numerator / denominator
            return max(0.0, min(1.0, p_fix))
        
        except (OverflowError, ZeroDivisionError):
            return 1.0 if fitness_ratio > 1.0 else 0.0
    
    def select_beneficial_mutation(self, mutations: List[torch.Tensor], 
                                 fitness_ratios: List[float]) -> torch.Tensor:
        """基于固定概率选择有益变异"""
        if not mutations:
            return torch.zeros_like(mutations[0] if mutations else torch.tensor(0.0))
        
        fixation_probs = []
        N = 100  # 假设种群大小
        
        for i, (mutation, fitness_ratio) in enumerate(zip(mutations, fitness_ratios)):
            k = mutation.numel()
            prob = self.fixation_probability(fitness_ratio, k, N)
            fixation_probs.append(prob)
        
        # 加权选择
        if sum(fixation_probs) > 0:
            probs = torch.tensor(fixation_probs)
            probs = probs / probs.sum()
            
            selected_idx = torch.multinomial(probs, 1).item()
            return mutations[selected_idx]
        else:
            return mutations[0]


class ViroEvoOpt:
    """
    Viral Evolutionary Optimizer - 完全独立的病毒进化优化器
    
    核心理论：
    1. 参数θ → 病毒种群P(θ) (S, I, R)
    2. SIR微分方程组描述种群动态
    3. Gillespie算法模拟随机事件
    4. Moran过程进行进化选择
    5. 参数更新：θ_I ← θ_I - η∇L(θ_I) + ξ
    """
    
    def __init__(self, params: Iterable[torch.Tensor],
                 lr: float = 1e-3,
                 beta: float = 0.1,      # 感染率
                 gamma: float = 0.05,    # 恢复率
                 mu: float = 1e-4,       # 变异率（HIV突变率）
                 alpha: float = 1.0,     # 损失-死亡率缩放
                 temperature: float = 1.0,  # Boltzmann温度
                 gillespie_steps: int = 20,  # Monte Carlo步数（10-50）
                 dt: float = 0.01):      # SIR积分时间步
        """
        初始化病毒进化优化器
        
        Args:
            params: 要优化的参数
            lr: 学习率η
            beta: 感染率β=0.1（设计规定）
            gamma: 恢复率γ=0.05（设计规定）
            mu: 变异率μ=1e-4（HIV突变率，设计规定）
            alpha: 损失到死亡率缩放
            temperature: Boltzmann选择温度T
            gillespie_steps: Gillespie模拟步数（设计：10-50）
            dt: SIR微分方程积分时间步
        """
        self.param_groups = [{'params': list(params)}]
        self.lr = lr
        self.beta = beta
        self.gamma = gamma  
        self.mu = mu
        self.alpha = alpha
        self.temperature = temperature
        self.gillespie_steps = gillespie_steps
        self.dt = dt
        
        # 初始化组件
        self.gillespie = GillespieSimulator()
        self.moran = MoranProcess(temperature)
        self.populations = {}  # {param_id: ViralPopulation}
        self.state = {}        # 优化器状态
        
        # 初始化病毒种群
        self._initialize_populations()
    
    def _initialize_populations(self):
        """初始化病毒种群"""
        for group in self.param_groups:
            for param in group['params']:
                param_id = id(param)
                param_size = param.numel()
                
                # 创建病毒种群
                population = ViralPopulation(
                    total_params=param_size,
                    beta=self.beta,
                    gamma=self.gamma,
                    mu=self.mu,
                    alpha=self.alpha
                )
                
                self.populations[param_id] = population
                self.state[param_id] = {
                    'step': 0,
                    'mutations_count': 0,
                    'r0_history': [],
                    'last_loss': 1.0
                }
    
    def step(self, closure=None):
        """执行一步病毒进化优化"""
        if closure is not None:
            loss = closure()
        else:
            loss = None
        
        loss_value = loss.item() if loss is not None else 1.0
        
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue
                
                param_id = id(param)
                population = self.populations[param_id]
                state = self.state[param_id]
                
                state['step'] += 1
                
                with torch.no_grad():
                    # 阶段1：更新参数分类（基于梯度活跃度）
                    population.update_parameter_classification(param.grad)
                    
                    # 阶段2：SIR动力学演化
                    # 超参调整：β随epoch增加，μ衰减（设计规定）
                    adaptive_beta = self.beta * (1.0 + 0.01 * state['step'])
                    adaptive_beta = min(adaptive_beta, 0.5)  # 上限
                    adaptive_mu = self.mu * math.exp(-0.001 * state['step'])
                    adaptive_mu = max(adaptive_mu, 1e-6)  # 下限
                    
                    population.beta = adaptive_beta
                    population.mu = adaptive_mu
                    
                    # 执行SIR微分方程组
                    dS, dI, dR = population.sir_dynamics(self.dt, loss_value)
                    
                    # 记录R_0
                    R0 = population.basic_reproduction_number(loss_value)
                    state['r0_history'].append(R0)
                    
                    # 阶段3：Gillespie随机模拟
                    # 每步模拟SIR + Gillespie 10-50次（Monte Carlo）
                    viral_events = self.gillespie.simulate_viral_events(
                        population, self.gillespie_steps
                    )
                    
                    # 阶段4：病毒变异生成
                    # ξ ~ Poisson(μ) · N(0, σ²)（模拟病毒基因漂移）
                    viral_mutation = self.gillespie.generate_viral_mutation(
                        param.shape, population.mu, param.device
                    )
                    state['mutations_count'] += torch.sum(torch.abs(viral_mutation) > 1e-6).item()
                    
                    # 阶段5：Moran过程进化选择
                    # 创建多个变异候选
                    mutations = [
                        viral_mutation,
                        viral_mutation * 0.5,
                        -viral_mutation * 0.3
                    ]
                    
                    # 估算适应度比
                    fitness_ratios = []
                    for mutation in mutations:
                        # 简化的适应度估算（基于梯度对齐）
                        grad_norm = torch.norm(param.grad).item()
                        mut_norm = torch.norm(mutation).item()
                        
                        if grad_norm > 0 and mut_norm > 0:
                            alignment = -torch.sum(param.grad * mutation).item() / (grad_norm * mut_norm)
                            estimated_loss_change = alignment * 0.1
                            fitness_ratio = self.moran.fitness_ratio(loss_value, loss_value + estimated_loss_change)
                        else:
                            fitness_ratio = 1.0
                        
                        fitness_ratios.append(fitness_ratio)
                    
                    # 选择有益变异
                    selected_mutation = self.moran.select_beneficial_mutation(mutations, fitness_ratios)
                    
                    # 阶段6：参数更新
                    # 对感染参数：θ_I ← θ_I - η∇L(θ_I) + ξ
                    param_flat = param.view(-1)
                    grad_flat = param.grad.view(-1)
                    mutation_flat = selected_mutation.view(-1)
                    
                    # 应用到感染的参数
                    for i in population.infected_indices:
                        if i < len(param_flat):
                            # 病毒进化更新公式
                            param_flat[i] = param_flat[i] - self.lr * grad_flat[i] + mutation_flat[i]
                    
                    # 完整更新：θ_{t+1} = weighted avg(θ_S, θ_I, θ_R)
                    # 权重∝I（活性感染主导）
                    if population.N > 0:
                        infection_weight = population.I / population.N
                        standard_weight = (population.S + population.R) / population.N
                        
                        # 标准梯度更新（非感染参数）
                        standard_update = param.grad * self.lr * standard_weight
                        
                        # 病毒进化已经应用到感染参数，现在平衡更新
                        param.add_(-standard_update)
                    
                    state['last_loss'] = loss_value
        
        return loss
    
    def get_viral_metrics(self) -> Dict:
        """获取病毒进化指标"""
        total_mutations = sum(state['mutations_count'] for state in self.state.values())
        avg_r0 = 0.0
        avg_infected = 0.0
        total_populations = len(self.populations)
        
        for population in self.populations.values():
            avg_r0 += population.basic_reproduction_number(1.0)  # 标准化损失
            avg_infected += population.I
        
        if total_populations > 0:
            avg_r0 /= total_populations
            avg_infected /= total_populations
        
        return {
            'total_viral_mutations': total_mutations,
            'average_reproduction_number': avg_r0,
            'average_infected_population': avg_infected,
            'epidemic_strength': 1.0 if avg_r0 > 1.0 else 0.0,
            'total_viral_events': sum(len(state['r0_history']) for state in self.state.values())
        }
    
    def zero_grad(self):
        """清零梯度"""
        for group in self.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.zero_()
    
    def state_dict(self):
        """保存状态"""
        return {
            'state': self.state,
            'lr': self.lr,
            'beta': self.beta,
            'gamma': self.gamma,
            'mu': self.mu,
            'alpha': self.alpha,
            'temperature': self.temperature,
            'gillespie_steps': self.gillespie_steps,
            'dt': self.dt
        }
    
    def load_state_dict(self, state_dict):
        """加载状态并重新初始化种群"""
        self.state = state_dict.get('state', {})
        self.lr = state_dict.get('lr', self.lr)
        self.beta = state_dict.get('beta', self.beta)
        self.gamma = state_dict.get('gamma', self.gamma)
        self.mu = state_dict.get('mu', self.mu)
        self.alpha = state_dict.get('alpha', self.alpha)
        self.temperature = state_dict.get('temperature', self.temperature)
        self.gillespie_steps = state_dict.get('gillespie_steps', self.gillespie_steps)
        self.dt = state_dict.get('dt', self.dt)
        
        # 重新初始化病毒种群
        self._initialize_populations()