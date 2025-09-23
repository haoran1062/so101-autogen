# -*- coding: utf-8 -*-
"""
核心模块 - 仿真基础管理

注意：此模块需要Isaac Sim环境，建议延迟导入
"""

# 延迟导入Isaac Sim相关模块，避免在非Isaac Sim环境下导入失败
try:
    from .simulation_manager import SimulationManager
    from .world_setup import WorldSetup
    __all__ = ['SimulationManager', 'WorldSetup']
except ImportError:
    # 如果Isaac Sim环境不可用，提供警告
    import warnings
    warnings.warn("Isaac Sim环境不可用，core模块功能受限", ImportWarning)
    __all__ = []
