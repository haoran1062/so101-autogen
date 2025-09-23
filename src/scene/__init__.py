# -*- coding: utf-8 -*-
"""
场景管理模块 - 负责物体加载、位置管理和随机生成
"""

from .random_generator import RandomPositionGenerator

# ObjectLoader需要Isaac Sim环境，延迟导入
def get_object_loader():
    """延迟导入ObjectLoader，避免在非Isaac Sim环境下导入失败"""
    from .object_loader import ObjectLoader
    return ObjectLoader

__all__ = ['RandomPositionGenerator', 'get_object_loader']
