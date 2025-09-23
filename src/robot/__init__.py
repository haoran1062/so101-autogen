# -*- coding: utf-8 -*-
"""
机械臂控制模块 - Phase 2
包含IK控制器、夹爪控制器等机械臂相关功能
"""

# 延迟导入，避免在非Isaac Sim环境下导入失败
def get_ik_controller():
    """延迟导入IK控制器"""
    from .ik_controller import IKController
    return IKController

def get_gripper_controller():
    """延迟导入夹爪控制器"""
    from .gripper_controller import GripperController
    return GripperController

__all__ = ['get_ik_controller', 'get_gripper_controller']
