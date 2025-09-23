# -*- coding: utf-8 -*-
"""
相机控制模块
包含多相机管理和切换功能
"""

def get_multi_camera_controller():
    """延迟导入多相机控制器"""
    from .multi_camera import MultiCameraController
    return MultiCameraController

def get_multi_camera_controller_from_ref():
    """延迟导入从大脚本移植的多相机控制器"""
    from .multi_camera_controller import MultiCameraController
    return MultiCameraController

__all__ = ['get_multi_camera_controller', 'get_multi_camera_controller_from_ref']
