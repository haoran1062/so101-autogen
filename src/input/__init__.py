# -*- coding: utf-8 -*-
"""
输入控制模块
包含键盘处理和输入管理功能
"""

def get_keyboard_handler():
    """延迟导入键盘处理器"""
    from .keyboard_handler import KeyboardHandler
    return KeyboardHandler

def get_input_manager():
    """延迟导入输入管理器"""
    from .input_manager import InputManager
    return InputManager

__all__ = ['get_keyboard_handler', 'get_input_manager']
