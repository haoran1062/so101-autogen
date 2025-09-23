# -*- coding: utf-8 -*-
"""
VLA Policy Client - Based on LeRobot's LeRobotServicePolicyClient
Used to connect to a VLA model server for inference.
"""

import pickle
import torch
import grpc
import time
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Attempt to import LeRobot modules
try:
    from lerobot.transport import services_pb2_grpc, services_pb2
    from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
    from lerobot.scripts.server.helpers import RemotePolicyConfig, TimedObservation
    LEROBOT_AVAILABLE = True
except ImportError:
    logger.warning("LeRobot modules are not available. VLA client functionality will be limited.")
    LEROBOT_AVAILABLE = False

# SO-101 joint names (copied from reference code)
SINGLE_ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift", 
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper"
]


class SO101VLAPolicyClient:
    """
    SO-101 VLA Policy Client
    Based on LeRobot's LeRobotServicePolicyClient, adapted for our environment.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 4399,
        timeout_ms: int = 5000,
        checkpoint_path: str = None,
        action_horizon: int = 50,
        language_instruction: str = "pick and place the orange",
        device: str = "cuda"
    ):
        """
        Initializes the VLA policy client.
        
        Args:
            host: The hostname of the policy server.
            port: The port of the policy server.
            timeout_ms: The timeout in milliseconds.
            checkpoint_path: The path to the model checkpoint.
            action_horizon: The length of the action sequence.
            language_instruction: The language instruction for the task.
            device: The computation device.
        """
        if not LEROBOT_AVAILABLE:
            raise ImportError("LeRobot modules are not available. Cannot create the VLA client.")
        
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.action_horizon = action_horizon
        self.language_instruction = language_instruction
        self.device = device
        
        # Camera information configuration
        self.camera_infos = {
            "front": (480, 640, 3),
            "wrist": (480, 640, 3)
        }
        
        # List of camera keys - following the reference implementation
        self.camera_keys = ["front", "wrist"]
        
        # Create LeRobot feature configuration
        self.lerobot_features = self._create_lerobot_features()
        
        # Create policy configuration
        self.policy_config = RemotePolicyConfig(
            "smolvla",  # Policy type
            checkpoint_path,  # Path to the pretrained model
            self.lerobot_features,
            action_horizon,
            device
        )
        
        # Initialize gRPC connection
        self._init_grpc_connection()
        
        # Initialize the service
        self._init_service()
        
        logger.info(f"✅ VLA policy client initialized.")
        logger.info(f"   Server: {host}:{port}")
        logger.info(f"   Model Path: {checkpoint_path}")
        logger.info(f"   Action Horizon: {action_horizon}")
    
    
    def _create_lerobot_features(self):
        """Creates the LeRobot feature configuration."""
        lerobot_features = {}
        
        # Joint state features
        lerobot_features['observation.state'] = {
            'dtype': 'float32',
            'shape': (6,),
            'names': [f'{joint_name}.pos' for joint_name in SINGLE_ARM_JOINT_NAMES],
        }
        
        # Camera image features
        for camera_key, camera_image_shape in self.camera_infos.items():
            lerobot_features[f'observation.images.{camera_key}'] = {
                'dtype': 'image',
                'shape': (camera_image_shape[0], camera_image_shape[1], 3),
                'names': ['height', 'width', 'channels'],
            }
        
        return lerobot_features
    
    def _init_grpc_connection(self):
        """Initializes the gRPC connection."""
        service_address = f'{self.host}:{self.port}'
        
        self.channel = grpc.insecure_channel(
            service_address, 
            grpc_channel_options()
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        
        logger.info(f"✅ gRPC connection established: {service_address}")
    
    def _init_service(self):
        """Initializes the service connection."""
        try:
            # Send Ready signal
            self.stub.Ready(services_pb2.Empty())
            
            # Send policy configuration
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)
            
            logger.info("📤 Sending policy configuration to the server...")
            self.stub.SendPolicyInstructions(policy_setup)
            logger.info("✅ Policy server is ready.")
            
        except grpc.RpcError as e:
            logger.error(f"❌ Failed to connect to the policy server: {e}")
            raise RuntimeError(f"Could not connect to the policy server at {self.host}:{self.port}")
    
    def _send_observation(self, observation_dict: dict):
        """Sends observation data to the policy server - follows the reference implementation."""
        # Process observation data exactly as in the reference implementation.
        raw_observation = {f"{key}": observation_dict[key].cpu().numpy().astype(np.uint8)[0] for key in self.camera_keys}
        raw_observation["task"] = observation_dict["task_description"]

        # Process joint position data - as in the reference implementation.
        joint_pos = self._convert_joint_pos_to_lerobot(observation_dict["joint_pos"])
        joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
        for joint_name in joint_names:
            raw_observation[f"{joint_name}.pos"] = joint_pos[0, joint_names.index(joint_name)].item()
        
        # Create a timed observation - as in the reference implementation.
        self.latest_action_step = getattr(self, 'latest_action_step', 0) + 1
        observation = TimedObservation(
            timestamp=time.time(),
            observation=raw_observation,
            timestep=self.latest_action_step,
        )
        
        # Send observation data - as in the reference implementation.
        observation_bytes = pickle.dumps(observation)
        observation_iterator = send_bytes_in_chunks(
            observation_bytes,
            services_pb2.Observation,
            log_prefix="[CLIENT] Observation",
            silent=True,
        )
        _ = self.stub.SendObservations(observation_iterator)
    
    def _receive_action(self):
        """Receives action data from the policy server."""
        actions_chunk = self.stub.GetActions(services_pb2.Empty())
        if len(actions_chunk.data) == 0:
            logger.warning("Received an empty response, waiting for the next call.")
            return None
        return pickle.loads(actions_chunk.data)
    
    def _convert_joint_pos_to_lerobot(self, joint_pos):
        """Converts joint positions to LeRobot format - follows the reference implementation."""
        # Conversion logic from the reference code.
        if isinstance(joint_pos, torch.Tensor):
            joint_pos = joint_pos.cpu().numpy()
        
        # Ensure there is a batch dimension.
        if len(joint_pos.shape) == 1:
            joint_pos_batch = joint_pos[None, ...]  # (1, 6)
        else:
            joint_pos_batch = joint_pos  # Already (1, 6)
        
        # Convert to degrees (as in the reference code).
        joint_pos_deg = joint_pos_batch / np.pi * 180.0
        
        # Use the full conversion logic from the reference code.
        processed_action = np.zeros_like(joint_pos_deg)
        
        # Define joint limits (from the reference code).
        joint_limits = {
            'shoulder_pan': (-110.0, 110.0),
            'shoulder_lift': (-100.0, 100.0),
            'elbow_flex': (-100.0, 90.0),
            'wrist_flex': (-95.0, 95.0),
            'wrist_roll': (-160.0, 160.0),
            'gripper': (-10.0, 100.0)
        }
        
        # Define motor limits (from the reference code).
        motor_limits = {
            'shoulder_pan': (-100.0, 100.0),
            'shoulder_lift': (-100.0, 100.0),
            'elbow_flex': (-100.0, 100.0),
            'wrist_flex': (-100.0, 100.0),
            'wrist_roll': (-100.0, 100.0),
            'gripper': (0.0, 100.0)
        }
        
        joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
        
        for idx, joint_name in enumerate(joint_names):
            motor_limit_range = motor_limits[joint_name]
            joint_limit_range = joint_limits[joint_name]
            processed_action[:, idx] = (joint_pos_deg[:, idx] - joint_limit_range[0]) / (joint_limit_range[1] - joint_limit_range[0]) \
                * (motor_limit_range[1] - motor_limit_range[0]) + motor_limit_range[0]
        
        return processed_action
    
    def _convert_lerobot_action_to_leisaac(self, action):
        """Converts LeRobot actions to LeIsaac format - follows the reference implementation."""
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        
        # logger.debug(f"🔍 Input action shape: {action.shape}")
        
        # Define joint limits (from the reference code).
        joint_limits = {
            'shoulder_pan': (-110.0, 110.0),
            'shoulder_lift': (-100.0, 100.0),
            'elbow_flex': (-100.0, 90.0),
            'wrist_flex': (-95.0, 95.0),
            'wrist_roll': (-160.0, 160.0),
            'gripper': (-10.0, 100.0)
        }
        
        # Define motor limits (from the reference code).
        motor_limits = {
            'shoulder_pan': (-100.0, 100.0),
            'shoulder_lift': (-100.0, 100.0),
            'elbow_flex': (-100.0, 100.0),
            'wrist_flex': (-100.0, 100.0),
            'wrist_roll': (-100.0, 100.0),
            'gripper': (0.0, 100.0)
        }
        
        joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
        
        # Conversion, as in the reference implementation.
        processed_action = np.zeros_like(action)
        
        # Check the shape of the action tensor
        if len(action.shape) == 1:
            # Shape is (6,) - the format returned by the VLA model
            action = action[None, :]  # Add a batch dimension to make it (1, 6)
            processed_action = np.zeros_like(action)
            for idx, joint_name in enumerate(joint_names):
                if idx < action.shape[1]:  # Ensure index is not out of bounds
                    motor_limit_range = motor_limits[joint_name]
                    joint_limit_range = joint_limits[joint_name]
                    
                    # Convert from motor limits back to joint limits (in degrees)
                    processed_degree = (action[:, idx] - motor_limit_range[0]) / (motor_limit_range[1] - motor_limit_range[0]) \
                        * (joint_limit_range[1] - joint_limit_range[0]) + joint_limit_range[0]
                    
                    # Convert to radians (using numpy's pi)
                    processed_radius = processed_degree / 180.0 * np.pi
                    processed_action[:, idx] = processed_radius
                else:
                    logger.warning(f"⚠️ Joint index {idx} is out of bounds for the action tensor.")
        elif len(action.shape) == 2:
            # Shape is (batch_size, 6)
            processed_action = np.zeros_like(action)
            for idx, joint_name in enumerate(joint_names):
                if idx < action.shape[1]:  # Ensure index is not out of bounds
                    motor_limit_range = motor_limits[joint_name]
                    joint_limit_range = joint_limits[joint_name]
                    
                    # Convert from motor limits back to joint limits (in degrees)
                    processed_degree = (action[:, idx] - motor_limit_range[0]) / (motor_limit_range[1] - motor_limit_range[0]) \
                        * (joint_limit_range[1] - joint_limit_range[0]) + joint_limit_range[0]
                    
                    # Convert to radians (using numpy's pi)
                    processed_radius = processed_degree / 180.0 * np.pi
                    processed_action[:, idx] = processed_radius
                else:
                    logger.warning(f"⚠️ Joint index {idx} is out of bounds for the action tensor.")
        else:
            logger.error(f"❌ Unsupported action tensor shape: {action.shape}")
            return action
        
        return processed_action
    
    def _convert_action_to_isaac_sim(self, action_chunk):
        """Converts LeRobot actions to Isaac Sim format - follows the reference implementation."""
        if action_chunk is None:
            logger.warning("⚠️ Action chunk is None, returning None.")
            return None
        
        try:
            # Process actions as in the reference implementation.
            action_list = []
            for i, action in enumerate(action_chunk):
                action_tensor = action.get_action()
                
                # Handle different action tensor shapes
                if len(action_tensor.shape) == 1:
                    # If it's a 1D tensor, add a batch dimension
                    if action_tensor.shape[0] == 6:
                        action_tensor = action_tensor[None, :]
                        # logger.debug(f"🔍 Action {i} 1D->2D: {action_tensor.shape}")
                    else:
                        logger.warning(f"⚠️ Action {i} has unexpected 1D shape: {action_tensor.shape}. Expected 6 joints, using zero action.")
                        action_tensor = torch.zeros(1, 6)
                elif len(action_tensor.shape) == 2:
                    if action_tensor.shape[1] == 6:
                        # logger.debug(f"🔍 Action {i} has normal 2D shape: {action_tensor.shape}")
                        pass
                    elif action_tensor.shape[1] == 1:
                        logger.warning(f"⚠️ Action {i} has only 1 joint, expected 6. Expanding to 6 joints.")
                        # Copy the single joint value to all 6 joints
                        single_value = action_tensor[0, 0].item()
                        action_tensor = torch.full((1, 6), single_value)
                        # logger.debug(f"🔍 Action {i} expanded: {action_tensor.shape}, Value: {single_value:.4f}")
                    else:
                        logger.warning(f"⚠️ Action {i} has unexpected shape: {action_tensor.shape}. Expected 6 joints, using zero action.")
                        action_tensor = torch.zeros(1, 6)
                elif len(action_tensor.shape) == 3:
                    # Handle 3D tensors (1, 1, 6) or (1, 6, 1)
                    if action_tensor.shape[2] == 6:
                        action_tensor = action_tensor[0, :, :]  # Remove the first dimension
                        # logger.debug(f"🔍 Action {i} 3D->2D: {action_tensor.shape}")
                    elif action_tensor.shape[1] == 6:
                        action_tensor = action_tensor[0, :, :]  # Remove the first dimension
                        # logger.debug(f"🔍 Action {i} 3D->2D: {action_tensor.shape}")
                    else:
                        logger.warning(f"⚠️ Action {i} has unexpected 3D shape: {action_tensor.shape}. Using zero action.")
                        action_tensor = torch.zeros(1, 6)
                else:
                    logger.warning(f"⚠️ Action {i} has unexpected dimensions: {action_tensor.shape}. Using zero action.")
                    action_tensor = torch.zeros(1, 6)
                
                action_list.append(action_tensor)
            
            concat_action = torch.cat(action_list, dim=0)
            # logger.debug(f"🔍 Action chunk size: {len(action_chunk)}")
            # logger.debug(f"🔍 Concatenated action shape: {concat_action.shape}")
            # logger.debug(f"🔍 Concatenated action value range: [{concat_action.min():.4f}, {concat_action.max():.4f}]")
            
            # Use the conversion function from the reference implementation
            concat_action = self._convert_lerobot_action_to_leisaac(concat_action)
            
            # logger.debug(f"🔍 Converted action shape: {concat_action.shape}")
            # logger.debug(f"🔍 Converted action value range: [{concat_action.min():.4f}, {concat_action.max():.4f}]")
            
            # Return in shape [50, 1, 6] as per the reference implementation.
            # Note: concat_action is already in [50, 6] shape, we need to add the middle dimension.
            final_action = torch.from_numpy(concat_action[:, None, :])
            # logger.debug(f"🔍 Final action shape: {final_action.shape}")
            
            return final_action
            
        except Exception as e:
            logger.error(f"❌ Action conversion failed: {e}")
            import traceback
            logger.error(f"Detailed error: {traceback.format_exc()}")
            return None
    
    def get_action(self, observation_dict: dict) -> torch.Tensor:
        """
        Gets the action prediction from the VLA model.
        
        Args:
            observation_dict: A dictionary of observation data.
                - front: Front camera image (480, 640, 3) uint8
                - wrist: Wrist camera image (480, 640, 3) uint8  
                - joint_pos: Joint positions (6,) float32
                - task_description: The task description string.
        
        Returns:
            torch.Tensor: An action sequence (action_horizon, 1, 6).
        """
        try:
            # Send observation data
            self._send_observation(observation_dict)
            
            # Attempt to receive action data multiple times until a valid response is received
            max_attempts = 3
            for attempt in range(max_attempts):
                # Wait a bit for the server to process
                time.sleep(0.2)
                
                # Receive action data
                action_chunk = self._receive_action()
                
                if action_chunk is not None:
                    break
                else:
                    logger.warning(f"Received an empty response, attempt {attempt + 1}/{max_attempts}")
                    if attempt < max_attempts - 1:
                        time.sleep(0.3)  # Wait longer
            
            if action_chunk is None:
                # If all attempts fail, return a zero action
                logger.warning("Received empty responses in all attempts. Returning a zero action.")
                zero_action = torch.zeros(self.action_horizon, 1, 6)
                return zero_action
            
            # Convert action format
            actions = self._convert_action_to_isaac_sim(action_chunk)
            
            if actions is None:
                logger.warning("Action conversion failed. Returning a zero action.")
                zero_action = torch.zeros(self.action_horizon, 1, 6)
                return zero_action
            
            return actions
            
        except Exception as e:
            logger.error(f"❌ Failed to get VLA action: {e}")
            # Return a zero action as a fallback
            zero_action = torch.zeros(self.action_horizon, 1, 6)
            return zero_action
    
    def close(self):
        """Closes the client connection."""
        if hasattr(self, 'channel'):
            self.channel.close()
            logger.info("✅ VLA client connection has been closed.")
