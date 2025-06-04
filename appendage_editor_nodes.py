import json
import copy
import math
import torch
import numpy as np
from .util import scale, draw_pose_json
from .openpose_editor_nodes import OpenposeEditorNode

class AppendageEditorNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "POSE_KEYPOINT": ("POSE_KEYPOINT",),
                "appendage_type": ([
                    "left_upper_arm", "left_forearm", "left_full_arm",
                    "right_upper_arm", "right_forearm", "right_full_arm", 
                    "left_upper_leg", "left_lower_leg", "left_full_leg",
                    "right_upper_leg", "right_lower_leg", "right_full_leg",
                    "left_hand", "right_hand", "left_foot", "right_foot",
                    "torso", "shoulders"
                ], {
                    "default": "left_upper_arm"
                }),
            },
            "optional": {
                "scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05
                }),
                "x_offset": ("FLOAT", {
                    "default": 0.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.01
                }),
                "y_offset": ("FLOAT", {
                    "default": 0.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.01
                }),
                "rotation": ("FLOAT", {
                    "default": 0.0,
                    "min": -180.0,
                    "max": 180.0,
                    "step": 1.0
                }),
                "bidirectional_scale": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If true, scales in both directions from pivot. If false, only scales away from body to prevent cannibalizing adjacent parts."
                }),
                "length_scale": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If true, scales the total length of the appendage while keeping the attachment point fixed. Overrides bidirectional_scale."
                }),
                "person_index": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 100,
                    "tooltip": "Person to edit (-1 for all people)"
                }),
                "list_mismatch_behavior": (["truncate", "loop", "repeat"], {"default": "loop", "tooltip": "Truncate: Truncate the list to the shortest length. Loop: Loop the list to the longest length. Repeat: Repeat the list to the longest length."}),
            },
        }

    RETURN_NAMES = ("POSE_KEYPOINT",)
    RETURN_TYPES = ("POSE_KEYPOINT",)
    FUNCTION = "edit_appendage"
    CATEGORY = "ultimate-openpose"

    def edit_appendage(self, POSE_KEYPOINT, appendage_type, scale=1.0, x_offset=0.0, y_offset=0.0, rotation=0.0, bidirectional_scale=False, length_scale=False, person_index=-1, list_mismatch_behavior="loop"):
        if POSE_KEYPOINT is None:
            return (None,)
        
        # Deep copy to avoid modifying the original
        pose_data = copy.deepcopy(POSE_KEYPOINT)
        if not isinstance(pose_data, list):
            pose_data = [pose_data]
        
        pose_count = len(pose_data)
        
        # Normalize scale parameters to handle lists vs single floats using the original node's methods
        scale_params = [scale, x_offset, y_offset, rotation]
        output_length = OpenposeEditorNode.determine_output_length(scale_params, pose_count, list_mismatch_behavior)
        
        scale_list = OpenposeEditorNode.normalize_scale_parameter(scale, output_length, list_mismatch_behavior)
        x_offset_list = OpenposeEditorNode.normalize_scale_parameter(x_offset, output_length, list_mismatch_behavior)
        y_offset_list = OpenposeEditorNode.normalize_scale_parameter(y_offset, output_length, list_mismatch_behavior)
        rotation_list = OpenposeEditorNode.normalize_scale_parameter(rotation, output_length, list_mismatch_behavior)
        
        # Process each frame with its corresponding parameter values
        output_pose_data = []
        
        for i in range(output_length):
            # Get the pose data for this index
            pose_idx = i if i < pose_count else pose_count - 1
            if list_mismatch_behavior == "loop" and pose_count > 0:
                pose_idx = i % pose_count
            
            # Get current frame and parameter values
            current_frame = copy.deepcopy(pose_data[pose_idx])
            current_scale = scale_list[i]
            current_x_offset = x_offset_list[i]
            current_y_offset = y_offset_list[i]
            current_rotation = rotation_list[i]
            
            # Apply transformations to this frame
            if 'people' in current_frame:
                people_to_edit = range(len(current_frame['people'])) if person_index == -1 else [person_index]
                
                for person_idx in people_to_edit:
                    if person_idx >= len(current_frame['people']):
                        continue
                        
                    person = current_frame['people'][person_idx]
                    
                    if appendage_type in ["left_hand", "right_hand"]:
                        self._edit_hand_appendage(person, appendage_type, current_scale, current_x_offset, current_y_offset, current_rotation, bidirectional_scale, length_scale)
                    else:
                        self._edit_body_appendage(person, appendage_type, current_scale, current_x_offset, current_y_offset, current_rotation, bidirectional_scale, length_scale)
            
            output_pose_data.append(current_frame)
        
        return (output_pose_data if isinstance(POSE_KEYPOINT, list) else output_pose_data[0],)
    
    def _edit_hand_appendage(self, person, appendage_type, scale_factor, x_offset, y_offset, rotation, bidirectional_scale, length_scale):
        """Edit hand appendages using hand keypoints."""
        keypoint_field = "hand_left_keypoints_2d" if appendage_type == "left_hand" else "hand_right_keypoints_2d"
        
        if keypoint_field not in person or not person[keypoint_field]:
            return
        
        keypoints = person[keypoint_field]
        
        # Use wrist (first point) as pivot for hands
        if len(keypoints) >= 3 and keypoints[2] > 0:
            pivot = [keypoints[0], keypoints[1]]
        else:
            # Calculate center of mass if wrist not available
            pivot = self._calculate_center_of_mass(keypoints)
            if pivot is None:
                return
        
        # Apply transformations
        new_keypoints = self._apply_transformations(keypoints, scale_factor, x_offset, y_offset, rotation, pivot, bidirectional_scale, length_scale)
        person[keypoint_field] = new_keypoints
    
    def _edit_body_appendage(self, person, appendage_type, scale_factor, x_offset, y_offset, rotation, bidirectional_scale, length_scale):
        """Edit body appendages (arms, legs, feet) using body pose keypoints."""
        if 'pose_keypoints_2d' not in person or not person['pose_keypoints_2d']:
            return
        
        keypoints = person['pose_keypoints_2d']
        
        # Get keypoint indices for the specific appendage
        appendage_indices, pivot_index = self._get_appendage_indices(appendage_type)
        if not appendage_indices:
            return
        
        # Calculate pivot point for the appendage
        pivot = self._calculate_appendage_pivot(keypoints, appendage_indices, pivot_index)
        if pivot is None:
            return
        
        # Apply transformations only to the appendage keypoints
        new_keypoints = keypoints[:]
        
        if length_scale and scale_factor != 1.0:
            # For length scaling, we need to handle multi-segment appendages specially
            new_keypoints = self._apply_length_scale(keypoints, appendage_indices, pivot_index, scale_factor)
        
        for i in range(0, len(keypoints), 3):
            keypoint_idx = i // 3
            if keypoint_idx in appendage_indices and len(keypoints) > i+2:
                x, y, conf = keypoints[i], keypoints[i+1], keypoints[i+2]
                
                if conf > 0:
                    # If length scaling was applied, use the new keypoints
                    if length_scale and scale_factor != 1.0:
                        x, y = new_keypoints[i], new_keypoints[i+1]
                    else:
                        # Apply rotation
                        if rotation != 0.0:
                            rad = math.radians(rotation)
                            cos_r, sin_r = math.cos(rad), math.sin(rad)
                            rel_x, rel_y = x - pivot[0], y - pivot[1]
                            x = rel_x * cos_r - rel_y * sin_r + pivot[0]
                            y = rel_x * sin_r + rel_y * cos_r + pivot[1]
                        
                        # Apply scaling with directional control
                        if scale_factor != 1.0:
                            if bidirectional_scale:
                                scaled_point = scale([x, y], scale_factor, pivot)
                                x, y = scaled_point[0], scaled_point[1]
                            else:
                                # Unidirectional scaling - only scale away from body
                                x, y = self._apply_unidirectional_scale([x, y], scale_factor, pivot, keypoint_idx, pivot_index)
                    
                    # Always apply rotation and offset even with length scaling
                    if length_scale and rotation != 0.0:
                        rad = math.radians(rotation)
                        cos_r, sin_r = math.cos(rad), math.sin(rad)
                        rel_x, rel_y = x - pivot[0], y - pivot[1]
                        x = rel_x * cos_r - rel_y * sin_r + pivot[0]
                        y = rel_x * sin_r + rel_y * cos_r + pivot[1]
                    
                    # Apply offset
                    x += x_offset
                    y += y_offset
                
                new_keypoints[i] = x
                new_keypoints[i+1] = y
        
        person['pose_keypoints_2d'] = new_keypoints
    
    def _get_appendage_indices(self, appendage_type):
        """Get OpenPose keypoint indices for specific appendages and their pivot points."""
        # COCO 18-keypoint format (0-based) used by ComfyUI ControlNet Aux OpenPose Pose node:
        # 0: Nose, 1: Neck, 2: RShoulder, 3: RElbow, 4: RWrist, 5: LShoulder, 6: LElbow, 7: LWrist,
        # 8: RHip, 9: RKnee, 10: RAnkle, 11: LHip, 12: LKnee, 13: LAnkle, 14: REye, 15: LEye, 16: REar, 17: LEar
        
        appendage_map = {
            # Arms - COCO format
            "left_upper_arm": ([5, 6], 5),          # LShoulder, LElbow (pivot: shoulder)
            "left_forearm": ([6, 7], 6),            # LElbow, LWrist (pivot: elbow)
            "left_full_arm": ([5, 6, 7], 5),        # LShoulder, LElbow, LWrist (pivot: shoulder)
            "right_upper_arm": ([2, 3], 2),         # RShoulder, RElbow (pivot: shoulder)
            "right_forearm": ([3, 4], 3),           # RElbow, RWrist (pivot: elbow)
            "right_full_arm": ([2, 3, 4], 2),       # RShoulder, RElbow, RWrist (pivot: shoulder)
            
            # Legs - COCO format (FIXED!)
            "left_upper_leg": ([11, 12], 11),       # LHip, LKnee (pivot: hip)
            "left_lower_leg": ([12, 13], 12),       # LKnee, LAnkle (pivot: knee) - FIXED: was [13,14] which was LAnkle,REye!
            "left_full_leg": ([11, 12, 13], 11),    # LHip, LKnee, LAnkle (pivot: hip)
            "right_upper_leg": ([8, 9], 8),         # RHip, RKnee (pivot: hip)
            "right_lower_leg": ([9, 10], 9),        # RKnee, RAnkle (pivot: knee)
            "right_full_leg": ([8, 9, 10], 8),      # RHip, RKnee, RAnkle (pivot: hip)
            
            # Feet - COCO format (no foot keypoints in COCO, use ankle only)
            "left_foot": ([13], 13),                # LAnkle only (pivot: ankle)
            "right_foot": ([10], 10),               # RAnkle only (pivot: ankle)
            
            # Torso and Shoulders - COCO format
            "torso": ([1, 2, 5, 8, 11], 1),         # Neck, RShoulder, LShoulder, RHip, LHip (pivot: neck)
            "shoulders": ([2, 5], 1),               # RShoulder, LShoulder (pivot: neck)
        }
        
        result = appendage_map.get(appendage_type, ([], None))
        return result[0], result[1]
    
    def _calculate_appendage_pivot(self, keypoints, appendage_indices, pivot_index):
        """Calculate pivot point for body appendage using specified pivot index."""
        if pivot_index is not None:
            # Use specific pivot point (e.g., shoulder for upper arm, elbow for forearm)
            i = pivot_index * 3
            if len(keypoints) > i+2 and keypoints[i+2] > 0:
                return [keypoints[i], keypoints[i+1]]
        
        # Fallback to center of mass if pivot point not available
        valid_points = []
        for idx in appendage_indices:
            i = idx * 3
            if len(keypoints) > i+2 and keypoints[i+2] > 0:
                valid_points.append([keypoints[i], keypoints[i+1]])
        
        if not valid_points:
            return None
        
        pivot_x = sum(p[0] for p in valid_points) / len(valid_points)
        pivot_y = sum(p[1] for p in valid_points) / len(valid_points)
        return [pivot_x, pivot_y]
    
    def _apply_unidirectional_scale(self, point, scale_factor, pivot, keypoint_idx, pivot_index):
        """Apply scaling only in the direction away from the body/pivot."""
        x, y = point
        
        if keypoint_idx == pivot_index:
            # Don't scale the pivot point itself
            return x, y
        
        # Calculate direction vector from pivot to point
        dx = x - pivot[0]
        dy = y - pivot[1]
        
        # Scale only the distance, keeping direction
        distance = math.sqrt(dx*dx + dy*dy)
        if distance > 0:
            new_distance = distance * scale_factor
            scale_ratio = new_distance / distance
            
            new_x = pivot[0] + dx * scale_ratio
            new_y = pivot[1] + dy * scale_ratio
            return new_x, new_y
        
        return x, y
    
    def _calculate_center_of_mass(self, keypoints):
        """Calculate center of mass from valid keypoints."""
        valid_points = []
        for i in range(0, len(keypoints), 3):
            if len(keypoints) > i+2 and keypoints[i+2] > 0:
                valid_points.append([keypoints[i], keypoints[i+1]])
        
        if not valid_points:
            return None
        
        pivot_x = sum(p[0] for p in valid_points) / len(valid_points)
        pivot_y = sum(p[1] for p in valid_points) / len(valid_points)
        return [pivot_x, pivot_y]
    
    def _apply_transformations(self, keypoints, scale_factor, x_offset, y_offset, rotation, pivot, bidirectional_scale, length_scale):
        """Apply transformations to all keypoints."""
        new_keypoints = []
        for i in range(0, len(keypoints), 3):
            if len(keypoints) > i+2:
                x, y, conf = keypoints[i], keypoints[i+1], keypoints[i+2]
                
                if conf > 0:
                    # Apply rotation
                    if rotation != 0.0:
                        rad = math.radians(rotation)
                        cos_r, sin_r = math.cos(rad), math.sin(rad)
                        rel_x, rel_y = x - pivot[0], y - pivot[1]
                        x = rel_x * cos_r - rel_y * sin_r + pivot[0]
                        y = rel_x * sin_r + rel_y * cos_r + pivot[1]
                    
                    # Apply scaling
                    if scale_factor != 1.0:
                        if length_scale:
                            # Length scale - only scale the distance, keeping direction
                            distance = math.sqrt((x - pivot[0])**2 + (y - pivot[1])**2)
                            if distance > 0:
                                new_distance = distance * scale_factor
                                scale_ratio = new_distance / distance
                                x, y = pivot[0] + (x - pivot[0]) * scale_ratio, pivot[1] + (y - pivot[1]) * scale_ratio
                        else:
                            if bidirectional_scale:
                                scaled_point = scale([x, y], scale_factor, pivot)
                                x, y = scaled_point[0], scaled_point[1]
                            else:
                                # For hands, use unidirectional scaling from wrist
                                x, y = self._apply_unidirectional_scale([x, y], scale_factor, pivot, i//3, 0)
                    
                    # Apply offset
                    x += x_offset
                    y += y_offset
                
                new_keypoints.extend([x, y, conf])
        
        return new_keypoints

    def _apply_length_scale(self, keypoints, appendage_indices, pivot_index, scale_factor):
        """Apply length scaling to multi-segment appendages."""
        new_keypoints = keypoints[:]
        
        if scale_factor == 1.0:
            return new_keypoints
        
        # Get valid appendage points in order
        appendage_points = []
        for idx in sorted(appendage_indices):
            i = idx * 3
            if len(keypoints) > i+2 and keypoints[i+2] > 0:
                appendage_points.append([keypoints[i], keypoints[i+1], idx])
        
        if len(appendage_points) < 2:
            return new_keypoints
        
        # Calculate current total length
        total_length = 0.0
        for i in range(1, len(appendage_points)):
            dx = appendage_points[i][0] - appendage_points[i-1][0]
            dy = appendage_points[i][1] - appendage_points[i-1][1]
            total_length += math.sqrt(dx*dx + dy*dy)
        
        if total_length == 0:
            return new_keypoints
        
        # Calculate new total length
        new_total_length = total_length * scale_factor
        
        # Keep pivot point fixed and redistribute others
        pivot_point = None
        for point in appendage_points:
            if point[2] == pivot_index:
                pivot_point = point
                break
        
        if pivot_point is None:
            pivot_point = appendage_points[0]  # Use first point as pivot
        
        # Redistribute points along the appendage direction
        for i, point in enumerate(appendage_points):
            if point[2] == pivot_point[2]:
                continue  # Keep pivot fixed
            
            # Calculate original distance from pivot
            dx = point[0] - pivot_point[0]
            dy = point[1] - pivot_point[1]
            original_distance = math.sqrt(dx*dx + dy*dy)
            
            if original_distance > 0:
                # Calculate new distance (scaled)
                new_distance = original_distance * scale_factor
                
                # Calculate new position
                direction_x = dx / original_distance
                direction_y = dy / original_distance
                
                new_x = pivot_point[0] + direction_x * new_distance
                new_y = pivot_point[1] + direction_y * new_distance
                
                # Update keypoints
                idx = point[2] * 3
                new_keypoints[idx] = new_x
                new_keypoints[idx+1] = new_y
        
        return new_keypoints


class PoseRendererNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "POSE_KEYPOINT": ("POSE_KEYPOINT",),
            },
            "optional": {
                "show_body": ("BOOLEAN", {"default": True}),
                "show_face": ("BOOLEAN", {"default": True}),
                "show_hands": ("BOOLEAN", {"default": True}),
                "resolution_x": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 12800
                }),
                "pose_marker_size": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 100
                }),
                "face_marker_size": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 100
                }),
                "hand_marker_size": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 100
                }),
            },
        }

    RETURN_NAMES = ("IMAGE",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "render_pose"
    CATEGORY = "ultimate-openpose"

    def render_pose(self, POSE_KEYPOINT, show_body=True, show_face=True, show_hands=True, resolution_x=-1, pose_marker_size=4, face_marker_size=3, hand_marker_size=2):
        if POSE_KEYPOINT is None:
            # Create default blank image
            W, H = 512, 768
            W_scaled = resolution_x if resolution_x >= 64 else W
            H_scaled = int(H * (W_scaled * 1.0 / W))
            blank_img = np.zeros((H_scaled, W_scaled, 3), dtype=np.uint8)
            pose_img_np = np.array([blank_img]).astype(np.float32) / 255
            return (torch.from_numpy(pose_img_np),)
        
        # Convert pose keypoints to JSON string for rendering
        if isinstance(POSE_KEYPOINT, list):
            pose_json = json.dumps(POSE_KEYPOINT)
        else:
            pose_json = json.dumps([POSE_KEYPOINT])
        
        # Use existing draw_pose_json function to render
        pose_imgs = draw_pose_json(
            pose_json, 
            resolution_x, 
            show_body, 
            show_face, 
            show_hands, 
            pose_marker_size, 
            face_marker_size, 
            hand_marker_size, 
            1.0,  # hands_scale
            1.0,  # body_scale  
            1.0,  # head_scale
            1.0   # overall_scale
        )
        
        if pose_imgs:
            pose_imgs_np = np.array(pose_imgs).astype(np.float32) / 255
            return (torch.from_numpy(pose_imgs_np),)
        
        # Create default blank image if rendering fails
        W, H = 512, 768
        W_scaled = resolution_x if resolution_x >= 64 else W
        H_scaled = int(H * (W_scaled * 1.0 / W))
        blank_img = np.zeros((H_scaled, W_scaled, 3), dtype=np.uint8)
        pose_img_np = np.array([blank_img]).astype(np.float32) / 255
        return (torch.from_numpy(pose_img_np),) 