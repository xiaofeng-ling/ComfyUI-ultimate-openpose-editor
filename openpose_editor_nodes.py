import json
import torch
import numpy as np
from .util import draw_pose_json, draw_pose

OpenposeJSON = dict

class OpenposeEditorNode:
    @staticmethod
    def normalize_scale_parameter(scale_param, target_length, behavior):
        """
        Normalize a scale parameter to a list of the target length.
        
        Args:
            scale_param: Either a single float or list of floats
            target_length: Desired length of output list
            behavior: "truncate", "loop", or "repeat"
        
        Returns:
            List of floats with length determined by behavior
        """
        # Convert single value to list
        if not isinstance(scale_param, (list, tuple)):
            scale_list = [scale_param]
        else:
            scale_list = list(scale_param)
        
        if len(scale_list) == target_length:
            return scale_list
        
        if behavior == "truncate":
            return scale_list[:target_length]
        elif behavior == "loop":
            if len(scale_list) == 0:
                return [1.0] * target_length
            result = []
            for i in range(target_length):
                result.append(scale_list[i % len(scale_list)])
            return result
        elif behavior == "repeat":
            if len(scale_list) == 0:
                return [1.0] * target_length
            if len(scale_list) >= target_length:
                return scale_list[:target_length]
            else:
                result = scale_list[:]
                last_value = scale_list[-1]
                while len(result) < target_length:
                    result.append(last_value)
                return result
        else:
            raise ValueError(f"Unknown behavior: {behavior}")
    
    @staticmethod
    def determine_output_length(scale_params, pose_count, behavior):
        """
        Determine the output length based on scale parameters and behavior.
        """
        # Get all list lengths
        lengths = [pose_count]
        for param in scale_params:
            if isinstance(param, (list, tuple)):
                lengths.append(len(param))
        
        if behavior == "truncate":
            return min(lengths)
        else:  # loop or repeat
            return max(lengths)

    @classmethod
    def INPUT_TYPES(s):
        return {
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
                "hands_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05
                }),
                "body_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05
                }),
                "head_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05
                }),
                "overall_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.05
                }),
                "list_mismatch_behavior": (["truncate", "loop", "repeat"], {"default": "loop"}),
                "POSE_JSON": ("STRING", {"multiline": True}),
                "POSE_KEYPOINT": ("POSE_KEYPOINT",{"default": None}),
            },
        }

    RETURN_NAMES = ("POSE_IMAGE", "POSE_KEYPOINT", "POSE_JSON")
    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT", "STRING")
    OUTPUT_NODE = True
    FUNCTION = "load_pose"
    CATEGORY = "ultimate-openpose"

    def load_pose(self, show_body, show_face, show_hands, resolution_x, pose_marker_size, face_marker_size, hand_marker_size, hands_scale, body_scale, head_scale, overall_scale, list_mismatch_behavior, POSE_JSON: str, POSE_KEYPOINT=None) -> tuple[OpenposeJSON]:
        '''
        priority output is: POSE_JSON > POSE_KEYPOINT
        priority edit is: POSE_KEYPOINT > POSE_JSON
        '''
        
        # Determine the input data and count
        if POSE_JSON:
            POSE_JSON = POSE_JSON.replace("'",'"').replace('None','[]')
            POSE_PASS = POSE_JSON
            if POSE_KEYPOINT is not None:
                POSE_PASS = json.dumps(POSE_KEYPOINT,indent=4).replace("'",'"').replace('None','[]')
            
            # Parse to determine image count
            if POSE_JSON.startswith('{'):
                pose_data = [json.loads(POSE_JSON)]
            else:
                pose_data = json.loads(POSE_JSON)
            pose_count = len(pose_data)
            
        elif POSE_KEYPOINT is not None:
            if isinstance(POSE_KEYPOINT, list):
                pose_data = POSE_KEYPOINT
                pose_count = len(pose_data)
            else:
                pose_data = [POSE_KEYPOINT]
                pose_count = 1
            POSE_JSON = json.dumps(pose_data, indent=4).replace("'",'"').replace('None','[]')
            POSE_PASS = POSE_JSON
        else:
            # Default case - create blank image
            W=512
            H=768
            pose_draw = dict(bodies={'candidate':[], 'subset':[]}, faces=[], hands=[])
            pose_out = dict(pose_keypoints_2d=[], face_keypoints_2d=[], hand_left_keypoints_2d=[], hand_right_keypoints_2d=[])
            people=[dict(people=[pose_out], canvas_height=H, canvas_width=W)]

            W_scaled = resolution_x
            if resolution_x < 64:
                W_scaled = W
            H_scaled = int(H*(W_scaled*1.0/W))
            pose_img = [draw_pose(pose_draw, H_scaled, W_scaled, pose_marker_size, face_marker_size, hand_marker_size)]
            pose_img_np = np.array(pose_img).astype(np.float32) / 255

            return {
                "ui": {"POSE_JSON": people},
                "result": (torch.from_numpy(pose_img_np), people, json.dumps(people))
            }
        
        # Normalize scale parameters
        scale_params = [hands_scale, body_scale, head_scale, overall_scale]
        output_length = self.determine_output_length(scale_params, pose_count, list_mismatch_behavior)
        
        hands_scale_list = self.normalize_scale_parameter(hands_scale, output_length, list_mismatch_behavior)
        body_scale_list = self.normalize_scale_parameter(body_scale, output_length, list_mismatch_behavior)
        head_scale_list = self.normalize_scale_parameter(head_scale, output_length, list_mismatch_behavior)
        overall_scale_list = self.normalize_scale_parameter(overall_scale, output_length, list_mismatch_behavior)
        
        # Process each image with its corresponding scale values
        all_pose_imgs = []
        output_pose_data = []
        
        for i in range(output_length):
            # Get the pose data for this index
            pose_idx = i if i < pose_count else pose_count - 1
            if list_mismatch_behavior == "loop" and pose_count > 0:
                pose_idx = i % pose_count
            
            current_pose_json = json.dumps([pose_data[pose_idx]])
            
            # Get scale values for this index
            current_hands_scale = hands_scale_list[i]
            current_body_scale = body_scale_list[i]
            current_head_scale = head_scale_list[i]
            current_overall_scale = overall_scale_list[i]
            
            # Process this image
            pose_imgs = draw_pose_json(
                current_pose_json, 
                resolution_x, 
                show_body, 
                show_face, 
                show_hands, 
                pose_marker_size, 
                face_marker_size, 
                hand_marker_size, 
                current_hands_scale, 
                current_body_scale, 
                current_head_scale, 
                current_overall_scale
            )
            
            if pose_imgs:
                all_pose_imgs.extend(pose_imgs)
                # Store the processed pose data
                processed_pose = json.loads(current_pose_json)[0]
                output_pose_data.append(processed_pose)
        
        if all_pose_imgs:
            pose_imgs_np = np.array(all_pose_imgs).astype(np.float32) / 255
            return {
                "ui": {"POSE_JSON": [json.dumps(output_pose_data, indent=4)]},
                "result": (torch.from_numpy(pose_imgs_np), output_pose_data, json.dumps(output_pose_data))
            }
        
        # Fallback to original behavior if no images generated
        pose_imgs = draw_pose_json(POSE_JSON, resolution_x, show_body, show_face, show_hands, pose_marker_size, face_marker_size, hand_marker_size, hands_scale_list[0] if hands_scale_list else 1.0, body_scale_list[0] if body_scale_list else 1.0, head_scale_list[0] if head_scale_list else 1.0, overall_scale_list[0] if overall_scale_list else 1.0)
        if pose_imgs:
            pose_imgs_np = np.array(pose_imgs).astype(np.float32) / 255
            return {
                "ui": {"POSE_JSON": [POSE_PASS]},
                "result": (torch.from_numpy(pose_imgs_np), json.loads(POSE_JSON), POSE_JSON)
            }
