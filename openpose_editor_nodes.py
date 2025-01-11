import json
import torch
import numpy as np
from .util import draw_pose_json, draw_pose

OpenposeJSON = dict

class OpenposeEditorNode:
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
                "POSE_JSON": ("STRING", {"multiline": True}),
                "POSE_KEYPOINT": ("POSE_KEYPOINT",{"default": None}),
            },
        }

    RETURN_NAMES = ("POSE_IMAGE", "POSE_KEYPOINT", "POSE_JSON")
    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT", "STRING")
    OUTPUT_NODE = True
    FUNCTION = "load_pose"
    CATEGORY = "ultimate-openpose"

    def load_pose(self, show_body, show_face, show_hands, resolution_x, pose_marker_size, face_marker_size, hand_marker_size, POSE_JSON: str, POSE_KEYPOINT=None) -> tuple[OpenposeJSON]:
        '''
        priority output is: POSE_JSON > POSE_KEYPOINT
        priority edit is: POSE_KEYPOINT > POSE_JSON
        '''
        if POSE_JSON:
            POSE_JSON = POSE_JSON.replace("'",'"').replace('None','[]')
            POSE_PASS = POSE_JSON
            if POSE_KEYPOINT is not None:
                POSE_PASS = json.dumps(POSE_KEYPOINT,indent=4).replace("'",'"').replace('None','[]')
            pose_imgs = draw_pose_json(POSE_JSON, resolution_x, show_body, show_face, show_hands, pose_marker_size, face_marker_size, hand_marker_size)
            if pose_imgs:
                pose_imgs_np = np.array(pose_imgs).astype(np.float32) / 255
                return {
                    "ui": {"POSE_JSON": [POSE_PASS]},
                    "result": (torch.from_numpy(pose_imgs_np), json.loads(POSE_JSON), POSE_JSON)
                }
        elif POSE_KEYPOINT is not None:
            POSE_JSON = json.dumps(POSE_KEYPOINT,indent=4).replace("'",'"').replace('None','[]')
            pose_imgs = draw_pose_json(POSE_JSON, resolution_x, show_body, show_face, show_hands, pose_marker_size, face_marker_size, hand_marker_size)
            if pose_imgs:
                pose_imgs_np = np.array(pose_imgs).astype(np.float32) / 255
                return {
                    "ui": {"POSE_JSON": [POSE_JSON]},
                    "result": (torch.from_numpy(pose_imgs_np), json.loads(POSE_JSON), POSE_JSON)
                }

        # otherwise output blank images
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
