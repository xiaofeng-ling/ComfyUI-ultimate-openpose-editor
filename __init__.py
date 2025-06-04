from .openpose_editor_nodes import OpenposeEditorNode
from .appendage_editor_nodes import AppendageEditorNode, PoseRendererNode


WEB_DIRECTORY = "js"

NODE_CLASS_MAPPINGS = {
    "OpenposeEditorNode": OpenposeEditorNode,
    "AppendageEditorNode": AppendageEditorNode,
    "PoseRendererNode": PoseRendererNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenposeEditorNode": "Openpose Editor Node",
    "AppendageEditorNode": "Appendage Editor",
    "PoseRendererNode": "Pose Renderer",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
