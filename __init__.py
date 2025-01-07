from .openpose_editor_nodes import OpenposeEditorNode


WEB_DIRECTORY = "js"

NODE_CLASS_MAPPINGS = {
    "OpenposeEditorNode": OpenposeEditorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenposeEditorNode": "Openpose Editor Node",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
