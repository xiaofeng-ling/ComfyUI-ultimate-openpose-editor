# ComfyUI-ultimate-openpose-editor

This is an improved version of [ComfyUI-openpose-editor](https://github.com/huchenlei/ComfyUI-openpose-editor) 
in ComfyUI, enable input and output with flexible choices. Much more convenient and easier to use. It integrates the render function which you also can intall it separately from my [ultimate-openpose-render](https://github.com/westNeighbor/ComfyUI-ultimate-openpose-render) repo or search in the Custom Nodes Manager

## Install & Features & Usage

- Just install this repo through the Manager or the Git, or you can just simply copy to your *ComfyUI/custom\_nodes/* directory
- Insert node by `Right Click -> ultimate-openpose -> Openpose Edito Node`
- The node enable multiple functions:
    - It can accept POSE\_KEYPOINTS or poses in json format as input.
    - To edit poses, right click the node, and open the editor through *Open in Openpose Editor* menu. The send back poses will be shown in the pose json input area. Be ware that the edit priority is POSE\_KEYPOINT > POSE\_JSON
    - It can output pose images, or POSE\_KEYPOINTS or the json poses. Be ware that the output priority is POSE\_JSON > POSE\_KEYPOINT
