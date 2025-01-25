<div align="center">

# ComfyUI ultimate openpose editor

</div>

<p align="center">
  <img src="assets/editor_example.jpg" />
</p>

This is an improved version of [ComfyUI-openpose-editor](https://github.com/huchenlei/ComfyUI-openpose-editor) 
in ComfyUI, enable input and output with flexible choices. Much more convenient and easier to use. It integrates the render function which you also can intall it separately from my [ultimate-openpose-render](https://github.com/westNeighbor/ComfyUI-ultimate-openpose-render) repo or search in the Custom Nodes Manager

If you like the project, please give me a star! ⭐

## Installation

*I used [**Registry**](https://registry.comfy.org/) to publish my nodes which was a big mess for naming and version control, so discard the method, just manually install it. If you installed this through Manager of certain versions, it probably doesn't work cause the naming problem, remove them and install again manually following the below steps.*

- Do **NOT** install this node through the `Manager -> Custom Nodes Manger` search method. Just manually install it, go to ComfyUI `/custom_nodes` directory
    ```bash
    git clone https://github.com/westNeighbor/ComfyUI-ultimate-openpose-editor
    cd ./ComfyUI-ultimate-openpose-editor
    pip install -r requirements.txt # if you use portable version, see below
    ```
    if you use portable version, install requirement accordingly, for example, I have portable in my E: disk
    ```bash
    E:/ComfyUI_windows_portable/python_embeded/python.exe -m pip install -r requirements.txt
    ```
- Restart ComfyUI

## Usage
- Insert node by `Right Click -> ultimate-openpose -> Openpose Editor Node`
- `Right Click` on the node and select the `Open in Openpose Editor` to do the editting
    <p align="center">
      <img src="assets/editor_example_1.png" />
    </p>
- send back after editting
    <p align="center">
      <img src="assets/editor_example_2.png" />
    </p>

## Features
The node is very functional and rich features to fit all your needs.
- It is totally local running, no internet requiring like the [ComfyUI-openpose-editor](https://github.com/huchenlei/ComfyUI-openpose-editor)
- It can handle all kinds of situations

    - It can go without any input, you can get an empty image or add poses of persons in the editor ui.

    - It can accept POSE\_KEYPOINTS or poses in json format as input.

    - To edit poses, right click the node, and open the editor through `Open in Openpose Editor` menu. The send back poses will be shown in the pose json input area. Be ware that the edit priority is **POSE\_KEYPOINT > POSE\_JSON** 

    - It can output pose images, or POSE\_KEYPOINTS or the json poses. Be ware that the output priority is **POSE\_JSON > POSE\_KEYPOINT**

    - It integrates the render options too, so you can use it as an render node too, or check my [ultimate-openpose-render](https://github.com/westNeighbor/ComfyUI-ultimate-openpose-render) node.

    <p align="center">
      <img src="assets/editor_example_3.jpg" />
    </p>


## Credits
- https://github.com/huchenlei/ComfyUI-openpose-editor
