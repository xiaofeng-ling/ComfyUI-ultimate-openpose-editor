import { app } from "../../scripts/app.js";
import { ComfyDialog, $el } from "../../scripts/ui.js";
import { ComfyApp } from "../../scripts/app.js";


function addMenuHandler(nodeType, cb) {
    const getOpts = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function () {
        const r = getOpts.apply(this, arguments);
        cb.apply(this, arguments);
        return r;
    };
}

class OpenposeEditorDialog extends ComfyDialog {
    static timeout = 5000;
    static instance = null;

    static getInstance() {
        if (!OpenposeEditorDialog.instance) {
            OpenposeEditorDialog.instance = new OpenposeEditorDialog();
        }

        return OpenposeEditorDialog.instance;
    }

    constructor() {
        super();
        this.element = $el("div.comfy-modal", {
            parent: document.body,
            style: {
                width: "80vw",
                height: "80vh",
            },
        }, [
            $el("div.comfy-modal-content", {
                style: {
                    width: "100%",
                    height: "100%",
                },
            }, this.createButtons()),
        ]);
        this.is_layout_created = false;

        window.addEventListener("message", (event) => {
            if (event.source !== this.iframeElement.contentWindow) {
                return;
            }

            const message = event.data;
            if (message.modalId === 0) {
                const targetNode = ComfyApp.clipspace_return_node;
                const textAreaElement = targetNode.widgets[14].element;
                textAreaElement.value = JSON.stringify(event.data.poses);
		ComfyApp.onClipspaceEditorClosed();
                this.close();
            }
        });
    }

    createButtons() {
        const closeBtn = $el("button", {
            type: "button",
            textContent: "Close",
            onclick: () => this.close(),
        });
        return [
            closeBtn,
        ];
    }

    close() {
        super.close();
    }

    async show() {
        if (!this.is_layout_created) {
            this.createLayout();
            this.is_layout_created = true;
            await this.waitIframeReady();
        }

        const targetNode = ComfyApp.clipspace_return_node;
        if ((targetNode.inputs?.[0].link || targetNode.inputs?.[targetNode.inputs.length-1].widget) && targetNode.widgets.length > 15){
            const textAreaElement = targetNode.widgets[15].element;
            this.element.style.display = "flex";
            this.setCanvasJSONString(textAreaElement.value.replace(/'/g, '"'));
        } else {
            const textAreaElement = targetNode.widgets[14].element;
            this.element.style.display = "flex";
            if (textAreaElement.value === "") {
                let resolution_x = targetNode.widgets[3].value;
                let resolution_y = Math.floor(768*(resolution_x*1.0/512));
                if (resolution_x < 64){
                    resolution_x = 512;
                    resolution_y = 768;
                }

                let pose = `[{"people": [{"pose_keypoints_2d": [], "face_keypoints_2d": [], "hand_left_keypoints_2d": [], "hand_right_keypoints_2d": []}], "canvas_height": ${resolution_y}, "canvas_width": ${resolution_x}}]`;
                this.setCanvasJSONString(pose);
            } else {
                this.setCanvasJSONString(textAreaElement.value.replace(/'/g, '"'));
            }
        }
    }

    createLayout() {
        this.iframeElement = $el("iframe", {
            // Change to for local dev
            src: "extensions/ComfyUI-ultimate-openpose-editor/ui/OpenposeEditor.html",
            style: {
                width: "100%",
                height: "100%",
                border: "none",
            },
        });
        const modalContent = this.element.querySelector(".comfy-modal-content");
        while (modalContent.firstChild) {
            modalContent.removeChild(modalContent.firstChild);
        }
        modalContent.appendChild(this.iframeElement);
        modalContent.appendChild(this.createButtons()[0]);
    }

    waitIframeReady() {
        return new Promise((resolve, reject) => {
            const receiveMessage =  (event) => {
                if (event.source !== this.iframeElement.contentWindow) {
                    return;
                }
                if (event.data.ready) {
                    window.removeEventListener("message", receiveMessage);
                    clearTimeout(timeoutHandle);
                    resolve();
                }
            };
            const timeoutHandle = setTimeout(() => {
                reject(new Error("Timeout"));
            }, OpenposeEditorDialog.timeout);

            window.addEventListener("message", receiveMessage);
        });
    }

    setCanvasJSONString(jsonString) {
        this.iframeElement.contentWindow.postMessage({
            modalId: 0,
            poses: JSON.parse(jsonString)
        }, "*");
    }
}

app.registerExtension({
    name: "OpenposeEditor",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "OpenposeEditorNode") {
            addMenuHandler(nodeType, function (_, options) {
                options.unshift({
                    content: "Open in Openpose Editor",
                    callback: () => {
                        // `this` is the node instance
                        ComfyApp.copyToClipspace(this);
                        ComfyApp.clipspace_return_node = this;

                        const dlg = OpenposeEditorDialog.getInstance();
                        dlg.show();
                    },
                });
            });
        }
    }
});
