import gradio as gr
import torch
from torch.nn import Parameter
import modules.ui
import modules
from modules import devices, shared, extra_networks
from modules.script_callbacks import CFGDenoiserParams, on_cfg_denoiser

class Script(modules.scripts.Script):   
    def __init__(self):
        self.storedweights = {}
        self.shape = None
        self.set = False

    def title(self):
        return "CD Tuner"

    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible

    infotext_fields = None
    paste_field_names = []

    def ui(self, is_img2img):      
        with gr.Accordion("CD Tuner", open=False):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        d1 = gr.Slider(label="Detail", minimum=-10, maximum=10, value=0.0, step=0.1)
                        refresh_d1 = gr.Button(value='\U0001f504', elem_classes=["tool"])
                    with gr.Row():
                        d2 = gr.Slider(label="Detail2", minimum=-10, maximum=10, value=0.0, step=0.1,visible = True)
                        refresh_d2 = gr.Button(value='\U0001f504', elem_classes=["tool"])
                    with gr.Row():
                        cont = gr.Slider(label="Contrast", minimum=-20, maximum=20, value=0.0, step=0.1)
                        refresh_cont = gr.Button(value='\U0001f504', elem_classes=["tool"])
                with gr.Column():
                    with gr.Row():
                        col1 = gr.Slider(label="Cyan-Red", minimum=-20, maximum=20, value=0.0, step=0.1)
                        refresh_col1 = gr.Button(value='\U0001f504', elem_classes=["tool"])
                    with gr.Row():
                        col2 = gr.Slider(label="Magenta-Green", minimum=-20, maximum=20, value=0.0, step=0.1)
                        refresh_col2 = gr.Button(value='\U0001f504', elem_classes=["tool"])
                    with gr.Row():
                        col3 = gr.Slider(label="Yellow-Blue", minimum=-20, maximum=20, value=0.0, step=0.1)
                        refresh_col3 = gr.Button(value='\U0001f504', elem_classes=["tool"])
                scaling = gr.Checkbox(value=False, label="hr-scaling",interactive=True,elem_id="cdt-hr-scaling")

            with gr.Row():
                hd1 = gr.Slider(label="hr-Detail", minimum=-10, maximum=10, value=0.0, step=0.1)
                refresh_hd1 = gr.Button(value='\U0001f504', elem_classes=["tool"])
                hd2 = gr.Slider(label="hr-Detail2", minimum=-10, maximum=10, value=0.0, step=0.1,visible = True)
                refresh_hd2 = gr.Button(value='\U0001f504', elem_classes=["tool"])

            def infotexter(text):
                print(text)
                text = text.split(",")
                outs = [0] * 8 + [False]
                if len(text) == 9 : outs = [float(x) for x in text[:-1]] + [text[-1] == "True"]
                return [gr.update(value = x) for x in outs]

            params = [d1,d2,cont,col1,col2,col3,hd1,hd2,scaling]

            allsets = gr.Textbox(visible = False)
            allsets.change(fn=infotexter,inputs = [allsets], outputs =params)

            refresh_d1.click(fn=lambda x:gr.update(value = 0),outputs=[d1])
            refresh_d2.click(fn=lambda x:gr.update(value = 0),outputs=[d2])
            refresh_cont.click(fn=lambda x:gr.update(value = 0),outputs=[cont])
            refresh_col1.click(fn=lambda x:gr.update(value = 0),outputs=[col1])
            refresh_col2.click(fn=lambda x:gr.update(value = 0),outputs=[col2])
            refresh_col3.click(fn=lambda x:gr.update(value = 0),outputs=[col3])
            refresh_hd1.click(fn=lambda x:gr.update(value = 0),outputs=[hd1])
            refresh_hd2.click(fn=lambda x:gr.update(value = 0),outputs=[hd2])

        self.infotext_fields = ([(allsets,"CDT")])
        self.paste_field_names.append("CDT")

        return params

    def process(self, p, d1,d2,cont,col1,col2,col3,hd1,hd2,scaling):
        if self.set and self.storedweights and self.storedname == shared.opts.sd_model_checkpoint:
            for name in ADJUSTS:
                getset_nested_module_tensor(False,shared.sd_model, name, new_tensor = self.storedweights[name].clone())
                
        ratios = [d1,d2,cont,col1,col2,col3,hd1,hd2]
        self.active = False
        self.__init__()

        promptsets = fromprompts(p.all_prompts[0:1])

        if promptsets:
            for i,ratio in enumerate(promptsets):
                if ratio is not None:
                    ratios[i] = ratio

        if any(not(type(x) == float or type(x) == int) for x in ratios):return
        if all(x == 0 for x in ratios):return
        else: 
            self.active = True
            self.ratios = ratios

            p.extra_generation_params.update({"CDT":",".join([str(x) for x in self.ratios + [scaling]])})

            self.storedname = shared.opts.sd_model_checkpoint
            self.scaling = scaling

        if not hasattr(self,"cdt_dr_callbacks"):
            self.cdt_dr_callbacks = on_cfg_denoiser(self.denoiser_callback)

    def postprocess(self, p, processed, *args):
        if self.active:
            for name in ADJUSTS:
                getset_nested_module_tensor(False,shared.sd_model, name, new_tensor = self.storedweights[name].clone())
            self.set = False

    def denoiser_callback(self, params: CFGDenoiserParams):
        if self.active:
            if self.shape is None:self.shape = params.x.shape
            if self.shape == params.x.shape  and self.set:
                return
            else:
                if params.x.shape[2] * params.x.shape[3] > self.shape[2]*self.shape[3]:
                    in_hr = True
                    scale = ((params.x.shape[2] * params.x.shape[3]) / (self.shape[2]*self.shape[3])) if self.scaling else 1
                else: 
                    in_hr = False
                    scale = 1
                ratios = self.ratios
                if ratios[-2] and in_hr: ratios[0] = ratios[-2]
                if ratios[-1] and in_hr: ratios[1] = ratios[-1]
                ratios[:2] = [x * scale for x in ratios[:2]]
                ratios = fineman(self.ratios)
                print(f"\nCD Tuner Effective : {ratios},in Hires-fix:{in_hr}")
                for i,name in enumerate(ADJUSTS):
                    if name not in self.storedweights.keys(): self.storedweights[name] = getset_nested_module_tensor(True, shared.sd_model, name).clone()
                    if 4 > i:
                        new_weight = self.storedweights[name] * ratios[i]
                    else:
                        new_weight = self.storedweights[name] +  torch.tensor(ratios[i]).to(devices.device)
                    getset_nested_module_tensor(False,shared.sd_model, name, new_tensor = new_weight)
                self.set = True
                self.shape = params.x.shape

def getset_nested_module_tensor(clone,model, tensor_path, new_tensor = None):
    sdmodules = tensor_path.split('.')
    target_module = model
    last_attr = None

    for module_name in sdmodules if clone else sdmodules[:-1]:
        if module_name.isdigit():
            target_module = target_module[int(module_name)] 
        else:
            target_module = getattr(target_module, module_name) 

    if clone : return target_module

    last_attr = sdmodules[-1]
    setattr(target_module, last_attr, Parameter(new_tensor)) 


def fineman(fine):
    fine = [
        1 - fine[0] * 0.01,
        1+ fine[0] * 0.02,
        1 - fine[1] * 0.01,
        1+ fine[1] * 0.02,
        [fine[2] * 0.02] + colorcalc(fine[3:6])
                ]
    return fine

def colorcalc(cols):
    outs = [[y * cols[i] * 0.02 for y in x] for i,x in enumerate(COLS)]
    return [sum(x) for x in zip(*outs)]

def fromprompts(prompt):
    _, extra_network_data = extra_networks.parse_prompts(prompt)
    params = extra_network_data["cdt"] if "cdt" in extra_network_data.keys() else None
    
    if params is None : return [None] * 9
    
    sets = params[0].items[0]

    outs = [None] * 9
    if sets:
        if any(x in sets for x in IDENTIFIER):
            sets = sets.split(";")
            for set in sets:
                if "=" in set:
                    set = set.split("=")
                    if set[0] in IDENTIFIER:
                        outs[IDENTIFIER.index(set[0])] = float(set[1])
        else:
            sets = [float(x) for x in sets.split(";")]
            outs[:len(sets)] = sets
    print(outs)
    return outs

ADJUSTS =[
"model.diffusion_model.input_blocks.0.0.weight",
"model.diffusion_model.input_blocks.0.0.bias",
"model.diffusion_model.out.0.weight",
"model.diffusion_model.out.0.bias",
"model.diffusion_model.out.2.bias",
]

IDENTIFIER = ["d1","d2","cont","col1","col2","col3","hd1","hd2","hrs"]

COLS = [[1,1,0],[0,-1,-1],[1,0,1]]