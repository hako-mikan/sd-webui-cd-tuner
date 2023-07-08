import gradio as gr
import modules.ui
import torch
from modules import devices, shared
from modules.script_callbacks import (CFGDenoiserParams, on_cfg_denoiser)

ADJUSTS =[
"model.diffusion_model.input_blocks.0.0.weight",
"model.diffusion_model.input_blocks.0.0.bias",
"model.diffusion_model.out.0.weight",
"model.diffusion_model.out.0.bias",
"model.diffusion_model.out.2.weight",
"model.diffusion_model.out.2.bias",
]

LAVEL = ["CDT d1","CDT d2","CDT d3","CDT cont","CDT col1","CDT col2","CDT col3"]

from torch.nn import Parameter

def getset_nested_module_tensor(clone,model, tensor_path, new_tensor = None):
    modules = tensor_path.split('.')
    target_module = model
    last_attr = None

    for module_name in modules if clone else modules[:-1]:
        if module_name.isdigit():
            target_module = target_module[int(module_name)] 
        else:
            target_module = getattr(target_module, module_name) 

    if clone : return target_module

    last_attr = modules[-1]
    setattr(target_module, last_attr, Parameter(new_tensor)) 

def fineman(fine):
    fine = [
        1 - fine[0] * 0.01,
        1+ fine[0] * 0.02,
        1 - fine[1] * 0.01,
        1+ fine[1] * 0.02,
        1 - fine[2] * 0.01,
        [x*0.02 for x in fine[3:]]
                ]
    return fine

def lange(l):
    return range(len(l))

class Script(modules.scripts.Script):
    infotext_fields = None
    paste_field_names = []
    
    def __init__(self):
        self.storedweights = {}
        self.recoverd = True

    def title(self):
        return "CD Tuner"

    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible

    def ui(self, is_img2img):      
        with gr.Accordion("CD Tuner", open=False):
            with gr.Row():
                d1 = gr.Slider(label="Detail", minimum=-10, maximum=10, value=0, step=0.1)
                refresh_d1 = gr.Button(value='\U0001f504', elem_classes=["tool"])
            with gr.Row():
                d2 = gr.Slider(label="Detail2", minimum=-10, maximum=10, value=0, step=0.1,visible = True)
                refresh_d2 = gr.Button(value='\U0001f504', elem_classes=["tool"])
            with gr.Row():
                d3 = gr.Slider(label="Detail3", minimum=-10, maximum=10, value=0, step=0.1,visible = False)
                refresh_d3 = gr.Button(value='\U0001f504', elem_classes=["tool"],visible = False)
            with gr.Row():
                cont = gr.Slider(label="Contrast", minimum=-20, maximum=20, value=0, step=0.1)
                refresh_cont = gr.Button(value='\U0001f504', elem_classes=["tool"])
            with gr.Row():
                col1 = gr.Slider(label="Green-Purple", minimum=-20, maximum=20, value=0, step=0.1)
                refresh_col1 = gr.Button(value='\U0001f504', elem_classes=["tool"])
            with gr.Row():
                col2 = gr.Slider(label="Cyan-Red", minimum=-20, maximum=20, value=0, step=0.1)
                refresh_col2 = gr.Button(value='\U0001f504', elem_classes=["tool"])
            with gr.Row():
                col3 = gr.Slider(label="Yellow-Blue", minimum=-20, maximum=20, value=0, step=0.1)
                refresh_col3 = gr.Button(value='\U0001f504', elem_classes=["tool"])

            refresh_d1.click(fn=lambda x:gr.update(value = 0),outputs=[d1])
            refresh_d2.click(fn=lambda x:gr.update(value = 0),outputs=[d2])
            refresh_d3.click(fn=lambda x:gr.update(value = 0),outputs=[d3])
            refresh_cont.click(fn=lambda x:gr.update(value = 0),outputs=[cont])
            refresh_col1.click(fn=lambda x:gr.update(value = 0),outputs=[col1])
            refresh_col2.click(fn=lambda x:gr.update(value = 0),outputs=[col2])
            refresh_col3.click(fn=lambda x:gr.update(value = 0),outputs=[col3])

            params = [d1,d2,d3,cont,col1,col2,col3]

        self.infotext_fields = [(param,LAVEL[i]) for i, param in enumerate(params)]

        for _,name in self.infotext_fields:
            self.paste_field_names.append(name)
        return params

    def process(self, p, d1,d2,d3,cont,col1,col2,col3):
        if not self.recoverd and self.storedweights:
            for name in ADJUSTS:
                getset_nested_module_tensor(False,shared.sd_model, name, new_tensor = self.storedweights[name].clone())
        ratios = [d1,d2,d3,cont,col1,col2,col3]
        self.active = False
        if any(not(type(x) == float or type(x) == int) for x in ratios):return
        if all(x == 0 for x in ratios):return
        else: 
            self.active = True
            self.ratios = fineman(ratios)

            if not hasattr(self,"cdt_dr_callbacks"):
                self.cdt_dr_callbacks = on_cfg_denoiser(self.denoiser_callback)

            for name in ADJUSTS:
                model = shared.sd_model
                self.storedweights[name] = getset_nested_module_tensor(True,model,name).clone()
                self.storedname = shared.opts.sd_model_checkpoint

            for i, ratio in enumerate(ratios):
                p.extra_generation_params.update({LAVEL[i]:ratio})

    def postprocess(self, p, processed, *args):
        if self.active:
            for name in ADJUSTS:
                getset_nested_module_tensor(False,shared.sd_model, name, new_tensor = self.storedweights[name].clone())
            self.recoverd = True

    def denoiser_callback(self, params: CFGDenoiserParams):
        if self.active:
            for i,name in enumerate(ADJUSTS):
                if 5 > i:
                    new_weight = self.storedweights[name] * self.ratios[i]
                else:
                    new_weight = self.storedweights[name] +  torch.tensor(self.ratios[i]).to(devices.device)
                getset_nested_module_tensor(False,shared.sd_model, name, new_tensor = new_weight)
            self.recoverd = False