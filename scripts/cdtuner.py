import gradio as gr
import torch
from torch.nn import Parameter
import modules.ui
import modules
from modules import devices, shared, extra_networks
from modules.script_callbacks import CFGDenoiserParams, on_cfg_denoiser,CFGDenoisedParams, on_cfg_denoised

debug = False

class Script(modules.scripts.Script):   
    def __init__(self):
        self.active = False
        self.storedweights = {}
        self.shape = None
        self.done = [False,False]
        self.hr = 0

    def title(self):
        return "CD Tuner"

    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible

    infotext_fields = None
    paste_field_names = []

    def ui(self, is_img2img):      
        with gr.Accordion("CD Tuner", open=False):
            disable = gr.Checkbox(value=False, label="Disable",interactive=True,elem_id="cdt-disable")
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        d1 = gr.Slider(label="Detail(d1)", minimum=-10, maximum=10, value=0.0, step=0.1)
                        refresh_d1 = gr.Button(value='\U0001f504', elem_classes=["tool"])
                    with gr.Row():
                        d2 = gr.Slider(label="Detail 2(d2)", minimum=-10, maximum=10, value=0.0, step=0.1,visible = True)
                        refresh_d2 = gr.Button(value='\U0001f504', elem_classes=["tool"])
                with gr.Column():
                    with gr.Row():
                        hd1 = gr.Slider(label="hr-Detail(hd1)", minimum=-10, maximum=10, value=0.0, step=0.1)
                        refresh_hd1 = gr.Button(value='\U0001f504', elem_classes=["tool"])
                    with gr.Row():
                        hd2 = gr.Slider(label="hr-Detail 2(hd2)", minimum=-10, maximum=10, value=0.0, step=0.1,visible = True)
                        refresh_hd2 = gr.Button(value='\U0001f504', elem_classes=["tool"])
            with gr.Row():
                pass
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        cont1 = gr.Slider(label="Contrast(con1)", minimum=-20, maximum=20, value=0.0, step=0.1)
                        refresh_cont1 = gr.Button(value='\U0001f504', elem_classes=["tool"])
                    with gr.Row():
                        cont2 = gr.Slider(label="Contrast 2(con2)", minimum=-20, maximum=20, value=0.0, step=0.1)
                        refresh_cont2 = gr.Button(value='\U0001f504', elem_classes=["tool"])
                    with gr.Row():
                        bri = gr.Slider(label="Brightness(bri)", minimum=-20, maximum=20, value=0.0, step=0.1)
                        refresh_bri = gr.Button(value='\U0001f504', elem_classes=["tool"])
                with gr.Column():
                    with gr.Row():
                        col1 = gr.Slider(label="Cyan-Red(col1)", minimum=-20, maximum=20, value=0.0, step=0.1)
                        refresh_col1 = gr.Button(value='\U0001f504', elem_classes=["tool"])
                    with gr.Row():
                        col2 = gr.Slider(label="Magenta-Green(col2)", minimum=-20, maximum=20, value=0.0, step=0.1)
                        refresh_col2 = gr.Button(value='\U0001f504', elem_classes=["tool"])
                    with gr.Row():
                        col3 = gr.Slider(label="Yellow-Blue(col3)", minimum=-20, maximum=20, value=0.0, step=0.1)
                        refresh_col3 = gr.Button(value='\U0001f504', elem_classes=["tool"])

                scaling = gr.Checkbox(value=False, label="hr-scaling(hrs)",interactive=True,elem_id="cdt-hr-scaling")
                stop = gr.Slider(label="Stop Step", minimum=-1, maximum=20, value=-1, step=1)
                stoph = gr.Slider(label="Hr-Stop Step", minimum=-1, maximum=20, value=-1, step=1)

            def infotexter(text):
                if debug : print(text)
                text = text.split(",")
                if len(text) == 9:
                    outs = [float(x) for x in text[0:3]] +[0] * 2 + [float(x) for x in text[3:8]] + [text[-1] == "1"] + [-1] *2
                    outs.insert(3,0)
                elif len(text) == 13:
                    outs = [float(x) for x in text[:10]] + [text[10] == "1"] + [float(x) for x in text[11:]] 
                else:
                    outs = [0] * 10 + [False] + [-1] *2
                outs = outs + [False]
                return [gr.update(value = x) for x in outs]

            refresh_d1.click(fn=lambda x:gr.update(value = 0),outputs=[d1])
            refresh_d2.click(fn=lambda x:gr.update(value = 0),outputs=[d2])
            refresh_hd1.click(fn=lambda x:gr.update(value = 0),outputs=[hd1])
            refresh_hd2.click(fn=lambda x:gr.update(value = 0),outputs=[hd2])
            refresh_cont1.click(fn=lambda x:gr.update(value = 0),outputs=[cont1])
            refresh_cont2.click(fn=lambda x:gr.update(value = 0),outputs=[cont2])
            refresh_col1.click(fn=lambda x:gr.update(value = 0),outputs=[col1])
            refresh_col2.click(fn=lambda x:gr.update(value = 0),outputs=[col2])
            refresh_col3.click(fn=lambda x:gr.update(value = 0),outputs=[col3])
            refresh_bri.click(fn=lambda x:gr.update(value = 0),outputs=[bri])

            params = [d1,d2,cont1,cont2,bri,col1,col2,col3,hd1,hd2,scaling,stop,stoph,disable]

            allsets = gr.Textbox(visible = False)
            allsets.change(fn=infotexter,inputs = [allsets], outputs =params)

        self.infotext_fields = ([(allsets,"CDT")])
        self.paste_field_names.append("CDT")

        return params

    def process_batch(self, p, d1,d2,cont1,cont2,bri,col1,col2,col3,hd1,hd2,scaling,stop,stoph,disable,**kwargs):
        if (self.done[0] or self.done[1]) and self.storedweights and self.storedname == shared.opts.sd_model_checkpoint:
            restoremodel(self)

        allsets = [d1,d2,cont1,cont2,bri,col1,col2,col3,hd1,hd2,1 if scaling else 0,stop,stoph,0]

        self.__init__()

        psets = fromprompts(p.all_prompts[0:1])

        if disable:
            return
            
        for i, param in enumerate(psets):
            if param is not None:
                allsets[i] = param

        if debug: print("\n",allsets)

        if any(not(type(x) == float or type(x) == int) for x in allsets):return
        if all(x == 0 for x in allsets[:-3]):return
        else: 
            if debug:print("Start")
            self.active = True
            self.drratios = allsets[0:3]+allsets[8:10]
            self.ddratios= allsets[3:8]
            self.scaling = allsets[10]
            self.sts = allsets[11:]
            if hasattr(p,"enable_hr"): # Img2img doesn't have it.
                self.ehr = p.enable_hr
            else:
                self.ehr = False

            p.extra_generation_params.update({"CDT":",".join([str(x) for x in allsets[:-1]])})

            self.storedname = shared.opts.sd_model_checkpoint

        if not hasattr(self,"cdt_dr_callbacks"):
            self.cdt_dr_callbacks = on_cfg_denoiser(self.denoiser_callback)

        if not hasattr(self,"cdt_dd_callbacks"):
            self.cdt_dd_callbacks = on_cfg_denoised(self.denoised_callback)

    def postprocess_batch(self, p, *args,**kwargs):
        if True in self.done: restoremodel(self)
        self.done = [False,False]

    def denoiser_callback(self, params: CFGDenoiserParams):
        if self.active:
            if self.shape is None:self.shape = params.x.shape

            if params.x.shape[2] * params.x.shape[3] > self.shape[2]*self.shape[3]:
                self.hr = hr = 1
                scale = ((params.x.shape[2] * params.x.shape[3]) / (self.shape[2]*self.shape[3])) if self.scaling else 1
            else: 
                hr = 0
                scale = 1

            ratios = self.drratios

            if stopper(self,params.sampling_step,self.hr) : return

            if self.done[self.hr]:return
            else: self.done[hr] = True

            if hr:
                if ratios[-2] : ratios[0] = ratios[-2]
                if ratios[-1] : ratios[1] = ratios[-1]

            ratios[:2] = [x * scale for x in ratios[:2]]
            ratios = fineman(ratios)

            print(f"\nCD Tuner Effective : {ratios},in Hires-fix:{hr == 1}")

            for i,name in enumerate(ADJUSTS):
                if name not in self.storedweights.keys(): self.storedweights[name] = getset_nested_module_tensor(True, shared.sd_model, name).clone()
                if 4 > i:
                    new_weight = self.storedweights[name] * ratios[i]
                else:
                    new_weight = self.storedweights[name] +  torch.tensor(ratios[i]).to(devices.device)
                getset_nested_module_tensor(False,shared.sd_model, name, new_tensor = new_weight)
            
            self.shape = params.x.shape

    def denoised_callback(self, params: CFGDenoisedParams):
        if self.active:
            if self.ehr and not self.hr: return
            if params.sampling_step == params.total_sampling_steps-2 -self.sts[2]: 
                if any(x != 0 for x in self.ddratios):
                    ratios = [self.ddratios[0] * 0.02] +  colorcalc(self.ddratios[1:])
                    print(f"\nCD Tuner After Generation Effective: {ratios}")
                    for i, x in enumerate(ratios):
                        params.x[:,i,:,:] = params.x[:,i,:,:] - x * 20/3


def stopper(self,step,hr):
    judge = False
    if 0 > self.sts[hr]: return False 
    if step >= self.sts[hr]:
        judge = True
    if judge and self.done[hr]:
        restoremodel(self)
        self.done[hr] = False
    return judge


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

def restoremodel(self):
    for name in ADJUSTS:
        getset_nested_module_tensor(False,shared.sd_model, name, new_tensor = self.storedweights[name].clone())
    if debug:print("Restored")
    return

def fineman(fine):
    fine = [
        1 - fine[0] * 0.01,
        1+ fine[0] * 0.02,
        1 - fine[1] * 0.01,
        1+ fine[1] * 0.02,
        [fine[2] * 0.02,0,0,0]
                ]
    return fine

def colorcalc(cols):
    outs = [[y * cols[i] * 0.02 for y in x] for i,x in enumerate(COLS)]
    return [sum(x) for x in zip(*outs)]

def fromprompts(prompt):
    _, extra_network_data = extra_networks.parse_prompts(prompt)
    params = extra_network_data["cdt"] if "cdt" in extra_network_data.keys() else None
    
    outs = [None] * len(IDENTIFIER) 

    if params is None : return outs
    
    params = params[0].items[0]

    if params:
        if any(x in params for x in IDENTIFIER):
            params = params.split(";")
            for p in params:
                if "=" in p:
                    p = [x.strip() for x in p.split("=")]
                    if p[0] in IDENTIFIER:
                        if p[0] == "dis": return [0] * len(IDENTIFIER) 
                        outs[IDENTIFIER.index(p[0])] = float(p[1])
        else:
            params = [float(x) for x in params.split(";")]
            outs[:len(params)] = params
    if outs[10] is not None and outs[10] != 0: outs[10] = 1
    if debug : print(outs)
    return outs

ADJUSTS =[
"model.diffusion_model.input_blocks.0.0.weight",
"model.diffusion_model.input_blocks.0.0.bias",
"model.diffusion_model.out.0.weight",
"model.diffusion_model.out.0.bias",
"model.diffusion_model.out.2.bias",
]

IDENTIFIER = ["d1","d2","con1","con2","bri","col1","col2","col3","hd1","hd2","hrs","st1","st2","st3","dis"]

COLS = [[-1,1/3,2/3],[1,1,0],[0,-1,-1],[1,0,1]]
