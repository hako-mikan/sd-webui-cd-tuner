import gradio as gr
import torch
import os
import numpy as np
import PIL
import json
import random
from torch.nn import Parameter
import modules.ui
import modules
from functools import wraps
from modules import devices, shared, extra_networks
from modules.script_callbacks import CFGDenoiserParams, on_cfg_denoiser,CFGDenoisedParams, on_cfg_denoised
from packaging import version

debug = False

CD_T = "customscript/cdtuner.py/txt2img/Active/value"
CD_I = "customscript/cdtuner.py/img2img/Active/value"
CONFIG = shared.cmd_opts.ui_config_file

DEFAULTC = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0]

# Check for Gradio version 4; see Forge architecture rework
IS_GRADIO_4 = version.parse(gr.__version__) >= version.parse("4.0.0")
# check if Forge or auto1111 pure; extremely hacky
IS_FORGE = hasattr(shared, "default_sd_model_file") and "webui-forge" in shared.default_sd_model_file

if os.path.exists(CONFIG):
    with open(CONFIG, 'r', encoding="utf-8") as json_file:
        ui_config = json.load(json_file)
else:
    print("ui config file not found, using default values")
    ui_config = {}

startup_t = ui_config[CD_T] if CD_T in ui_config else None
startup_i = ui_config[CD_I] if CD_I in ui_config else None
active_t = "Active" if startup_t else "Not Active"
active_i = "Active" if startup_i else "Not Active"

class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    @wraps(gr.Button.__init__)
    def __init__(self, value="", *args, elem_classes=None, **kwargs):
        elem_classes = elem_classes or []
        super().__init__(*args, elem_classes=["tool", *elem_classes], value=value, **kwargs)

    def get_block_name(self):
        return "button"

class Script(modules.scripts.Script):   
    def __init__(self):
        #Color/Detail
        self.active = False
        self.storedweights = {}
        self.storedweights_vae = {}
        self.storedweights_vae2 = {}
        self.shape = None
        self.done = [False,False]
        self.pas = 0
        self.colored = 0
        self.randman = None
        self.saturation = 0
        self.saturation2 = 0

        #Color map
        self.activec = False
        self.ocells = []
        self.icells = []
        self.cmode = ""
        self.colors =[]

    def title(self):
        return "CD Tuner"

    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible

    infotext_fields = None
    paste_field_names = []

    def ui(self, is_img2img):      
        resetsymbol = '\U0001F5D1\U0000FE0F'

        with gr.Accordion(f"CD Tuner : {active_i if is_img2img else active_t}",open = False) as acc:
            with gr.Row():
                active = gr.Checkbox(value=True, label="Active",interactive=True,elem_id="cdt-active")
                toggle = gr.Button(value=f"Toggle startup with Active(Now:{startup_i if is_img2img else startup_t})")
            with gr.Tab("Color/Detail"):
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            d1 = gr.Slider(label="Detail(d1)", minimum=-10, maximum=10, value=0.0, step=0.1)
                            refresh_d1 = ToolButton(value=resetsymbol)
                        with gr.Row():
                            d2 = gr.Slider(label="Detail 2(d2)", minimum=-10, maximum=10, value=0.0, step=0.1,visible = True)
                            refresh_d2 = ToolButton(value=resetsymbol)
                    with gr.Column():
                        with gr.Row():
                            hd1 = gr.Slider(label="hr-Detail(hd1)", minimum=-10, maximum=10, value=0.0, step=0.1)
                            refresh_hd1 = ToolButton(value=resetsymbol)
                        with gr.Row():
                            hd2 = gr.Slider(label="hr-Detail 2(hd2)", minimum=-10, maximum=10, value=0.0, step=0.1,visible = True)
                            refresh_hd2 = ToolButton(value=resetsymbol)
                with gr.Row():
                    pass
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            cont1 = gr.Slider(label="Contrast(con1)", minimum=-20, maximum=20, value=0.0, step=0.1)
                            refresh_cont1 = ToolButton(value=resetsymbol)
                        with gr.Row():
                            cont2 = gr.Slider(label="Contrast 2(con2)", minimum=-20, maximum=20, value=0.0, step=0.1)
                            refresh_cont2 = ToolButton(value=resetsymbol)
                        with gr.Row():
                            bri = gr.Slider(label="Brightness(bri)", minimum=-20, maximum=20, value=0.0, step=0.1)
                            refresh_bri = ToolButton(value=resetsymbol)
                        with gr.Row():
                            sat = gr.Slider(label="saturation(sat)", minimum=-20, maximum=20, value=0.0, step=0.1)
                            refresh_sat = ToolButton(value=resetsymbol)
                    with gr.Column():
                        with gr.Row():
                            col1 = gr.Slider(label="Cyan-Red(col1)", minimum=-20, maximum=20, value=0.0, step=0.1)
                            refresh_col1 = ToolButton(value=resetsymbol)
                        with gr.Row():
                            col2 = gr.Slider(label="Magenta-Green(col2)", minimum=-20, maximum=20, value=0.0, step=0.1)
                            refresh_col2 = ToolButton(value=resetsymbol)
                        with gr.Row():
                            col3 = gr.Slider(label="Yellow-Blue(col3)", minimum=-20, maximum=20, value=0.0, step=0.1)
                            refresh_col3 = ToolButton(value=resetsymbol)
                        with gr.Row():
                            sat2 = gr.Slider(label="saturation2(sat2)", minimum=-20, maximum=20, value=0.0, step=0.1)
                            refresh_sat2 = ToolButton(value=resetsymbol)

                    scaling = gr.Checkbox(value=False, label="hr-scaling(hrs)",interactive=True,elem_id="cdt-hr-scaling")
                    stop = gr.Slider(label="Stop Step", minimum=-1, maximum=20, value=-1, step=1)
                    stoph = gr.Slider(label="Hr-Stop Step", minimum=-1, maximum=20, value=-1, step=1)
                with gr.Row():
                    opts = gr.CheckboxGroup(choices=["Apply once"],show_label = False)

            with gr.Tab("Color Map"):
                with gr.Row():
                    cmode = gr.Radio(label="Split by", choices=["Horizontal","Vertical"], value="Horizontal", type="value", interactive=True)
                    ratios = gr.Textbox(label="Split Ratio",lines=1,value="1,1",interactive=True,elem_id="Split_ratio",visible=True)
                with gr.Row():
                    with gr.Column():
                        maketemp = gr.Button(value="Visualize Map")
                        colors = gr.Textbox(label="colors",interactive=True,visible=True)
                        fst = gr.Slider(label="Mapping Stop Step", minimum=0, maximum=150, value=2, step=1)
                        att = gr.Slider(label="Strength", minimum=0, maximum=2, value=1, step=0.1)
                        presets = gr.Dropdown(label="add from presets", choices=list(COLORPRESET.keys()))
                    
                    presets.change(fn=lambda x, y:COLORPRESET[y] if x == "" else x + ";"+ COLORPRESET[y], inputs =[colors,presets], outputs=[colors])
                    
                    with gr.Column():
                        areasimg = gr.Image(type="pil", show_label = False, height = 256, width = 256)
                    
                    dtrue =  gr.Checkbox(value = True, visible = False)                
                    dfalse =  gr.Checkbox(value = False,visible = False)     

                maketemp.click(fn=makecells, inputs =[ratios,cmode,dtrue],outputs = [areasimg])

            def infotexter(text):
                if text == "":return [gr.update()] * (len(params) +1)
                if debug : print(text)
                text = text.split(",")
                if len(text) == 9:
                    outs = [float(x) for x in text[0:3]] +[0] * 2 + [float(x) for x in text[3:8]] + [text[-1] == "1"] + [-1] *2 + [0,0]
                    outs.insert(3,0)
                elif len(text) == 13:
                    outs = [float(x) for x in text[:10]] + [text[10] == "1"] + [float(x) for x in text[11:]] + [0,0]
                elif len(text) == 14:
                    outs = [float(x) for x in text[:10]] + [text[10] == "1"] + [float(x) for x in text[11:]] + [0]
                elif len(text) == 15:
                    outs = [float(x) for x in text[:10]] + [text[10] == "1"] + [float(x) for x in text[11:]] 
                else:
                    outs = [0] * 10 + [False] + [-1] *2 + [0, 0]
                outs = [""] + [True] + outs
                return [gr.update(value = x) for x in outs]

            def infotexter_c(text):
                if text == "":return [gr.update()] * (len(paramsc) +1)
                if debug : print(text)
                text = text.split("_")
                if len(text) != 5:
                    text = ["","Horizonal","",2,0.2]
                if debug : print(text)
                text = [""] + text
                return [gr.update(value = x) for x in text]

            refresh_d1.click(fn=lambda:gr.update(value = 0),outputs=[d1], show_progress=False)
            refresh_d2.click(fn=lambda:gr.update(value = 0),outputs=[d2], show_progress=False)
            refresh_hd1.click(fn=lambda:gr.update(value = 0),outputs=[hd1], show_progress=False)
            refresh_hd2.click(fn=lambda:gr.update(value = 0),outputs=[hd2], show_progress=False)
            refresh_cont1.click(fn=lambda:gr.update(value = 0),outputs=[cont1], show_progress=False)
            refresh_cont2.click(fn=lambda:gr.update(value = 0),outputs=[cont2], show_progress=False)
            refresh_col1.click(fn=lambda:gr.update(value = 0),outputs=[col1], show_progress=False)
            refresh_col2.click(fn=lambda:gr.update(value = 0),outputs=[col2], show_progress=False)
            refresh_col3.click(fn=lambda:gr.update(value = 0),outputs=[col3], show_progress=False)
            refresh_bri.click(fn=lambda:gr.update(value = 0),outputs=[bri], show_progress=False)
            refresh_sat.click(fn=lambda:gr.update(value = 0),outputs=[sat], show_progress=False)
            refresh_sat2.click(fn=lambda:gr.update(value = 0),outputs=[sat2], show_progress=False)

            params = [active,d1,d2,cont1,cont2,bri,col1,col2,col3,hd1,hd2,scaling,stop,stoph,sat,sat2]
            paramsc = [ratios,cmode,colors,fst,att]

            allsets = gr.Textbox(visible = False)
            allsets.change(fn=infotexter,inputs = [allsets], outputs =[allsets] + params)

            allsets_c = gr.Textbox(visible = False)
            allsets_c.change(fn=infotexter_c,inputs = [allsets_c], outputs =[allsets_c] +paramsc)

            def f_toggle(is_img2img):
                key = CD_I if is_img2img else CD_T

                with open(CONFIG, 'r', encoding="utf-8") as json_file:
                    data = json.load(json_file)
                data[key] = not data[key]

                with open(CONFIG, 'w', encoding="utf-8") as json_file:
                    json.dump(data, json_file, indent=4) 

                return gr.update(value = f"Toggle startup Active(Now:{data[key]})")

            toggle.click(fn=f_toggle,inputs=[gr.Checkbox(value = is_img2img, visible = False)],outputs=[toggle])
            active.change(fn=lambda x:gr.update(label = f"CD Tuner : {'Active' if x else 'Not Active'}"),inputs=active, outputs=[acc])


        self.infotext_fields = ([(allsets,"CDT"),(allsets_c,"CDTC")])
        self.paste_field_names.append("CDT")
        self.paste_field_names.append("CDTC")

        return params + paramsc + [opts]

    def process_batch(self, p, active, d1,d2,cont1,cont2,bri,col1,col2,col3,hd1,hd2,scaling,stop,stoph,sat,sat2,ratios,cmode,colors,fst,att,opts,**kwargs):
        if (self.done[0] or self.done[1]) and self.storedweights and self.storedname == shared.opts.sd_model_checkpoint:
            restoremodel(self)

                # ["d1","d2","con1","con2","bri","col1","col2","col3","hd1","hd2","hrs","st1","st2","dis","sat"]
                     #0  1   2       3        4   5     6     7     8     9    10                      11    12    13 14
        allsets = [d1,d2,cont1,cont2,bri,col1,col2,col3,hd1,hd2,1 if scaling else 0,stop,stoph,0,sat,sat2]
        allsets_c = [ratios,cmode,colors,fst,att]
        self.opts = opts

        self.__init__()

        if not active:
            return

        psets, psets_c = fromprompts(p.all_prompts[0:1])

        for i, param in enumerate(psets):
            if param is not None:
                allsets[i] = param
        
        for i, param in enumerate(psets_c):
            if param is not None:
                if param == "H" :param ="Horizontal"
                if param == "V" :param ="Vertical"
                allsets_c[i] = param

        ratios, cmode, colors, fst, att = allsets_c

        if debug: print("\n",allsets)
        if debug: print("\n",allsets_c)

        self.isxl = hasattr(shared.sd_model,"conditioner")

        self.isrefiner = getattr(p, "refiner_switch_at") is not None

        if any(not(type(x) == float or type(x) == int) for x in allsets):return
        if allsets == DEFAULTC and (att == 0 or colors == ""):return
        else:
            if not all(x == 0 for x in allsets[:10] + allsets[14:16]):
                if debug:print("Start")
                self.active = True
                self.drratios = allsets[0:3]+allsets[8:10]
                self.ddratios= allsets[3:8]
                self.scaling = allsets[10]
                self.sts = allsets[11:14]
                self.saturation = allsets[14]
                self.saturation2 = allsets[15]
                if hasattr(p,"enable_hr"): # Img2img doesn't have it.
                    self.hr = p.enable_hr
                else:
                    self.hr = False

                p.extra_generation_params.update({"CDT":",".join([str(x) for x in allsets[:-2] + [allsets[-1]]])})

                self.storedname = shared.opts.sd_model_checkpoint

            if colors != "":
                self.activec = True
                if not ("Hor" in cmode or "Ver" in cmode) :cmode = "Vertical"
                self.cmode = cmode
                self.fst = int(fst)
                self.att = float(att) * 0.5
                self.batch = p.batch_size
                self.ocells, self.icells = makecells(ratios, cmode,False)
                colors = colors.split("|") if "|" in colors else colors.split(";")
                for i, cols in enumerate(colors):
                    if " " in cols:colors[i] = cols.split(" ")
                    if "," in cols:colors[i] = cols.split(",")
                for cols in colors:
                     self.colors.append([float(x) for x in cols])
                for cols in self.colors:
                    if len(cols) ==3:
                        cols.insert(0,sum(cols)/3)

                if 5 > self.fst :
                    self.satt = 1.15 - self.fst * 0.15
                else:
                    self.satt = 0.4

                p.extra_generation_params.update({"CDTC":"_".join([str(x) for x in allsets_c])})

            if self.saturation != 0:
                vaedealer(self)

            if self.saturation2 != 0:
                vaedealer2(self)

        print(f"\nCD Tuner Effective : {allsets}")

        if not hasattr(self,"cdt_dr_callbacks"):
            self.cdt_dr_callbacks = on_cfg_denoiser(self.denoiser_callback)

        if not hasattr(self,"cdt_dd_callbacks"):
            self.cdt_dd_callbacks = on_cfg_denoised(self.denoised_callback)

    def postprocess_batch(self, p, *args,**kwargs):
        if True in self.done: 
            restoremodel_l(shared.sd_model.forge_objects_after_applying_lora.unet.model if IS_FORGE else shared.sd_model)
            restoremodel(self)
        if self.saturation != 0:
            vaeunloader(self)
        if self.saturation2 != 0:
            vaeunloader2(self)
        self.done = [False,False]
        if "Apply once" in self.opts:
            self.active = self.activec = False

    def denoiser_callback(self, params: CFGDenoiserParams):
        # params.x [batch,ch[4],height,width]
        #print(self.activec,self.colors,self.ocells,self.icells,params.sampling_step)
        if self.activec:
            if self.shape is None:self.shape = params.x.shape
            if params.x.shape[2] * params.x.shape[3] > self.shape[2]*self.shape[3]:
                self.colored = 0
                self.pas = 1
            if self.colored == params.sampling_step and self.colored < self.fst:
                c = 0
                scale = torch.mean(torch.abs(params.x[:,:,:,:]))
                h,w = params.x.shape[2], params.x.shape[3]
                enhance = 6
                hr_att = 0.25 if self.pas else 1
                for i,ocell in enumerate(self.ocells):
                    for icell in self.icells[i]:
                        if "Ver" in self.cmode:
                            s3 = slice(int(h*ocell[0]), int(h*ocell[1]))
                            s4 = slice(int(w*icell[0]), int(w*icell[1]))
                        elif "Hor" in self.cmode:
                            s3 = slice(int(h*icell[0]),int(h*icell[1]))
                            s4 = slice(int(w*ocell[0]),int(w*ocell[1]))

                        for s2 in range(1,4):
                            scale = torch.mean(torch.abs(params.x[:,s2,:,:])) 
                            cratio =(sum(abs(x * 50) for x in colorcalc(self.colors[c],self.isxl)))/10 * (1/(1+(1 + params.sampling_step)**1.5/10)) * self.att * hr_att * self.satt
                            if 0 > cratio : continue
                            params.x[:-self.batch,s2,s3,s4] =(1 - cratio) * params.x[:-self.batch,s2,s3,s4] - colorcalc(self.colors[c],self.isxl)[s2-1]*enhance * scale * cratio
                            params.x[-self.batch:,s2,s3,s4] =(1 - cratio) * params.x[-self.batch:,s2,s3,s4] + colorcalc(self.colors[c],self.isxl)[s2-1]*enhance * scale * cratio
                            
                        c += 1
                self.colored += 1

        if self.active:
            if debug: print(self.drratios)
            if self.shape is None:self.shape = params.x.shape
            if params.x.shape[2] * params.x.shape[3] > self.shape[2]*self.shape[3]:
                self.pas = 1
                scale = ((params.x.shape[2] * params.x.shape[3]) / (self.shape[2]*self.shape[3])) if self.scaling else 1
            else: 
                scale = 1

            ratios = self.drratios

            if stopper(self,self.pas,params.sampling_step): return

            #dont operate twice
            if self.done[self.pas] and not self.isrefiner:
                return
            else:
                self.done[self.pas] = True

            if self.pas:
                if ratios[-2] : ratios[0] = ratios[-2]
                if ratios[-1] : ratios[1] = ratios[-1]

            ratios[:2] = [x * scale for x in ratios[:2]]
            ratios = fineman(ratios)
            if debug: print(ratios)
            for i,name in enumerate(ADJUSTS):
                if name not in self.storedweights.keys() or self.isrefiner:
                    self.storedweights[name] = getset_nested_module_tensor(True, shared.sd_model, name).clone()
                if 4 > i:
                    dtype = self.storedweights[name].dtype
                    new_weight = self.storedweights[name].to(devices.device, devices.dtype) * torch.tensor(ratios[i]).to(devices.device,devices.dtype)
                    if dtype == torch.float8_e4m3fn:
                        self.storedweights[name] = self.storedweights[name].to(dtype)
                    else:
                        self.storedweights[name] = self.storedweights[name].to(devices.dtype)
                else:
                    dtype = self.storedweights[name].dtype
                    new_weight = self.storedweights[name].to(devices.device,devices.dtype) +  torch.tensor(ratios[i]).to(devices.device,devices.dtype)
                    if dtype == torch.float8_e4m3fn:
                        self.storedweights[name] = self.storedweights[name].to(dtype)
                    else:
                        self.storedweights[name] = self.storedweights[name].to(devices.dtype)
                getset_nested_module_tensor(False,shared.sd_model, name, new_tensor = new_weight)
            
            self.shape = params.x.shape


    def denoised_callback(self, params: CFGDenoisedParams):
        if self.active:
            if self.isrefiner:
                restoremodel_l(shared.sd_model.forge_objects_after_applying_lora.unet.model if IS_FORGE else shared.sd_model)
                restoremodel(self)
            if self.hr and not self.pas: return
            if params.sampling_step == params.total_sampling_steps-2 -self.sts[2]: 
                if any(x != 0 for x in self.ddratios):
                    ratios = [self.ddratios[0] * 0.02] +  colorcalc(self.ddratios[1:],self.isxl)
                    print(f"\nCD Tuner After Generation Effective: {ratios}")
                    for i, x in enumerate(ratios):
                        params.x[:,i,:,:] = params.x[:,i,:,:] - x * 20/3

def vaedealer(self):
    for name in VAEKEYS:
        if name not in self.storedweights_vae:
            self.storedweights_vae[name] = getset_nested_module_tensor(True, shared.sd_model, name).clone()
        new_weight = self.storedweights_vae[name].to(devices.device) * (1 + self.saturation * 0.075) 
        getset_nested_module_tensor(False,shared.sd_model, name, new_tensor = new_weight)

def vaeunloader(self):
    for name in VAEKEYS and self.storedweights_vae:
        getset_nested_module_tensor(False,shared.sd_model, name, new_tensor = self.storedweights_vae[name].clone().to(devices.device) )

def vaedealer2(self):
    for name in VAEKEYS2:
        if name not in self.storedweights_vae2:
            self.storedweights_vae2[name] = getset_nested_module_tensor(True, shared.sd_model, name).clone()
        new_weight = self.storedweights_vae2[name].to(devices.device) * (1 + self.saturation2 * 0.02) 
        getset_nested_module_tensor(False,shared.sd_model, name, new_tensor = new_weight)

def vaeunloader2(self):
    for name in VAEKEYS2 and self.storedweights_vae2:
        getset_nested_module_tensor(False,shared.sd_model, name, new_tensor = self.storedweights_vae2[name].clone().to(devices.device) )

def stopper(self,pas,step):
    judge = False
    if 0 > self.sts[pas]: return False 
    if step >= self.sts[pas]:
        judge = True
    if judge and self.done[pas]:
        restoremodel_l(shared.sd_model.forge_objects_after_applying_lora.unet.model if IS_FORGE else shared.sd_model)
        restoremodel(self)
        self.done[pas] = False
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
        getset_nested_module_tensor(False,shared.sd_model, name, new_tensor = self.storedweights[name].clone().to(devices.device))
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

def colorcalc(cols,isxl):
    colors = COLSXL if isxl else COLS
    outs = [[y * cols[i] * 0.02 for y in x] for i,x in enumerate(colors)]
    return [sum(x) for x in zip(*outs)]

def fromprompts(prompt):
    _, extra_network_data = extra_networks.parse_prompts(prompt)
    params = extra_network_data["cdt"] if "cdt" in extra_network_data.keys() else None
    params_c = extra_network_data["cdtc"] if "cdtc" in extra_network_data.keys() else None

    outs = [None] * len(IDENTIFIER) 
    outs_c = [None] * len(IDENTIFIER_C) 

    if params is None and params_c is None : return outs, outs_c
    
    if params:
        params = params[0].items[0]
        if any(x in params for x in IDENTIFIER):
            params = params.split(";")
            for p in params:
                if "=" in p:
                    p = [x.strip() for x in p.split("=")]
                    if p[0] in IDENTIFIER:
                        if p[0] == "dis": return [0] * len(IDENTIFIER), outs_c
                        outs[IDENTIFIER.index(p[0])] = float(p[1])
        else:
            params = [float(x) for x in params.split(";")]
            outs[:len(params)] = params

    if params_c:
        params_c = params_c[0].items[0]
        if any(x in params_c for x in IDENTIFIER_C):
            params_c = params_c.split(";")
            for p in params_c:
                if "=" in p:
                    p = [x.strip() for x in p.split("=")]
                    if p[0] in IDENTIFIER_C:
                        outs_c[IDENTIFIER_C.index(p[0])] = p[1]
        else:
            params_c = [x for x in params_c.split(";")]
            outs_c[:len(params_c)] = params_c

    if outs[10] is not None and outs[10] != 0: outs[10] = 1
    if debug : print(outs)
    if debug : print(outs_c)
    return outs, outs_c

def lange(l):
    return range(len(l))

def makecells(aratio,mode,in_ui):
    if not ("Hor" in mode or "Ver" in mode) :mode = "Vertical"
    aratio = aratio.replace(",", " ")
    aratios = aratio.split("|") if "|" in aratio else aratio.split(";")
    if len(aratios) == 1 : aratios[0] = "1 " + aratios[0]
    h = w = 128
    icells = []
    ocells = []

    def startend(lst):
        o = []
        s = 0
        lst = [l/sum(lst) for l in lst]
        for i in lange(lst):
            if i == 0 :o.append([0,lst[0]])
            else : o.append([s, s + lst[i]])
            s = s + lst[i]
        return o

    for rc in aratios:
        rc = rc.split(" ")
        rc = [float(r) for r in rc]
        if len(rc) == 1 : rc = [rc[0]]*2
        ocells.append(rc[0])
        icells.append(startend(rc[1:]))

    fx = np.zeros((h,w, 3), np.uint8)
    ocells = startend(ocells)

    c = 0
    for i,ocell in enumerate(ocells):
        for icell in icells[i]:
            if "Vertical" in mode:
                fx[int(h*ocell[0]):int(h*ocell[1]),int(w*icell[0]):int(w*icell[1]),:] = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]
            elif "Horizontal" in mode: 
                fx[int(h*icell[0]):int(h*icell[1]),int(w*ocell[0]):int(w*ocell[1]),:] = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]
            c += 1
    img = PIL.Image.fromarray(fx)
    draw = PIL.ImageDraw.Draw(img)
    c = 0
    def wbdealer(col):
        if sum(col) > 380:return "black"
        else:return "white"

    for i,ocell in enumerate(ocells):
        for j,icell in enumerate(icells[i]):
            if "Vertical" in mode:
                draw.text((int(w*icell[0]),int(h*ocell[0])),f"{c}",wbdealer(fx[int(h*ocell[0]),int(w*icell[0])]))
            elif "Horizontal" in mode: 
                draw.text((int(w*ocell[0]),int(h*icell[0])),f"{c}",wbdealer(fx[int(h*icell[0]),int(w*ocell[0])]))
            c += 1
    if in_ui:
        return img
    else:
        print(ocells,icells)
        return ocells, icells


def latentfromrgb(rgb):
    outs = [[y * rgb[i] for y in x] for i,x in enumerate(COLS[1:])]
    return [sum(x) for x in zip(*outs)]

forge_prefix = "forge_objects_after_applying_lora.unet." if IS_FORGE else ""

ADJUSTS =[
f"{forge_prefix}model.diffusion_model.input_blocks.0.0.weight",
f"{forge_prefix}model.diffusion_model.input_blocks.0.0.bias",
f"{forge_prefix}model.diffusion_model.out.0.weight",
f"{forge_prefix}model.diffusion_model.out.0.bias",
f"{forge_prefix}model.diffusion_model.out.2.bias",
]

NAMES =[
"model.diffusion_model.input_blocks.0.0",
"model.diffusion_model.out.0",
"model.diffusion_model.out.2",
]

IDENTIFIER = ["d1","d2","con1","con2","bri","col1","col2","col3","hd1","hd2","hrs","st1","st2","dis","sat"]
IDENTIFIER_C = ["sp","md","cols","stc","str"]


COLS = [[-1,1/3,2/3],[1,1,0],[0,-1,-1],[1,0,1]]
COLSXL = [[0,0,1],[1,0,0],[-1,-1,0],[-1,1,0]]

def restoremodel_l(model):
    for name, module in model.named_modules():
        if name not in NAMES: continue
        if hasattr(module, "network_weights_backup"):
            if module.network_weights_backup is not None:
                if isinstance(module, torch.nn.MultiheadAttention):
                    module.in_proj_weight.copy_(module.network_weights_backup[0])
                    module.out_proj.weight.copy_(module.network_weights_backup[1])
                else:
                    module.weight.copy_(module.network_weights_backup)
                del module.network_weights_backup
                if hasattr(module, "network_current_names"): del module.network_current_names

        if hasattr(module, "network_bias_backup"):
            if module.network_bias_backup is not None:
                if isinstance(module, torch.nn.MultiheadAttention):
                    module.in_proj_bias.copy_(module.network_bias_backup[0])
                    module.out_proj.bias.copy_(module.network_bias_backup[1])
                else:
                    module.bias.copy_(module.network_bias_backup)
                del module.network_bias_backup
                if hasattr(module, "network_current_names"): del module.network_current_names


COLORPRESET = {
    "Cyan": "-5, 0, 0",
    "Magenda": "0, -5, 0",
    "Yellow": "0, 0, -5",
    "Red": "5, -5, -5",
    "Green": "0, 5, 0",
    "Blue": "0, 0, 5",
    "VeryLightBlue": "-5, -5, 0",
    "YellowGreen": "-5, 0, -5",
    "Orange": "0, -5, -5",
    "Malachite": "-5, 5, 0",
    "BrightCyan": "-5, 0, 5",
    "VioretPink": "0, -5, 5",
    "DeepPink": "5, -5, 0",
    "GuardsmanRed": "5, 0, -5",
    "DeepGreen": "0, 5, -5",
    "Black": "5, 5, 0",
    "BrightNeonPink": "5, 0, 5",
    "NeonBlue": "0, 5, 5",
    "GreenYellow": "-5, -5, -5",
    "LightBlue": "-5, -5, 5",
    "LimeGreen": "-5, 5, -5",
    "DeepBrown": "5, 5, -5",
    "VioletBlue": "5, 5, 5"
}

VAEKEYS = [
"first_stage_model.decoder.up.1.upsample.conv.weight",
"first_stage_model.decoder.up.0.block.0.nin_shortcut.weight",
]

VAEKEYS2 = [
"first_stage_model.decoder.up.3.upsample.conv.weight",
"first_stage_model.decoder.up.2.upsample.conv.weight",
"first_stage_model.decoder.up.1.block.0.nin_shortcut.weight",
"first_stage_model.decoder.conv_in.weight",
"first_stage_model.decoder.conv_out.weight",
]


# Forge patches

# See discussion at, class versus instance __module__
# https://github.com/LEv145/--sd-webui-ar-plus/issues/24
# Hack for Forge with Gradio 4.0; see `get_component_class_id` in `venv/lib/site-packages/gradio/components/base.py`
if IS_GRADIO_4:
    ToolButton.__module__ = "modules.ui_components"
