import gradio as gr
import torch
import numpy as np
import PIL
import random
from torch.nn import Parameter
import modules.ui
import modules
from modules import devices, shared, extra_networks
from modules.script_callbacks import CFGDenoiserParams, on_cfg_denoiser,CFGDenoisedParams, on_cfg_denoised

debug = False


class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool",
                         elem_classes=kwargs.pop('elem_classes', []),
                         **kwargs)

    def get_block_name(self):
        return "button"


class Script(modules.scripts.Script):   
    def __init__(self):
        self.active = False
        self.storedweights = {}
        self.shape = None
        self.done = [False,False]
        self.hr = 0
        self.colored = 0
        self.randman = None

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

        with gr.Accordion("CD Tuner", open=False):
            disable = gr.Checkbox(value=False, label="Disable",interactive=True,elem_id="cdt-disable")
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

                    scaling = gr.Checkbox(value=False, label="hr-scaling(hrs)",interactive=True,elem_id="cdt-hr-scaling")
                    stop = gr.Slider(label="Stop Step", minimum=-1, maximum=20, value=-1, step=1)
                    stoph = gr.Slider(label="Hr-Stop Step", minimum=-1, maximum=20, value=-1, step=1)

            with gr.Tab("Color Map"):
                with gr.Row():
                    cmode = gr.Radio(label="Split by", choices=["Horizonal","Vertical"], value="Horizontal", type="value", interactive=True)
                    ratios = gr.Textbox(label="Split Ratio",lines=1,value="1,1",interactive=True,elem_id="Split_ratio",visible=True)
                with gr.Row():
                    with gr.Column():
                        maketemp = gr.Button(value="Visualize Map")
                        colors = gr.Textbox(label="colors",interactive=True,visible=True)
                        fst = gr.Slider(label="Mapping Stop Step", minimum=0, maximum=150, value=2, step=1)
                        att = gr.Slider(label="Strength", minimum=0, maximum=2, value=1, step=0.1)
                    
                    with gr.Column():
                        areasimg = gr.Image(type="pil", show_label = False, height = 256, width = 256)
                    
                    dtrue =  gr.Checkbox(value = True, visible = False)                
                    dfalse =  gr.Checkbox(value = False,visible = False)     

                maketemp.click(fn=makecells, inputs =[ratios,cmode,dtrue],outputs = [areasimg])

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

            def infotexter_c(text):
                if debug : print(text)
                text = text.split("_")
                if len(text) != 5:
                    text = ["","Horizonal","",2,0.2]
                if debug : print(text)
                return [gr.update(value = x) for x in text]

            refresh_d1.click(fn=lambda x:gr.update(value = 0),outputs=[d1], show_progress=False)
            refresh_d2.click(fn=lambda x:gr.update(value = 0),outputs=[d2], show_progress=False)
            refresh_hd1.click(fn=lambda x:gr.update(value = 0),outputs=[hd1], show_progress=False)
            refresh_hd2.click(fn=lambda x:gr.update(value = 0),outputs=[hd2], show_progress=False)
            refresh_cont1.click(fn=lambda x:gr.update(value = 0),outputs=[cont1], show_progress=False)
            refresh_cont2.click(fn=lambda x:gr.update(value = 0),outputs=[cont2], show_progress=False)
            refresh_col1.click(fn=lambda x:gr.update(value = 0),outputs=[col1], show_progress=False)
            refresh_col2.click(fn=lambda x:gr.update(value = 0),outputs=[col2], show_progress=False)
            refresh_col3.click(fn=lambda x:gr.update(value = 0),outputs=[col3], show_progress=False)
            refresh_bri.click(fn=lambda x:gr.update(value = 0),outputs=[bri], show_progress=False)

            params = [d1,d2,cont1,cont2,bri,col1,col2,col3,hd1,hd2,scaling,stop,stoph,disable]
            paramsc = [ratios,cmode,colors,fst,att]

            allsets = gr.Textbox(visible = False)
            allsets.change(fn=infotexter,inputs = [allsets], outputs =params)

            allsets_c = gr.Textbox(visible = False)
            allsets_c.change(fn=infotexter_c,inputs = [allsets_c], outputs =paramsc)

        self.infotext_fields = ([(allsets,"CDT"),(allsets_c,"CDTC")])
        self.paste_field_names.append("CDT")
        self.paste_field_names.append("CDTC")

        return params + paramsc

    def process_batch(self, p, d1,d2,cont1,cont2,bri,col1,col2,col3,hd1,hd2,scaling,stop,stoph,disable,ratios,cmode,colors,fst,att,**kwargs):
        if (self.done[0] or self.done[1]) and self.storedweights and self.storedname == shared.opts.sd_model_checkpoint:
            restoremodel(self)

        allsets = [d1,d2,cont1,cont2,bri,col1,col2,col3,hd1,hd2,1 if scaling else 0,stop,stoph,0]
        allsets_c = [ratios,cmode,colors,fst,att]

        self.__init__()

        psets, psets_c = fromprompts(p.all_prompts[0:1])

        if disable:
            return
            
        for i, param in enumerate(psets):
            if param is not None:
                allsets[i] = param
        
        for i, param in enumerate(psets_c):
            if param is not None:
                if param == "H" :param ="Horizontal"
                if param == "V" :param ="Vertical"
                allsets_c[i] = param

        ratios,cmode,colors,fst,att = allsets_c

        if debug: print("\n",allsets)
        if debug: print("\n",allsets_c)

        self.isxl = hasattr(shared.sd_model,"conditioner")

        if any(not(type(x) == float or type(x) == int) for x in allsets):return
        if all(x == 0 for x in allsets[:-3]) and all(x == "" for x in [ratios,cmode,colors]):return
        else:
            if not all(x == 0 for x in allsets[:-3]):
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

        if not hasattr(self,"cdt_dr_callbacks"):
            self.cdt_dr_callbacks = on_cfg_denoiser(self.denoiser_callback)

        if not hasattr(self,"cdt_dd_callbacks"):
            self.cdt_dd_callbacks = on_cfg_denoised(self.denoised_callback)

    def postprocess_batch(self, p, *args,**kwargs):
        if True in self.done: restoremodel(self)
        self.done = [False,False]

    def denoiser_callback(self, params: CFGDenoiserParams):
        # params.x [batch,ch[4],height,width]
        #print(self.activec,self.colors,self.ocells,self.icells,params.sampling_step)
        if self.activec:
            if self.shape is None:self.shape = params.x.shape
            if params.x.shape[2] * params.x.shape[3] > self.shape[2]*self.shape[3]:
                self.colored = 0
                self.hr = 1
            if self.colored == params.sampling_step and self.colored < self.fst:
                c = 0
                scale = torch.mean(torch.abs(params.x[:,:,:,:]))
                h,w = params.x.shape[2], params.x.shape[3]
                enhance = 6
                hr_att = 0.25 if self.hr else 1
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
                    new_weight = self.storedweights[name].to(devices.device) * ratios[i]
                else:
                    new_weight = self.storedweights[name].to(devices.device) +  torch.tensor(ratios[i]).to(devices.device)
                getset_nested_module_tensor(False,shared.sd_model, name, new_tensor = new_weight)
            
            self.shape = params.x.shape

    def denoised_callback(self, params: CFGDenoisedParams):
        if self.active:
            if self.ehr and not self.hr: return
            if params.sampling_step == params.total_sampling_steps-2 -self.sts[2]: 
                if any(x != 0 for x in self.ddratios):
                    ratios = [self.ddratios[0] * 0.02] +  colorcalc(self.ddratios[1:],self.isxl)
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
                        if p[0] == "dis": return [0] * len(IDENTIFIER) 
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
    colors = []

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

ADJUSTS =[
"model.diffusion_model.input_blocks.0.0.weight",
"model.diffusion_model.input_blocks.0.0.bias",
"model.diffusion_model.out.0.weight",
"model.diffusion_model.out.0.bias",
"model.diffusion_model.out.2.bias",
]

IDENTIFIER = ["d1","d2","con1","con2","bri","col1","col2","col3","hd1","hd2","hrs","st1","st2","st3","dis"]
IDENTIFIER_C = ["sp","md","cols","stc","str"]


COLS = [[-1,1/3,2/3],[1,1,0],[0,-1,-1],[1,0,1]]
COLSXL = [[0,0,1],[1,0,0],[-1,-1,0],[-1,1,0]]
