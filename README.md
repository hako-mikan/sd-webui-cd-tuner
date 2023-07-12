# CD(Color/Detail) Tuner
Color/Detail control for Stable Diffusion web-ui/色調や書き込み量を調節するweb-ui拡張です。

[日本語](#使い方)

Update 2023.07.13.0030(JST)
- add brightness
- color adjusting method is changed
- add disable checkbox

![](https://raw.githubusercontent.com/hako-mikan/sd-webui-cd-tuner/imgs/sample0.png)

This is an extension to modify the amount of detailing and color tone in the output image. It intervenes in the generation process, not on the image after it's generated. It works on a mechanism different from LoRA and is compatible with 1.X and 2.X series. In particular, it can significantly improve the quality of generated products during Hires.fix.

## Usage
It automatically activates when any value is set to non-zero. Please be careful as inevitably the amount of noise increases as the amount of detailing increases. During the use of Hires.fix, the output might look different, so it is recommended to try with expected settings. Values around 5 should be good, but it also depends on the model. If a positive value is input, the detailing will increase.

### Detail1,2 Drawing/Noise Amount
When set to negative, it becomes flat and slightly blurry. When set to positive, the detailing increases and becomes noisy. Even if it is noisy in normal generation, it might become clean with hires.fix, so be careful. Detail1 and 2 both have similar effects, but Detail1 seems to have a stronger effect on the composition. In the case of 2.X series, the reaction of Detail 1 may be the opposite of normal, with more drawings in negative.
![](https://raw.githubusercontent.com/hako-mikan/sd-webui-cd-tuner/imgs/sample8.png)

### Contrast: Contrast/Drawing Amount, Brightness
Contrast and brightness change, and at the same time the amount of detailing also changes. It would be quicker to see the sample.
The difference between Contrast 1 and Contrast 2 lies in whether the adjustment is made during the generation process or after the generation is complete. Making the adjustment during the generation process results in a more natural outcome, but it may also alter the composition.
![](https://raw.githubusercontent.com/hako-mikan/sd-webui-cd-tuner/imgs/sample7.png)

### Color1,2,3 Color Tone
You can tune the color tone. For `Cyan-Red`, it becomes `Cyan` when set to negative and `Red` when set to positive.

![](https://raw.githubusercontent.com/hako-mikan/sd-webui-cd-tuner/imgs/sample6.png)

### Hr-Detail1,2 ,Hires-Scaling
In the case of using Hires-fix, the optimal settings often differ from the usual. Basically, when using Hires-Fix, it is better to input larger values than when not using it. Hr-Detail1,2 is used when you want to set a different value from when not used during Hires-Fix generation. Hires-Scaling is a feature that automatically sets the value at the time of Hires-Fix. The value of Hires-scale squared is usually multiplied by the original value.

## Use in XYZ plot/API
You can specify the value in prompt by entering in the following format. Please use this if you want to use it in XYZ plot.

```
<cdt:d1=2;col1=-3>
<cdt:d2=2;hrs=1>
<cdt:1>
<cdt:0;0;0;-2.3;0,2>
<cdt:0;0;0;-2.3;0;2;0;0;1> 
```

The available identifiers are `d1,d2,con1,con2,bri,col1,col2,col3,hd1,hd2,hrs,st1,st2`. When describing in the format of `0,0,0...`, please write in this order. It is okay to fill in up to the necessary places. The delimiter is a semicolon (;). If you write `1,0,4`, `d1,d2,cont` will be set automatically and the rest will be `0`. `hrs` turns on when a number other than `0` is entered.
This value will be prioritized if a value other than `0` is set.
At this time, `Skipping unknown extra network: cdt` will be displayed, but this is normal operation.

### Stop Step
You can specify the number of steps to stop the adjustment. In Hires-Fix, the effects are often not noticeable after the initial few steps. This is because in most samplers, a rough image is already formed within the first 10 steps.

## Examples of use
The left is before use, the right is after use. Click the image to enlarge it. Here, we are increasing the amount of drawing and making it blue. The difference is clearer when enlarged.

You can expect an improvement in reality with real-series models.
![](https://raw.githubusercontent.com/hako-mikan/sd-webui-cd-tuner/imgs/sample4.png)
![](https://raw.githubusercontent.com/hako-mikan/sd-webui-cd-tuner/imgs/sample5.png)

# Color/Detail control for Stable Diffusion web-ui
出力画像の描き込み量や色調を変更する拡張機能です。生成後の画像に対してではなく生成過程に介入します。LoRAとは異なる仕組みで動いています。2.X系統にも対応しています。特にHires.fix時の生成品質を大幅に向上させることができます。

## 使い方
どれかの値が0以外に設定されている場合、自動的に有効化します。描き込み量が増えると必然的にノイズも増えることになるので気を付けてください。Hires.fix使用時では出力が違って見える場合があるので想定される設定で試すことをおすすめします。数値は大体5までの値を入れるとちょうど良いはずですがそこはモデルにも依存します。正の値を入力すると描き込みが増えたりします。  

### Detail1,2 描き込み量/ノイズ
マイナスにするとフラットに、そして少しぼけた感じに。プラスにすると描き込みが増えノイジーになります。通常の生成でノイジーでもhires.fixできれいになることがあるので注意してください。Detail1,2共に同様の効果がありますが、Detail1は2に比べて構図への影響が強く出るようです。2.X系統の場合、Detail 1の反応が通常とは逆になり、マイナスで書き込みが増える場合があるようです。
![](https://raw.githubusercontent.com/hako-mikan/sd-webui-cd-tuner/imgs/sample8.png)

### Contrast : コントラスト/描き込み量
コントラストや明るさがかわり、同時に描き込み量も変わります。サンプルを見てもらった方が早いですね。
![](https://raw.githubusercontent.com/hako-mikan/sd-webui-cd-tuner/imgs/sample7.png)

### Color1,2,3 色調
色調を補正できます。`Cyan-Red`ならマイナスにすると`Cyan`、プラスにすると`Red`になります。
![](https://raw.githubusercontent.com/hako-mikan/sd-webui-cd-tuner/imgs/sample6.png)

### Hr-Detail1,2 ,Hires-Scaling
Hires-fixを使用する場合、最適な設定値が通常とは異なる場合が多いです。基本的にはHires-Fix使用時には未使用時より大きめの値を入れた方が良い結果が得られます。Hr-Detail1,2ではHires-Fix生成時に未使用時とは異なる値を設定したい場合に使用します。Hires-Scalingは自動的にHires-Fix使用時の値を設定する機能です。おおむねHires-scaleの2乗の値が元の値に掛けられます。

## XYZ plot・APIでの利用について
promptに以下の書式で入力することでpromptで値を指定できます。XYZ plotで利用したい場合にはこちらを利用して下さい。

```
<cdt:d1=2;col1=-3>
<cdt:d2=2;hrs=1>
<cdt:1>
<cdt:0;0;0;-2.3;0,2>
<cdt:0;0;0;-2.3;0;2;0;0;1> 
```

使用できる識別子は`d1,d2,con1,con2,bri,col1,col2,col3,hd1,hd2,hrs,st1,st2`です。`0,0,0...`の形式で記述する場合にはこの順に書いてください。区切りはセミコロン「;」です。記入は必要なところまでで大丈夫です。`1,0,4`なら自動的に`cont`までが設定され残りは`0`になります。`hrs`は`0`以外の数値が入力されるとオンになります。
`0`以外の値が設定されている場合にはこちらの値が優先されます。
このとき`Skipping unknown extra network: cdt`と表示されますが正常な動作です。

### stop step
補正を停止するステップ数を指定できます。Hires-Fixでは最初の数ステップ以降は効果が感じられないことが多いです。大概のサンプラーで10ステップ絵までには大まかな絵ができあがっているからです。

## 使用例
リアル系モデルでリアリティの向上が見込めます。
![](https://raw.githubusercontent.com/hako-mikan/sd-webui-cd-tuner/imgs/sample4.png)
![](https://raw.githubusercontent.com/hako-mikan/sd-webui-cd-tuner/imgs/sample5.png)
