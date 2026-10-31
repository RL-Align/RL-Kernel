"""Render the six miles + Hopper RFC diagrams using the issue #563 style.

The geometry and visual style follow the approved Chinese roadmap.
Requires reportlab, pymupdf and Pillow; run in the desired output directory.
"""
from pathlib import Path
import math
import fitz
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.lib import colors

ASSETS = Path(__file__).resolve().parent
INK=colors.HexColor('#1E1E1E'); BLUE=colors.HexColor('#1971C2')
PALE=colors.HexColor('#99C9EF'); GRAY=colors.HexColor('#515151')

class Figure:
    def __init__(self,c,w=1200,h=660):
        self.c=c; self.w=w; self.h=h; self.hatched_regions=[]
    def label(self,x,y,s,size=23,bold=False,color=INK,align='center'):
        c=self.c; c.setFillColor(color);c.setFont('Helvetica' if bold else 'Helvetica',size)
        # Keep thin blue hatching clear of glyphs, without a text-box fill.
        width=pdfmetrics.stringWidth(s,'Helvetica',size)
        left=x-width/2 if align=='center' else (x-width if align=='right' else x)
        baseline=self.h-y-size*.34
        if any(rx <= x <= rx+rw and ry <= y <= ry+rh for rx,ry,rw,rh in self.hatched_regions):
            c.saveState(); c.setFillColor(colors.white); c.setStrokeColor(colors.white)
            c.setLineWidth(1.45)
            text=c.beginText(left,baseline); text.setFont('Helvetica',size)
            text.setTextRenderMode(2); text.textOut(s); c.drawText(text); c.restoreState()
            c.setFillColor(color); c.setFont('Helvetica',size)
        fn={'center':c.drawCentredString,'left':c.drawString,'right':c.drawRightString}[align]
        fn(x,baseline,s)
    def node(self,x,y,w,h,lines,new=False,small=False):
        c=self.c;yy=self.h-y-h
        c.setFillColor(colors.white);c.setStrokeColor(INK);c.setLineWidth(1.8);c.roundRect(x,yy,w,h,13,fill=1,stroke=1)
        if new:
            c.saveState();p=c.beginPath();p.roundRect(x+1,yy+1,w-2,h-2,12);c.clipPath(p,stroke=0,fill=0)
            c.setStrokeColor(PALE);c.setLineWidth(.6)
            for k in range(-int(h),int(w)+int(h),8): c.line(x+k,yy,x+k+h,yy+h)
            c.restoreState();c.setStrokeColor(INK);c.setLineWidth(1.8);c.roundRect(x,yy,w,h,13,stroke=1,fill=0)
            self.hatched_regions.append((x,y,w,h))
            self.label(x-8,y+8,'[+]',size=20,bold=True,color=BLUE,align='right')
        specs=[]
        for idx,l in enumerate(lines):
            if isinstance(l,tuple): specs.append(l)
            else: specs.append((l,24 if idx==0 else 19,idx==0))
        specs=[(s,min(sz,(w-22)/max(pdfmetrics.stringWidth(s,'Helvetica' if (b and b[0]) else 'Helvetica',sz),1)*sz),*(b or [False])) for s,sz,*b in specs]
        total=sum(sz*1.4 for _,sz,*_ in specs)
        cy=y+(h-total)/2
        for line,sz,*b in specs:
            self.label(x+w/2,cy+sz*.68,line,size=sz,bold=b[0] if b else False)
            cy+=sz*1.4
        return (x,y,w,h)
    def group(self,x,y,w,h,title):
        c=self.c;c.setStrokeColor(GRAY);c.setLineWidth(1.4);c.setDash(1.5,5);c.roundRect(x,self.h-y-h,w,h,16,stroke=1,fill=0);c.setDash()
        tw=pdfmetrics.stringWidth(title,'Helvetica',20)
        c.setFillColor(colors.white);c.rect(x+20,self.h-y-13,tw+20,26,fill=1,stroke=0)
        self.label(x+30,y,title,size=20,align='left')
    def arrow(self,pts,blue=False,dash=False,label=None,at=None):
        c=self.c;col=BLUE if blue else INK;c.setStrokeColor(col);c.setFillColor(col);c.setLineWidth(2 if blue else 1.7)
        c.setDash(7,6) if dash else c.setDash()
        p=c.beginPath();p.moveTo(pts[0][0],self.h-pts[0][1])
        for x,y in pts[1:]:p.lineTo(x,self.h-y)
        c.drawPath(p);c.setDash()
        a,b=pts[-2:];ang=math.atan2(b[1]-a[1],b[0]-a[0]);r=9
        for off in [-.45,.45]:c.line(b[0],self.h-b[1],b[0]-r*math.cos(ang+off),self.h-(b[1]-r*math.sin(ang+off)))
        if label:
            lx,ly=at;tw=pdfmetrics.stringWidth(label,'Helvetica',18);c.setFillColor(colors.white);c.rect(lx-tw/2-5,self.h-ly-12,tw+10,24,fill=1,stroke=0);self.label(lx,ly,label,size=18,color=col)

def architecture(f):
    f.group(320,142,850,134,'Training and rollout integration')
    f.group(320,301,850,186,'Shared integration and kernel selection')
    f.group(320,521,850,135,'Device implementations and shared kernels')
    f.node(38,24,280,79,[('Existing: vime integration',23,False),('integrations/orchestrators/vime',16,False)])
    f.node(495,24,500,79,[('New: miles integration',25,False),('integrations/orchestrators/miles',20,False)],True)
    f.arrow([(318,63),(495,63)],blue=True,dash=True,label='Reuse the approach',at=(405,82))
    f.node(355,174,330,81,[('Extend: training integration',23,False),('Init hook + logprob provider',17,False),('engines/train/<backend>',17,False)],True)
    f.node(805,174,330,81,[('New: SGLang rollout adapter',23,False),('SGLang hooks / operator replacement',18,False),('engines/rollout/sglang',17,False)],True)
    f.node(370,330,750,59,[('Extend: shared operator integration',23,False),('integrations/common / engines/*/operators',19,False)],True)
    f.node(370,414,750,59,[('Extend: ops interfaces, registry / plan',23,False),('New: selector / performance implementations',20,False)],True)
    f.node(38,174,225,112,[('New: hybrid MoE model',22,False),('models/qwen3_5_moe',18,False),('GDN / attention / MoE',19,False)],True)
    f.node(38,356,225,110,[('Extend: contracts / platforms',18,False),('GDN / MoE numerical rules',19,False),('Hopper device capabilities',19,False)],True)
    f.node(360,548,225,64,[('Extend: CUDA Hopper',22,False),('H100 / H200',18,False)],True)
    f.node(640,548,215,64,[('Canonical reference',21,False),('Fixed-order baseline',18,False)],True)
    f.node(910,548,215,64,[('Shared validated kernels',20,False),('backends/shared',18,False)])
    for x,t in [(472,'csrc/cuda'),(747,'reference/'),(1017,'shared/triton')]:f.label(x,638,t,size=19)
    f.arrow([(650,103),(650,174)])
    f.arrow([(840,103),(970,174)])
    f.arrow([(520,255),(520,286),(770,286),(770,330)])
    f.arrow([(970,255),(970,330)])
    f.arrow([(745,389),(745,414)])
    f.arrow([(263,230),(297,230),(297,356),(370,356)],blue=True,dash=True)
    f.arrow([(263,411),(299,411),(299,445),(370,445)],blue=True,dash=True)
    for x in [472,747,1017]:
        f.arrow([(x,473),(x,548)])
        f.arrow([(x,612),(x,624)])
    f.label(150,529,'Reuse the Megatron integration',16)
    f.label(150,559,'Add other engines as needed',16)
    f.label(150,597,'State, EP and tests: see task table',14)


def dispatcher(f):
    f.group(18,22,1158,164,'Read configuration and registered kernels')
    f.node(48,69,276,86,['Chip and software versions','H100 / H200 / CUDA stack'],True)
    f.node(431,69,343,86,['Model, shape and phase','shape / dtype / profile'])
    f.node(880,69,263,86,['Registered kernels','registry / capability'])
    f.node(410,246,380,88,['Find compatible implementations','Check gradients, precision and consistency'])
    f.node(900,250,243,80,['No compatible kernel','Report unsupported configuration'])
    f.node(48,359,276,116,['Kernel measurements','Latency / workspace memory','performance + benchmarks'],True)
    f.node(410,411,380,91,['Select by latency and memory','selector / within the memory limit'],True)
    f.node(898,428,250,126,['Cache the selection','Chip / shape / rules / versions','Reuse on match; reselect on change'],True)
    f.node(410,565,380,76,['Bind kernels for this run','operators / plan / provenance'],True)
    for x,y in [(186,155),(602,155),(1012,155)]:f.arrow([(x,y),(x,208),(602,208),(602,246)])
    f.arrow([(790,290),(900,290)],label='None',at=(846,275))
    f.arrow([(602,334),(602,411)])
    f.arrow([(324,417),(410,451)],blue=True,dash=True)
    f.arrow([(602,502),(602,565)])
    f.arrow([(898,457),(790,457)],blue=True,dash=True,label='Cache hit',at=(845,437))
    f.arrow([(790,603),(1023,603),(1023,554)],blue=True,dash=True,label='Save choice',at=(903,603))
    f.label(186,550,'No measurements: use a valid default',16)
    f.label(186,580,'and mark it as untuned',18)

def speculative_component_figure(f, lang='cn'):
    en = lang == 'en'
    def t(cn, eng): return eng if en else cn
    f.group(30,31,360,586,t('推理引擎 / adapter','SGLang / adapter'))
    f.group(430,285,438,273,t('RL-Kernel 投机解码组件','RL-Kernel speculative component'))
    f.label(650,91,t('抽出共用的投机解码逻辑','Reusable speculative decoding'),24)
    f.label(650,136,t('共用协议 + sampling / packing','Common protocol + sampling / packing'),19)
    f.label(650,179,'CUDA Hopper',20)

    f.node(78,94,280,62,[
        (t('保存本轮前缀状态','Save the round-start state'),22,False),
        (t('记录已计算状态长度','Record computed-state length'),17,False)])
    f.node(78,203,280,62,[
        (t('生成草稿','Generate draft tokens'),23,False),
        (t('Draft model / MTP 等','Draft model / MTP / other methods'),17,False)])
    f.node(78,314,280,72,[
        (t('目标模型验证','Target model verification'),23,False),
        (t('运行已验证算子，得到目标概率','Validated kernels; target probabilities'),17,False)])
    f.node(465,314,370,76,[
        (t('接受 / 拒绝与补充采样','Accept / reject and sample'),23,False),
        (t('拒绝：残差采样；全接受：bonus','Reject: residual; all accepted: bonus'),17,False),
        (t('目标 / 草稿分布 + 显式 RNG','Target / draft distributions + RNG'),17,False)],True)
    f.node(465,448,370,99,[
        (t('整理保留序列','Pack the retained sequence'),23,False),
        (t('Tokens + 目标 logprob','Tokens + target logprobs'),19,False),
        (t('停止条件 + 状态更新要求','Stop rules + state-update plan'),18,False)],True)
    f.node(78,455,280,88,[
        (t('提交 / 恢复保留前缀状态','Commit / restore prefix state'),22,False),
        (t('KV / GDN / 卷积状态','KV / GDN / convolution state'),17,False),
        (t('更新本轮输出与状态长度','Update output and state lengths'),17,False)])
    f.node(931,463,245,100,[
        (t('Rollout 完成','Rollout complete'),23,False),
        (t('最终 tokens / 目标 logprob','Final tokens / target logprobs'),17,False),
        (t('采样配置 / 策略标识','Sampling config / policy ID'),17,False)])
    f.node(931,140,245,110,[
        (t('训练端独立重算','Training-side recomputation'),22,False),
        (t('相同 checkpoint / tokens','Same checkpoint / tokens'),17,False),
        (t('相同概率定义','Same probability semantics'),18,False)])

    f.arrow([(218,156),(218,203)])
    f.arrow([(218,265),(218,314)])
    f.arrow([(358,234),(889,234),(889,337),(835,337)],label=t('草稿分布','Draft distribution'),at=(535,214))
    f.arrow([(358,350),(465,350)])
    f.label(411,373,t('目标分布','Target'),16)
    f.arrow([(650,390),(650,448)])
    f.arrow([(465,499),(358,499)])
    f.arrow([(180,543),(180,577),(48,577),(48,125),(78,125)],label=t('继续下一轮','Next round'),at=(144,565))
    f.arrow([(218,543),(218,600),(1053,600),(1053,563)],label=t('达到停止条件','Finished'),at=(850,586))
    f.arrow([(1053,463),(1053,250)],blue=True,dash=True,label=t('logprob 校验','Logprob check'),at=(1053,355))
    f.label(414,640,t('实线：生成流程    虚线：完成后的校验','Solid: generation   Dashed: validation   Sampling shown: stochastic linear draft'),18)


def speculative(f):
    speculative_component_figure(f, 'en')

def pd_contract_figure(f, lang='cn'):
    cn = lang == 'cn'
    def t(a,b): return a if cn else b
    c=f.c
    c.setStrokeColor(colors.HexColor('#A5ADB5')); c.setLineWidth(1.2)
    for x,y1,y2 in [(140,84,610),(805,84,624),(1070,84,205),(1070,477,535)]:
        c.line(x,660-y1,x,660-y2)
    f.node(22,20,236,64,['Prefill',t('固定权重版本','Pin the weight version')])
    f.node(670,20,270,64,['Decode',t('使用同一版权重','Use the same weights')])
    f.node(956,20,228,64,['Training',t('使用该 rollout 的权重','Use the rollout checkpoint')])
    f.arrow([(140,116),(805,116)],label=t('请求、状态规格、契约 ID / hash','Request, state format, contract ID / hash'),at=(470,99))
    f.node(670,139,270,56,[t('分配接收缓冲','Allocate receive buffers')],True)
    f.arrow([(805,211),(140,211)],label=t('接收端就绪','Receiver ready'),at=(466,193))
    f.group(310,237,337,89,t('L1 框架层契约','L1 Framework contract'))
    for y,a,b in [
        (260,'权重 / 模型 / tokens','Weights / model / tokens'),
        (285,'位置 / mask / RoPE','Positions / mask / RoPE'),
        (310,'有效前缀 / 状态覆盖长度','Valid prefix / state coverage')]:
        f.label(478,y,t(a,b),size=18)
    f.node(22,270,236,67,[t('计算前缀','Compute the prefix'),'Attention / MoE'])
    f.arrow([(140,358),(805,358)],blue=True,label=t('KV / 混合状态 + 交接元数据','KV / hybrid state + handoff metadata'),at=(472,339))
    f.label(472,382,t('交接元数据包含契约 ID / hash','Handoff metadata includes contract ID / hash'),size=17,color=BLUE)
    f.group(310,418,337,153,t('L2 算子层契约','L2 Operator contract'))
    for y,a,b in [
        (443,'Attention：LSE / 输出合并','Attention / GDN: merge / state rules'),
        (470,'GEMM：分块 / 累加 / Split-K','GEMM: tiling / accumulation / Split-K'),
        (497,'MoE：路由 / dispatch / combine','MoE: router / dispatch / combine'),
        (524,'Norm / logprob：规约 / 舍入','Norm / logprob: reduction / rounding'),
        (551,'量化（如启用）：scale / 量化位置','Quantization, if used: scales / boundaries')]:
        f.label(478,y,t(a,b),size=17 if cn else 16)
    f.node(670,421,270,100,[
        (t('核对权重、前缀与契约','Check weights, prefix and contracts'),21,False),
        (t('L1 / L2 检查','Check L1 / L2'),18,False),
        (t('确认所需状态已收齐','Confirm required state is complete'),17,False)],True)
    f.node(22,544,250,66,[
        (t('检查失败：报错','On failure: report the error'),21,False),
        (t('或按匹配契约重算前缀','or recompute under matching contracts'),16,False)])
    f.arrow([(670,478),(658,478),(658,588),(272,588)],blue=True,dash=True)
    f.node(670,559,270,64,['Decode / Verify',t('继续生成，记录 logprob','Generate; record target logprobs')])
    f.arrow([(805,521),(805,559)],label=t('检查通过','Valid'),at=(866,540))
    f.group(964,220,216,243,t('训练对照','Training comparison'))
    for y,a,b in [
        (258,'同一权重 / tokens','Same weights / tokens'),
        (307,'相同位置 / mask','Same positions / mask'),
        (356,'相同 L1 / L2 规则','Same L1 / L2 rules'),
        (405,'相同 logprob 定义','Same logprob definition')]:
        f.label(1072,y,t(a,b),size=16)
    f.node(956,535,228,79,[t('独立重算并逐层比较','Recompute; compare layers'),t('目标 logprob / MoE 结果','Target logprobs / MoE outputs')],True)
    f.arrow([(805,623),(805,646),(1070,646),(1070,614)],blue=True,dash=True,label='tokens / logprob',at=(938,646))
    f.label(309,643,t('状态收齐、契约通过后，才执行 Decode','Decode starts only after state and contract checks pass'),size=17 if cn else 16)


def pd_consistency(f):
    pd_contract_figure(f, 'en')

def pd_topology(f):
    f.group(20,25,1160,230,'1P1D on Hopper: one Prefill node and one Decode node (example)')
    f.node(49,72,448,141,['P: 8 GPUs / EP8','256 experts; 32 per rank','Expert 80 on rank 2'])
    f.node(713,72,431,141,['D: 4 GPUs / EP4','256 experts; 64 per rank','Expert 80 on rank 1'])
    f.arrow([(497,145),(713,145)],blue=True,label='State handoff',at=(605,120))
    f.label(605,177,'KV / hybrid state',size=16)
    f.label(605,202,'Request / contract metadata',size=15)
    f.label(600,236,'Preserve logical expert IDs; validate top-k slots and numerical rules for the same input',19)
    f.group(20,305,1160,290,'Multi-node example: two Prefill nodes and four Decode nodes')
    f.label(257,345,'Prefill: EP16',23,True)
    f.label(942,345,'Decode: EP8',23,True)
    f.node(49,370,399,180,[])
    f.node(732,370,442,180,[])
    f.node(63,384,378, 58,['P-node 0: 8 GPUs','rank 0-7'])
    f.node(63,473,378, 58,['P-node 1: 8 GPUs','rank 8-15'])
    f.node(745,384,194, 58,['D-node 0: 2 GPUs','rank 0-1'])
    f.node(968,384,194, 58,['D-node 1: 2 GPUs','rank 2-3'])
    f.node(745,473,194, 58,['D-node 2: 2 GPUs','rank 4-5'])
    f.node(968,473,194, 58,['D-node 3: 2 GPUs','rank 6-7'])
    f.node(509,394,170,127,[('P to D',24,False),('KV / hybrid state',17,False),('Request / contract metadata',15,False),('Target layout',17,False)],True)
    f.arrow([(448,458),(509,458)])
    f.arrow([(679,458),(732,458)])
    f.label(260,564,'Within P: EP dispatch / return',19)
    f.label(950,564,'Within D: EP dispatch / return',19)
    f.label(600,634,'Measure separately: EP within P, P-to-D state transfer, and EP within D',20)

def pd_combine(f):
    f.group(23,25,321,177,'Arrival order (example)')
    for x,t in [(48,'C2'),(120,'C0'),(192,'C3'),(264,'C1')]:f.node(x,81,61,47,[(t,22,True)])
    f.label(181,166,'token / slot / expert in each message',16)
    f.node(411,25,548,177,[],True)
    f.label(685,53,'Write to fixed buffer slots',23,True)
    for j in range(8):
        x=441+(j%4)*125;y=87+(j//4)*57
        f.node(x,y,108,40,[(f'{j} : C{j}',20,False)])
    f.node(1015,42,160,144,['Check this token:','all 8 slots','received'])
    f.arrow([(344,119),(411,119)])
    f.arrow([(959,119),(1015,119)])
    f.group(23,265,1154,385,'Read from the buffer; reduce with a fixed FP32 tree')
    f.arrow([(1095,186),(1095,233),(600,233),(600,300)])
    for x in [153,451,749,1047]:f.arrow([(600,300),(x,300),(x,332)])
    for x,a,b in [(51,0,1),(349,2,3),(647,4,5),(945,6,7)]:f.node(x,332,204,57,[(f'C{a} + C{b}',25,True)])
    f.node(133,454,341, 59,[('(C0 + C1) + (C2 + C3)',22,False)])
    f.node(726,454,341, 59,[('(C4 + C5) + (C6 + C7)',22,False)])
    f.arrow([(153,389),(303,454)]);f.arrow([(451,389),(303,454)])
    f.arrow([(749,389),(896,454)]);f.arrow([(1047,389),(896,454)])
    f.node(444,563,312, 70,['Add both results','Cast at the specified output step'])
    f.arrow([(303,513),(540,563)]);f.arrow([(896,513),(660,563)])
    f.label(205,581,'No pairing by rank',20)
    f.label(205,616,'No arrival-ordered accumulation',18)
    f.label(975,581,'Check duplicate, missing and stale results',16)
    f.label(975,616,'Readiness does not change the sum order',16)

FIGS={'architecture':architecture,'dispatcher':dispatcher,'speculative':speculative,'pd-consistency':pd_consistency,'pd-topology':pd_topology,'pd-combine':pd_combine}

for name,fn in FIGS.items():
    path=ASSETS/f'{name}.pdf';c=canvas.Canvas(str(path),pagesize=(1200,660));fn(Figure(c));c.showPage();c.save()
    d=fitz.open(path);p=d[0];p.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(ASSETS/f'{name}.png')
    im=Image.open(ASSETS/f'{name}.png')
    im.quantize(colors=128,method=Image.Quantize.MEDIANCUT,dither=Image.Dither.NONE).save(ASSETS/f'{name}.png',optimize=True)
    (ASSETS/f'{name}.svg').write_text(p.get_svg_image(),encoding='utf-8');d.close()


