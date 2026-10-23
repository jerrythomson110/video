"""Rebuild a mobile-safe end card and perspective-correct phone display."""
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import subprocess
import imageio_ffmpeg
from pathlib import Path

ROOT = Path(__file__).resolve().parent
A = ROOT / 'assets'
A.mkdir(exist_ok=True)
FF = imageio_ffmpeg.get_ffmpeg_exe()
SRC = ROOT / 'Potter_Publishers_No_Gemini_Watermark.mp4'
NAVY = '#13283e'
GOLD = '#d9b779'
CREAM = '#f5efe3'
def font(n, serif=False, bold=False):
    name = 'DejaVuSerif' if serif else 'DejaVuSans'
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/'+name+('-Bold' if bold else '')+'.ttf', n)
def center(d, text, y, f, fill, width=1080):
    d.text((width/2,y),text,font=f,fill=fill,anchor='mt')

# Extract the existing brand mark rather than inventing a replacement.
subprocess.run([FF,'-y','-v','error','-ss','8','-i',str(SRC),'-frames:v','1',str(A/'potter_reference.png')],check=True)
ref=Image.open(A/'potter_reference.png').convert('RGB')
logo=ref.crop((260,590,810,1060)).resize((440,376),Image.Resampling.LANCZOS)
# Remove only the pale paper background around the mark.
a=np.array(logo)
alpha=np.where((a.min(axis=2)>145)&((a.max(axis=2)-a.min(axis=2))<48),0,255).astype('uint8')
logo=logo.convert('RGBA')
card=Image.new('RGB',(1080,1920),NAVY);d=ImageDraw.Draw(card)
d.rounded_rectangle((54,130,1026,1730),radius=34,outline=GOLD,width=2)
center(d,'POTTER PUBLISHERS',205,font(30,bold=True),GOLD)
d.rounded_rectangle((292,282,788,714),radius=26,fill=CREAM)
card.paste(logo,(320,310),logo)
d=ImageDraw.Draw(card)
center(d,'Your story.',765,font(78,serif=True),CREAM)
center(d,'Ready for the world.',870,font(62,serif=True),CREAM)
d.line((440,995,640,995),fill=GOLD,width=3)
center(d,'Real Stories. Lasting Impact.',1050,font(34),GOLD)
center(d,'Editing  /  Cover design  /  Formatting',1160,font(29),CREAM)
center(d,'From manuscript to published book.',1210,font(28),CREAM)
d.rounded_rectangle((145,1350,935,1458),radius=20,fill=GOLD)
center(d,'BOOK A FREE CONSULTATION',1387,font(30,bold=True),NAVY)
center(d,'potterpublishers.com',1510,font(39,bold=True),CREAM)
card.save(A/'potter_mobile_endcard.png')

screen=Image.new('RGBA',(600,1000),CREAM);s=ImageDraw.Draw(screen)
s.rectangle((0,0,600,130),fill=NAVY)
center(s,'POTTER',30,font(42,bold=True),CREAM,600)
center(s,'PUBLISHERS',83,font(22),GOLD,600)
center(s,'Your next chapter',205,font(39,serif=True),NAVY,600)
center(s,'starts here.',266,font(39,serif=True),NAVY,600)
s.line((220,370,380,370),fill=GOLD,width=4)
for text,y in [('Professional editing',430),('Beautiful book covers',500),('Publishing support',570)]:
    center(s,text,y,font(30),NAVY,600)
s.rounded_rectangle((45,700,555,820),radius=20,fill=NAVY)
center(s,'FREE CONSULTATION',742,font(28,bold=True),CREAM,600)
center(s,'potterpublishers.com',900,font(26),NAVY,600)
# Coordinates refer to the existing phone glass, keeping its bezel untouched.
dest=np.array([(764,1408),(1053,1453),(784,1714),(475,1635)],dtype=float)
source=np.array([(0,0),(600,0),(600,1000),(0,1000)],dtype=float)
mat=[];b=[]
for (x,y),(u,v) in zip(dest,source):
    mat.extend([[x,y,1,0,0,0,-u*x,-u*y],[0,0,0,x,y,1,-v*x,-v*y]])
    b.extend([u,v])
coeff=np.linalg.solve(np.array(mat),np.array(b))
phone=screen.transform((1080,1920),Image.Transform.PERSPECTIVE,coeff,Image.Resampling.BICUBIC)
phone.save(A/'potter_phone_overlay.png')
filters="[0:v]eq=contrast=1.025:saturation=1.035[v];[v][1:v]overlay=0:0:enable='between(t,5,6.933333)'[p];[p][2:v]overlay=0:0:enable='gte(t,6.966667)',format=yuv420p[out]"
subprocess.run([FF,'-y','-v','error','-i',str(SRC),'-loop','1','-i',str(A/'potter_phone_overlay.png'),'-loop','1','-i',str(A/'potter_mobile_endcard.png'),'-filter_complex',filters,'-map','[out]','-map','0:a:0','-c:v','libx264','-crf','18','-preset','fast','-c:a','copy','-t','10.05','-movflags','+faststart',str(ROOT/'Potter_Publishers_Mobile_Enhanced.mp4')],check=True)
(A/'potter_reference.png').unlink()
