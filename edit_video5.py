from pathlib import Path
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
ROOT=Path(__file__).resolve().parent
FF=imageio_ffmpeg.get_ffmpeg_exe()
SRC=ROOT/'source_5.mp4'
W,H=1080,1920
# Original, typeset cover art (no generated text artifacts).
cover=Image.new('RGB',(600,1000),'#132c3c');d=ImageDraw.Draw(cover)
gold='#dfbd7c'; cream='#f5ead1'
def f(n,serif=False):
 return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/'+('DejaVuSerif' if serif else 'DejaVuSans')+'.ttf',n)
def text(s,y,n,color=cream,serif=False):
 d.text((300,y),s,font=f(n,serif),fill=color,anchor='mt')
d.rectangle((22,22,578,978),outline=gold,width=3)
d.rectangle((35,35,565,965),outline=gold,width=1)
text('A  P O T T E R  E D I T I O N',75,18,gold)
text('THE',180,47,gold,True)
text('UNWRITTEN',250,65,cream,True)
text('CHAPTER',335,65,cream,True)
d.line((215,445,385,445),fill=gold,width=2)
# Fine botanical foil illustration.
for side in [-1,1]:
 pts=[]
 for i in range(110):
  t=i/109; pts.append((int(300+side*(25+165*t-25*t*t)),int(790-260*t)))
 d.line(pts,fill=gold,width=3)
 for j in range(1,10):
  t=j/10; x=300+side*(25+165*t-25*t*t); y=790-260*t
  for k in [-1,1]:
   d.line([(x,y),(x+side*(25+15*t),y-18*k-20)],fill=gold,width=2)
d.ellipse((285,555,315,585),outline=gold,width=2)
text('Every story deserves',815,24)
text('to be told.',850,24)
text('POTTER PUBLISHERS',920,22,gold)
cover.save(ROOT/'assets/potter_unwritten_cover.png')
art=cv2.cvtColor(np.array(cover),cv2.COLOR_RGB2BGR)
cap=cv2.VideoCapture(str(SRC)); fps=cap.get(cv2.CAP_PROP_FPS); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# Half-resolution frames suffice for stable planar tracking.
grays=[]
while True:
 ok,frame=cap.read()
 if not ok:break
 grays.append(cv2.cvtColor(cv2.resize(frame,(540,960)),cv2.COLOR_BGR2GRAY))
cap.release();n=len(grays)
ref=420
# Front cover corners at 14 seconds. Spine and page block stay original.
q=np.float32([[372,870],[660,880],[657,1474],[372,1440]])/2
quads={ref:q}
for direction in [1,-1]:
 oldq=q.copy();prev=ref
 for idx in range(ref+direction,n if direction==1 else 299,direction):
  mask=np.zeros_like(grays[prev]);cv2.fillConvexPoly(mask,np.int32(oldq),255)
  mask=cv2.erode(mask,np.ones((7,7),np.uint8))
  pts=cv2.goodFeaturesToTrack(grays[prev],maxCorners=160,qualityLevel=.01,minDistance=4,mask=mask)
  newq=oldq.copy()
  if pts is not None and len(pts)>=8:
   nxt,status,_=cv2.calcOpticalFlowPyrLK(grays[prev],grays[idx],pts,None,winSize=(25,25),maxLevel=3)
   valid=status.ravel()==1
   if valid.sum()>=8:
    mat,inliers=cv2.findHomography(pts[valid],nxt[valid],cv2.RANSAC,1.5)
    if mat is not None:newq=cv2.perspectiveTransform(oldq[None],mat)[0]
  quads[idx]=newq;oldq=newq;prev=idx
# Encode modified frames, preserving the source audio without re-encoding it.
enc=subprocess.Popen([FF,'-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s','1080x1920','-r',str(fps),'-i','pipe:0','-i',str(SRC),'-map','0:v','-map','1:a:0','-vf','delogo=x=850:y=1690:w=104:h=104','-c:v','libx264','-preset','fast','-crf','18','-c:a','copy','-movflags','+faststart','-shortest',str(ROOT/'Potter_Video5_Clean_New_Cover.mp4')],stdin=subprocess.PIPE)
cap=cv2.VideoCapture(str(SRC))
base=np.float32([[0,0],[599,0],[599,999],[0,999]])
for idx in range(n):
 ok,frame=cap.read()
 if not ok:break
 if idx in quads:
  dst=quads[idx]*2
  mat=cv2.getPerspectiveTransform(base,dst.astype(np.float32))
  warped=cv2.warpPerspective(art,mat,(W,H))
  mask=cv2.warpPerspective(np.ones((1000,600),np.uint8)*255,mat,(W,H))
  # Match the gentle lighting gradient of the photographed book.
  light=np.linspace(1.035,.87,H,dtype=np.float32)[:,None,None]
  warped=np.clip(warped.astype(np.float32)*light,0,255).astype(np.uint8)
  alpha=cv2.GaussianBlur(mask,(3,3),.6).astype(np.float32)[:,:,None]/255
  # Retain the white feather when it crosses the lower edge of the cover.
  yy=np.arange(H)[:,None]
  feather=(frame.min(axis=2)>196)&(yy>dst[:,1].max()-155)
  # Keep the cover solid; warm highlights are not foreground objects.
  frame=(frame*(1-alpha)+warped*alpha).astype(np.uint8)
 enc.stdin.write(frame.tobytes())
enc.stdin.close();assert enc.wait()==0
cap.release()
