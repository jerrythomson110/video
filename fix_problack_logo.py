"""Apply the supplied official logo to the phone header and final brand card."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import subprocess, imageio_ffmpeg
R=Path(__file__).resolve().parent
src=R/'Video6_No_Gemini_Watermark.mp4'
logo=Image.open(R/'assets/pro-black-logo.png').convert('RGBA')
# Dark backing keeps the white lettering of the supplied logo legible.
header=Image.new('RGB',(600,185),'#171b18')
mark=logo.resize((560,158),Image.Resampling.LANCZOS);header.paste(mark,(20,13),mark)
art=cv2.cvtColor(np.array(header),cv2.COLOR_RGB2BGR)
card=Image.new('RGB',(1080,1920),'#111612');d=ImageDraw.Draw(card)
d.line((130,590,950,590),fill='#227b32',width=3)
mark=logo.resize((820,231),Image.Resampling.LANCZOS);card.paste(mark,(130,745),mark)
d.line((130,1110,950,1110),fill='#227b32',width=3)
f=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',54)
d.text((540,1200),'problack.co',font=f,fill='white',anchor='mt')
card.save(R/'assets/problack_corrected_endcard.png')
end=cv2.cvtColor(np.array(card),cv2.COLOR_RGB2BGR)
c=cv2.VideoCapture(str(src));gray=[]
while True:
 ok,frame=c.read()
 if not ok:break
 gray.append(cv2.cvtColor(cv2.resize(frame,(540,960)),cv2.COLOR_BGR2GRAY))
c.release()
ref=60
q=np.float32([[330,402],[552,376],[563,455],[340,481]])/2
qs={ref:q}
for direction in [1,-1]:
 old=q.copy();prev=ref
 for i in range(ref+direction,98 if direction==1 else 47,direction):
  mask=np.zeros((960,540),np.uint8)
  # Track the header lettering and surrounding non-scrolling top chrome.
  cv2.fillConvexPoly(mask,old.astype(np.int32),255)
  pts=cv2.goodFeaturesToTrack(gray[prev],120,.005,3,mask=mask)
  if pts is not None:
   nxt,status,_=cv2.calcOpticalFlowPyrLK(gray[prev],gray[i],pts,None,winSize=(25,25),maxLevel=3)
   valid=status.ravel()==1
   if valid.sum()>=6:
    mat,_=cv2.findHomography(pts[valid],nxt[valid],cv2.RANSAC,1.5)
    if mat is not None:old=cv2.perspectiveTransform(old[None],mat)[0]
  qs[i]=old.copy();prev=i
ff=imageio_ffmpeg.get_ffmpeg_exe()
p=subprocess.Popen([ff,'-y','-v','error','-f','rawvideo','-pix_fmt','bgr24','-s','1080x1920','-r','30','-i','pipe:0','-i',str(src),'-map','0:v','-map','1:a:0','-c:v','libx264','-crf','18','-preset','fast','-c:a','copy','-shortest','-movflags','+faststart',str(R/'ProBlack_Official_Logo_Fixed.mp4')],stdin=subprocess.PIPE)
c=cv2.VideoCapture(str(src))
for i in range(len(gray)):
 ok,frame=c.read()
 if not ok:break
 if i in qs:
  mat=cv2.getPerspectiveTransform(np.float32([[0,0],[599,0],[599,184],[0,184]]),qs[i]*2)
  overlay=cv2.warpPerspective(art,mat,(1080,1920))
  mask=cv2.warpPerspective(np.full((185,600),255,np.uint8),mat,(1080,1920))
  a=mask[:,:,None].astype(np.float32)/255
  frame=(frame*(1-a)+overlay*a).astype(np.uint8)
 if i>=275:frame=end.copy()
 p.stdin.write(frame.tobytes())
p.stdin.close();assert p.wait()==0
c.release()
