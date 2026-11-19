import json, os, subprocess, sys
T=os.path.abspath(sys.argv[1]); name=sys.argv[2]; out=sys.argv[3]; vf=sys.argv[4]; crf=sys.argv[5] if len(sys.argv)>5 else "26"
d=f"{T}/{name}"; m=json.load(open(d+"/meta.json")); ts=m["ts"]
with open(d+"/list.txt","w") as f:
    for i,t in enumerate(ts):
        dur=(ts[i+1]-t) if i+1<len(ts) else 0.033
        f.write(f"file '{d}/{i:05d}.jpg'\nduration {dur:.4f}\n")
subprocess.run(["ffmpeg","-v","error","-y","-f","concat","-safe","0","-i",d+"/list.txt","-an","-vf",vf+",fps=30","-c:v","libx264","-crf",crf,"-preset","slow","-pix_fmt","yuv420p","-movflags","+faststart",out],check=True)
print(name, ts[0]-m["t0"])
