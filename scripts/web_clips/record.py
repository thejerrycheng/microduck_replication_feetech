import subprocess, sys, time, os, json, base64
from playwright.sync_api import sync_playwright
REPO=os.path.expanduser("~/Desktop/microduck_replication_feetech")
OUT=sys.argv[1]; ONLY=sys.argv[2:]  # optional list of segment names
srv=subprocess.Popen([os.path.expanduser("~/.venvs/microduck_live/bin/python"),"app/server.py","--port","8841"],cwd=REPO,stdout=subprocess.PIPE,text=True)
EMO=json.load(open(REPO+"/app/animations/manifest.json"))["clips"]; EM={c["name"]:c for c in EMO}

class Rec:
    def __init__(s,pg):
        s.pg=pg; s.cdp=pg.context.new_cdp_session(pg); s.dir=None; s.ts=[]; s.ev=[]; s.t0=0; s.meta=False; s.alt=False
        s.cdp.on("Page.screencastFrame",s.frame)
    def frame(s,e):
        s.cdp.send("Page.screencastFrameAck",{"sessionId":e["sessionId"]})
        if s.dir is None: return
        t=e["metadata"]["timestamp"]; i=len(s.ts); s.ts.append(t)
        open(f"{s.dir}/{i:05d}.jpg","wb").write(base64.b64decode(e["data"]))
    def start(s,name):
        d=f"{OUT}/{name}"; os.makedirs(d,exist_ok=True)
        for f in os.listdir(d): os.remove(d+"/"+f)
        s.ts=[]; s.ev=[]; s.dir=d; s.name=name
        s.cdp.send("Page.startScreencast",{"format":"jpeg","quality":92,"everyNthFrame":2}); s.t0=time.time(); s.wait(0.6)
    def stop(s):
        s.wait(0.3); s.cdp.send("Page.stopScreencast"); d=s.dir; s.dir=None
        json.dump({"t0":s.t0,"ts":s.ts,"ev":s.ev},open(d+"/meta.json","w")); print(s.name,len(s.ts),"frames",round(s.ts[-1]-s.ts[0],1),"s",flush=True)
    def wait(s,sec): s.pg.wait_for_timeout(int(sec*1000))
    def log(s,typ,key): s.ev.append([round(time.time()-s.t0,3),typ,key])
    def js(s,typ,name): s.pg.evaluate(f"window.duckKey('{typ}','{name}',{str(s.meta).lower()},{str(s.alt).lower()},false)")
    def down(s,*ks):
        for k in ks: s.js("down",k); s.log("down",k)
    def up(s,*ks):
        for k in ks: s.js("up",k); s.log("up",k)
    def mod(s,meta=None,alt=None):
        if meta is not None and meta!=s.meta: s.meta=meta; s.log("down" if meta else "up","meta")
        if alt is not None and alt!=s.alt: s.alt=alt; s.log("down" if alt else "up","alt")
        s.js("flags","")
    def tap(s,k): s.js("down",k); s.log("tap",k)
    def hold(s,keys,sec): s.down(*keys); s.wait(sec); s.up(*keys)
    def reset(s):
        s.pg.evaluate("window.duckKey('down','enter',false,false,false)"); s.wait(1.6)
    def level(s,cur,to):
        while cur<to: s.tap("d"); cur+=1; s.wait(0.5)
        while cur>to: s.tap("a"); cur-=1; s.wait(0.5)
    def emote(s,name,extra=1.0):
        s.tap(EM[name]["key"]); s.wait(EM[name]["duration_s"]+extra)
    def drag(s,dx,dy,sec=0.9,hold=0.5):
        pg=s.pg; vw=pg.evaluate("({width:innerWidth,height:innerHeight})"); pg.keyboard.down("Meta"); s.meta=True; s.log("down","meta")
        hit=False
        for (x,y) in [(.5,.5),(.5,.42),(.5,.58),(.5,.35),(.47,.5),(.53,.5),(.5,.65)]:
            px,py=vw["width"]*x,vw["height"]*y; pg.mouse.move(px,py); pg.mouse.down()
            if pg.evaluate("document.querySelector('.key[data-k=drag]').classList.contains('on')"): hit=True; break
            pg.mouse.up()
        if hit:
            s.log("down","drag"); n=int(sec*30)
            for i in range(1,n+1): pg.mouse.move(px+dx*i/n,py+dy*i/n); s.wait(1/30)
            s.wait(hold); pg.mouse.up(); s.log("up","drag")
        else: print("drag miss",flush=True)
        pg.keyboard.up("Meta"); s.meta=False; s.log("up","meta"); s.js("flags","")
        return hit

def orbit(pg,x,y,dx):
    pg.mouse.move(x,y); pg.mouse.down()
    for k in range(1,16): pg.mouse.move(x+dx*k/15,y); pg.wait_for_timeout(20)
    pg.mouse.up(); pg.wait_for_timeout(600)
def want(n): return not ONLY or n in ONLY
def falls(pg): return int(pg.evaluate("document.getElementById('falls').textContent") or 0)
def take(r,name,fn,tries=5,settle=0.6):
    for k in range(tries):
        r.reset(); r.wait(settle); f0=falls(r.pg); r.start(name); fn(); r.stop()
        if falls(r.pg)==f0: return
        print(name,"fell, retake",k,flush=True)


try:
    while True:
        l=srv.stdout.readline()
        if l.startswith("DUCK_LIVE_READY"): _,port,tok=l.split(); break
    url=f"http://127.0.0.1:{port}/?token={tok}"
    with sync_playwright() as p:
        L=lambda h:p.chromium.launch(channel="chrome",headless=True,args=["--use-angle=metal","--enable-gpu","--ignore-gpu-blocklist","--window-size=1600,%d"%h])
        b=L(987)
        # ---------------- pass A: the app with its interface
        pg=b.new_page(no_viewport=True); pg.goto(url); time.sleep(9); print("inner",pg.evaluate("[innerWidth,innerHeight]"),flush=True)
        orbit(pg,1000,250,-225); r=Rec(pg); lv=3
        if want("ui_walk"):
            r.reset(); r.level(lv,1); lv=1; r.start("ui_walk"); r.down("up"); r.wait(2.6)
            for _ in range(3): r.tap("d"); lv+=1; r.wait(2.6)
            r.up("up"); r.wait(1.0); r.hold(["down"],2.6); r.wait(0.8); r.stop()
        if want("ui_steer"):
            r.reset(); r.level(lv,3); lv=3; r.start("ui_steer")
            r.hold(["left"],2.2); r.wait(.4); r.hold(["right"],2.2); r.wait(.4)
            r.down("up"); r.wait(1.0); r.down("left"); r.wait(2.4); r.up("left"); r.down("right"); r.wait(2.4); r.up("right","up"); r.wait(.8)
            r.mod(alt=True); r.hold(["left"],2.4); r.wait(.3); r.hold(["right"],2.4); r.mod(alt=False); r.wait(.8); r.stop()
        if want("ui_run"):
            r.reset(); r.level(lv,3); lv=3; r.start("ui_run")
            r.mod(meta=True); r.down("up"); r.wait(3.4); r.up("up"); r.mod(meta=False); r.wait(1.6)
            r.tap("d"); r.wait(.5); r.tap("d"); lv=5; r.wait(.6)
            r.mod(meta=True); r.down("up"); r.wait(3.6); r.up("up"); r.mod(meta=False); r.wait(2.6); r.stop()
        if want("ui_sit"):
            r.reset(); r.level(lv,3); lv=3; r.start("ui_sit")
            r.tap("s"); r.wait(2.2); r.tap("s"); r.wait(1.8); r.tap("s"); r.wait(2.2); r.hold(["up"],2.6); r.wait(.8); r.stop()
        if want("ui_jump"):
            take(r,"ui_jump",lambda:(r.wait(.4),r.tap("space"),r.wait(4.2),r.tap("space"),r.wait(4.2)),settle=5.5)
        if want("ui_flip"):
            take(r,"ui_flip",lambda:(r.wait(.4),r.tap("r"),r.wait(5.0),r.tap("r"),r.wait(5.2)))
        if want("ui_push"):
            r.reset(); r.start("ui_push"); r.wait(.4)
            r.drag(260,-40); r.wait(3.2); r.drag(-300,-20,sec=0.7); r.wait(3.4)
            r.down("up"); r.wait(1.4); r.drag(240,60,sec=0.6,hold=0.4); r.wait(3.0); r.up("up"); r.wait(1.0); r.stop()
        if want("ui_emote"):
            r.reset(); r.start("ui_emote"); r.wait(.3)
            for n in ["hello","dance","laugh","peck"]: r.emote(n,0.8)
            r.stop()
        b.close(); b=L(1287)
        # ---------------- pass B: no interface
        pg=b.new_page(no_viewport=True); pg.goto(url); time.sleep(9); print("inner",pg.evaluate("[innerWidth,innerHeight]"),flush=True)
        pg.add_style_tag(content=".glass:not(#bubble){display:none!important}")
        orbit(pg,1300,300,-300); r=Rec(pg)
        if want("hero"):
            pg.mouse.move(800,600)
            for _ in range(9): pg.mouse.wheel(0,100); pg.wait_for_timeout(60)
            r.reset(); r.wait(4.0); r.start("hero"); r.wait(1.2)
            r.down("up"); r.wait(1.2); r.down("left"); r.wait(3.2); r.up("left"); r.wait(1.6); r.down("right"); r.wait(3.2); r.up("right"); r.wait(1.0); r.up("up"); r.wait(1.2)
            r.emote("hello",0.8); r.emote("excited",0.6); r.stop()
            for _ in range(9): pg.mouse.wheel(0,-100); pg.wait_for_timeout(60)
        for n in ["hello","happy","dance","sing","love","proud","showoff","peck","yawn","sneeze","startled","curious","laugh","excited","angry","sad","mock","nap","play_dead"]:
            if want("emo_"+n):
                take(r,"emo_"+n,lambda:r.emote(n,2.4 if n in("happy","showoff") else 0.9),tries=1 if n=="play_dead" else 4)
        b.close()
finally:
    srv.terminate()
