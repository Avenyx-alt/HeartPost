"""Record the HeartPost demo as a silent 2:00 screen recording that follows docs/VIDEO_SCRIPT.md."""
import time, json, sys
from playwright.sync_api import sync_playwright

import pathlib
APP = (pathlib.Path(__file__).resolve().parents[2] / "app").as_uri() + "/"
W, H, ZOOM = 1920, 1080, 1.4          # 1080p video; page zoomed so text is large and crisp
OUT = str(pathlib.Path(__file__).resolve().parent / "raw")

CURSOR_JS = r"""
(() => {
  const mk = () => {
    if (document.getElementById('__fc')) return;
    const c = document.createElement('div'); c.id='__fc';
    c.innerHTML = '<svg width="34" height="34" viewBox="0 0 24 24"><path d="M3 2l7.5 19 2.6-7.7L21 10.7z" fill="#111" stroke="#fff" stroke-width="1.6" stroke-linejoin="round"/></svg>';
    Object.assign(c.style,{position:'fixed',left:'-100px',top:'-100px',zIndex:2147483647,pointerEvents:'none',transform:'translate(-6px,-4px)',filter:'drop-shadow(0 2px 3px rgba(0,0,0,.35))'});
    document.documentElement.appendChild(c);
    const r = document.createElement('div'); r.id='__fr';
    Object.assign(r.style,{position:'fixed',width:'46px',height:'46px',marginLeft:'-23px',marginTop:'-23px',borderRadius:'50%',border:'4px solid rgba(179,38,30,.85)',zIndex:2147483646,pointerEvents:'none',opacity:0,transform:'scale(.3)',transition:'transform .45s ease-out, opacity .45s ease-out'});
    document.documentElement.appendChild(r);
    const z = 1;
    addEventListener('mousemove', e => { c.style.left = e.clientX*z+'px'; c.style.top = e.clientY*z+'px'; }, true);
    addEventListener('mousedown', e => { r.style.transition='none'; r.style.left=e.clientX*z+'px'; r.style.top=e.clientY*z+'px'; r.style.opacity=1; r.style.transform='scale(.3)';
      requestAnimationFrame(()=>requestAnimationFrame(()=>{ r.style.transition='transform .45s ease-out, opacity .45s ease-out'; r.style.transform='scale(1.4)'; r.style.opacity=0; })); }, true);
  };
  if (document.readyState === 'loading') addEventListener('DOMContentLoaded', mk); else mk();
})();
"""
ZOOM_JS = f"""
(() => {{ const z = () => {{ if (document.body) document.body.style.zoom = '{ZOOM}'; }};
  if (document.readyState === 'loading') addEventListener('DOMContentLoaded', z); else z(); }})();
try {{ localStorage.setItem('hpLang','en'); }} catch(e) {{}}
"""

def main():
    with sync_playwright() as pw:
        b = pw.chromium.launch(channel="chrome")
        ctx = b.new_context(viewport={"width": W, "height": H}, locale="en-US",
                            record_video_dir=OUT, record_video_size={"width": W, "height": H})
        ctx.add_init_script(ZOOM_JS); ctx.add_init_script(CURSOR_JS)
        created = time.monotonic()
        pg = ctx.new_page()
        pg.goto(APP + "index.html"); pg.wait_for_timeout(300)
        pg.mouse.move(W * .62, H * .78)
        start = time.monotonic()
        log = []

        def at(t):
            now = time.monotonic() - start
            if now < t: time.sleep(t - now)
            log.append((t, round(time.monotonic() - start, 2)))

        def box(sel):
            """Element centre in screen pixels (the body is zoomed)."""
            r = pg.evaluate("""s=>{const e=document.querySelector(s).getBoundingClientRect();return [e.left,e.top,e.width,e.height]}""", sel)
            z = pg.evaluate("1")
            return (r[0] + r[2] / 2) * z, (r[1] + r[3] / 2) * z, r[2] * z, r[3] * z

        def scroll_to(sel, top=110, ms=1300):
            pg.evaluate("""([s,top,ms])=>new Promise(res=>{const z=1;
                const e=document.querySelector(s); const y0=scrollY; const y1=Math.max(0,y0+e.getBoundingClientRect().top*z-top);
                const t0=performance.now(); const f=t=>{const k=Math.min(1,(t-t0)/ms); const ease=k<.5?2*k*k:1-Math.pow(-2*k+2,2)/2;
                scrollTo(0,y0+(y1-y0)*ease); k<1?requestAnimationFrame(f):res();}; requestAnimationFrame(f);})""", [sel, top, ms])

        def scroll_top(ms=900):
            pg.evaluate("""ms=>new Promise(res=>{const y0=scrollY,t0=performance.now();const f=t=>{const k=Math.min(1,(t-t0)/ms);
                scrollTo(0,y0*(1-(k<.5?2*k*k:1-Math.pow(-2*k+2,2)/2)));k<1?requestAnimationFrame(f):res();};requestAnimationFrame(f);})""", ms)

        def move(sel, steps=30, dx=0, dy=0):
            x, y, _, _ = box(sel); pg.mouse.move(x + dx, y + dy, steps=steps)

        def click(sel, steps=30):
            move(sel, steps); time.sleep(.25); x, y, _, _ = box(sel); pg.mouse.down(); time.sleep(.08); pg.mouse.up()

        def zoomed_click_el(sel):
            # use the real element click after the visual mouse click, so the action always lands
            pg.evaluate("s=>document.querySelector(s).click()", sel)

        # ---------- Part 1 · 0:00–0:17 · the problem (hero) ----------
        at(5);  move(".facts dl > div:nth-child(1)", steps=60, dx=-40)
        at(8);  move(".facts dl > div:nth-child(2)", steps=40, dx=-40)
        at(11); move(".facts dl > div:nth-child(3)", steps=40, dx=-40)
        at(14.5); scroll_to(".examples", top=40, ms=1500)

        # ---------- Part 2 · 0:17–0:45 · first patient ----------
        at(18); click('.chip[data-ex="0"]')
        at(26); scroll_to("#calcBox", top=180, ms=1200)
        at(28); click("#calcBox summary")
        at(29.5); move("#calcBody tr:nth-child(2)", steps=40, dx=-120)
        at(32); move("#calcBody tr.sum", steps=40, dx=-120)
        at(35); scroll_to("#truth", top=380, ms=1100)
        at(37); move("#truth", steps=40, dx=-80)

        # ---------- Part 3 · 0:45–1:15 · fairness and failure ----------
        at(44.5); scroll_to(".examples", top=40, ms=1000)
        at(46.5); click('.chip[data-ex="1"]')
        at(50); scroll_to("#truth", top=420, ms=1000)
        at(51.5); move("#truth", steps=40, dx=-60)
        at(58.5); scroll_to(".examples", top=40, ms=1000)
        at(60.5); click('.chip[data-ex="3"]')
        at(63.5); scroll_to("#truth", top=380, ms=1000)
        at(65); move("#truth", steps=40, dx=-60)
        at(69.5); move("#warnYoung", steps=40, dx=-60)

        # ---------- Part 4 · 1:15–1:45 · every unseen patient ----------
        at(74.5); scroll_to("#explorer", top=30, ms=1800)
        at(80); x0 = pg.evaluate("(()=>{const r=document.getElementById('xSvg').getBoundingClientRect();const z=1;return [ (r.left + (X.geo.padL+X.geo.plotW*X.tM)*r.width/X.geo.W)*z, (r.top+r.height*0.5)*z, (r.left+(X.geo.padL+X.geo.plotW*0.62)*r.width/X.geo.W)*z ]})()")
        pg.mouse.move(x0[0], x0[1], steps=40)
        at(83); pg.mouse.down()
        t_d=time.monotonic()
        for i in range(1, 25):                      # drag right over ~3 s
            pg.mouse.move(x0[0] + (x0[2] - x0[0]) * i / 24, x0[1])
        drag_s=time.monotonic()-t_d
        at(88)
        for i in range(1, 25):                      # and back
            pg.mouse.move(x0[2] - (x0[2] - x0[0]) * i / 24, x0[1])
        log.append(("drag_seconds", round(drag_s,2)))
        pg.mouse.up()
        pg.evaluate("X.tM=MODEL.t_men; xUpdate()")  # land exactly on our setting
        at(94); move("#xWomen", steps=40)
        at(96.5); x, y, _, _ = box("#xWomen"); pg.mouse.click(x, y)
        pg.wait_for_timeout(150)
        if pg.evaluate("document.getElementById('xWomen').checked"):   # make sure it really switched off
            pg.evaluate("(()=>{const c=document.getElementById('xWomen'); c.checked=false; c.dispatchEvent(new Event('change'))})()")
        log.append(("women_off_missed", pg.inner_text('#cMissed')))
        at(97); move("#xFair", steps=40, dx=-200)
        at(102); move("#xWomen", steps=30)
        at(103.5); x, y, _, _ = box("#xWomen"); pg.mouse.click(x, y)
        pg.wait_for_timeout(150)
        if not pg.evaluate("document.getElementById('xWomen').checked"):
            pg.evaluate("(()=>{const c=document.getElementById('xWomen'); c.checked=true; c.dispatchEvent(new Event('change'))})()")
        log.append(("women_on_missed", pg.inner_text('#cMissed')))

        # ---------- Part 5 · 1:45–2:00 · limitation and next step ----------
        at(105); scroll_top(900)
        at(106.3); click('.nav a[href="followup.html"]', steps=30)
        pg.wait_for_load_state("load"); pg.wait_for_timeout(200)
        pg.mouse.move(W * .5, H * .5)
        at(108.5); click('.role[data-role="doctor"]', steps=30)
        at(110.5); scroll_to("#doctorView", top=20, ms=1300)
        at(113); move(".row:nth-of-type(2) .why", steps=40)
        at(116); move(".row:nth-of-type(3) .why", steps=40)
        at(120.5)

        json.dump({"offset": start - created, "log": log}, open(OUT + "/timing.json", "w"))
        vid = pg.video.path()
        ctx.close(); b.close()
        print("video:", vid, "offset:", round(start - created, 2))
        late = [(t, a) for t, a in log if isinstance(t, (int, float)) and a - t > 0.4]
        print("late cues:", late, [l for l in log if isinstance(l[0], str)])

main()
