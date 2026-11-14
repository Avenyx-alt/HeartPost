import re
parts = [  # (start, end, speaker, text) — from docs/VIDEO_SCRIPT.md
 (0,17,"YEVA","Almost half of US counties have no cardiologist. In rural areas, the nearest one is an 87-mile round trip. So a nurse at a small clinic has to decide: which patient with chest pain really needs that trip? That's what HeartPost helps with."),
 (17,45,"YEVA","HeartPost uses only seven things a small clinic can measure, like age, type of chest pain and blood pressure. This is a real patient our model never saw. Red zone: refer soon. And it isn't a black box. Every score is a short sum of points you can check by hand. His real angiography result: heart disease. The model got it right."),
 (45,75,"IPATIY","This woman was sick too. With one threshold for everyone, she'd have been sent home. We found the model was missing half of sick women, so women get their own threshold, and now she's caught. But we don't hide mistakes. This woman also had heart disease, and the model said green. Young patients with unusual pain are its weak spot, and the app warns about it."),
 (75,105,"ARIANA","We trained on one US hospital and tested on three others the model never saw. Here are all 678 test patients. Move the threshold: you catch more sick people, but get more false alarms. A textbook model with thirteen tests missed 93 sick patients in the new hospitals. The simplest baseline rule missed 83. Ours missed 43. Now switch off the women's threshold, and ten sick women turn red: missed."),
 (105,120,"YEVA","Our data is from the 1980s, so next we'd test on today's patients with real clinicians, and grow this home follow-up demo. HeartPost: seven simple measurements, fair to women, and honest about its mistakes."),
]
def ts(t):
    ms=int(round(t*1000)); return f"{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}"
out=[]; n=1
for a,b,who,text in parts:
    sents=re.split(r'(?<=[.?!:])\s+(?=[A-Z0-9])', text)
    # merge very short sentences so each cue is readable
    chunks=[]
    for s_ in sents:
        if chunks and len(chunks[-1].split())+len(s_.split())<=14: chunks[-1]+=" "+s_
        else: chunks.append(s_)
    words=[len(c.split()) for c in chunks]; tot=sum(words); t=a
    for c,w in zip(chunks,words):
        d=(b-a)*w/tot; out.append(f"{n}\n{ts(t)} --> {ts(t+d-0.05)}\n{who}: {c}\n"); n+=1; t+=d
open("HeartPost_voiceover_EN.srt","w").write("\n".join(out))
print(n-1,"cues")
