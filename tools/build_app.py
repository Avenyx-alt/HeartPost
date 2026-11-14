import json, sys
root = sys.argv[1]
data = {key: json.dumps(json.load(open(f"{root}/app/{fn}.json")))
        for key, fn in [("MODEL", "model"), ("PERF", "performance"), ("EXAMPLES", "examples"), ("STORY", "story"), ("UNSEEN", "unseen")]}
for template, out in [("app_template.html", "index.html"), ("followup_template.html", "followup.html")]:
    html = open(f"{root}/tools/{template}", encoding="utf-8").read()
    for key, js in data.items():
        html = html.replace(f"/*{key}_JSON*/null", js)
    assert "_JSON*/null" not in html, template
    open(f"{root}/app/{out}", "w", encoding="utf-8").write(html)
    print("written", out, len(html))
