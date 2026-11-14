import json, sys
root = sys.argv[1]
html = open(f"{root}/tools/app_template.html", encoding="utf-8").read()
for key, fn in [("MODEL", "model"), ("PERF", "performance"), ("EXAMPLES", "examples"), ("STORY", "story")]:
    tag = f"/*{key}_JSON*/null"; assert tag in html, tag
    html = html.replace(tag, json.dumps(json.load(open(f"{root}/app/{fn}.json"))))
open(f"{root}/app/index.html", "w", encoding="utf-8").write(html)
print("app written", len(html))
