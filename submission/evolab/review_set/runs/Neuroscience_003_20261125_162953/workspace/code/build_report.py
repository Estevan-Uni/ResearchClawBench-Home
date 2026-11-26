import re
t = open('/workspace/outputs/tables.md').read()
tabs = {m.group(1): m.group(2).strip() for m in re.finditer(r'## (\S+)[^\n]*\n(.*?)(?=\n## |\Z)', t, re.S)}
s = open('/workspace/code/report_template.md').read()
s = re.sub(r'\{\{(\S+?)\}\}', lambda m: tabs[m.group(1)], s)
open('/workspace/report/report.md', 'w').write(s); print('written', len(s))
