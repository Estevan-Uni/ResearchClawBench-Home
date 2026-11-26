"""Score LaTeX answers against a hand-transcribed ground truth.
Normalisation: drop \\left/\\right, whitespace, $ delimiters, \\[ \\]; collapse {x}->x for single chars."""
import re, json, sys

GT = r"A_n = a_0 \left[ 1 + \frac{3}{4} \sum_{k=1}^{n} \left( \frac{4}{9} \right)^k \right]"

def extract(ans):
    m = re.findall(r"```(?:latex)?\s*(.*?)```", ans, re.S)
    s = m[0] if m else ans
    m2 = re.findall(r"\$\$(.*?)\$\$", s, re.S) or re.findall(r"\\\[(.*?)\\\]", s, re.S) or re.findall(r"\$(.*?)\$", s, re.S)
    if m2: s = max(m2, key=len)
    return s.strip()

def norm(s):
    s = s.replace("\\left", "").replace("\\right", "").replace("\\displaystyle", "")
    s = re.sub(r"\\begin\{(equation|align)\*?\}|\\end\{(equation|align)\*?\}", "", s)
    s = re.sub(r"[\s$]", "", s).replace("\\[", "").replace("\\]", "") if False else re.sub(r"[\s$]", "", s)
    prev = None
    while prev != s:
        prev = s; s = re.sub(r"\{(\\?[A-Za-z0-9])\}", r"\1", s)
    return s

def lev(a, b):
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (ca != cb))
    return d[-1]

def balanced(s):
    st = []; pairs = {")": "(", "]": "[", "}": "{"}
    for c in re.sub(r"\\[{}]", "", s):
        if c in "([{": st.append(c)
        elif c in ")]}":
            if not st or st[-1] != pairs[c]: return False
            st.pop()
    return not st

def score(ans):
    ex = extract(ans); a, g = norm(ex), norm(GT)
    d = lev(a, g)
    return {"extracted": ex, "normalised": a, "gt_normalised": g, "edit_distance": d,
            "ned": d / max(len(a), len(g)), "exact": a == g, "balanced": balanced(ex)}

if __name__ == "__main__":
    print(json.dumps(score(sys.argv[1]), indent=1))
