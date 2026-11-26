import pypdf, glob, os
for f in sorted(glob.glob('related_work/*.pdf')):
    r = pypdf.PdfReader(f)
    txt = "\n".join((p.extract_text() or '') for p in r.pages)
    out = 'outputs/' + os.path.basename(f).replace('.pdf', '.txt')
    open(out, 'w').write(txt)
    print(f, len(r.pages), len(txt))
