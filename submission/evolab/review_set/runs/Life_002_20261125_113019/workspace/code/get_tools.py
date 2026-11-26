import urllib.request, tarfile, os, sys
d = sys.argv[1]
os.chdir(d)
for url, fn in [("https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz","foldseek.tar.gz"),
                ("https://conda.anaconda.org/bioconda/linux-64/usalign-2024.07.30-hdbdd923_0.tar.bz2","usalign.tar.bz2")]:
    if not os.path.exists(fn):
        urllib.request.urlretrieve(url, fn); print("got", fn, os.path.getsize(fn), flush=True)
t = tarfile.open("foldseek.tar.gz"); t.extractall("fs"); 
t = tarfile.open("usalign.tar.bz2","r:bz2"); t.extractall("us")
print("done")
