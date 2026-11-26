"""Shared utilities for the ViCrop replication: bbox selection, VQA metric, image helpers."""
import re, json, os, numpy as np
from PIL import Image

GENERIC_Q = "Write a general description of the image."
SHORT_ANS = "Answer the question using a single word or phrase."

# ----------------------------------------------------------------------------------------------
# Adaptive bounding-box selection (follows the ViCrop reference procedure):
# windows of side bbox_size * r (r in RATIOS, bbox_size = encoder input resolution in pixels) slide
# over the token-level map; for each r keep the max-sum window; pick r maximising the "sharpness"
# (max window sum - mean of the 4 adjacent window sums) / (#blocks).
# ----------------------------------------------------------------------------------------------
RATIOS = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]


def select_bbox(att, region_wh, bbox_size, ratios=RATIOS, valid=None):
    """att: (h, w) non-negative map covering a region of region_wh=(W, H) pixels.
    Returns (x1, y1, x2, y2) in region pixel coordinates and the chosen ratio."""
    att = np.asarray(att, dtype=np.float64)
    if valid is not None:
        att = att * valid
    h, w = att.shape
    W, H = region_wh
    bw, bh = W / w, H / h  # pixel size of one block
    best = None
    cands = []
    for r in ratios:
        nx = min(max(1, int(bbox_size * r / bw)), w)
        ny = min(max(1, int(bbox_size * r / bh)), h)
        if w - nx < 1 and h - ny < 1:
            if r == ratios[0]:
                return (0, 0, W, H), r
            continue
        # integral image for window sums
        ii = np.pad(att, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        S = ii[ny:, nx:] - ii[:-ny, nx:] - ii[ny:, :-nx] + ii[:-ny, :-nx]  # (h-ny+1, w-nx+1)
        y, x = np.unravel_index(np.argmax(S), S.shape)
        adj = [S[y, x - 1]] if x > 0 else []
        if x < S.shape[1] - 1: adj.append(S[y, x + 1])
        if y > 0: adj.append(S[y - 1, x])
        if y < S.shape[0] - 1: adj.append(S[y + 1, x])
        diff = (S[y, x] - np.mean(adj)) / (nx * ny) if adj else 0.0
        cands.append((diff, r, x, y, nx, ny))
    diff, r, x, y, nx, ny = max(cands, key=lambda c: c[0])
    side = bbox_size * r
    xc = x * bw + bw * nx / 2
    yc = y * bh + bh * ny / 2
    xc = min(max(xc, side / 2), W - side / 2) if W > side else W / 2
    yc = min(max(yc, side / 2), H - side / 2) if H > side else H / 2
    x1, y1 = max(0, xc - side / 2), max(0, yc - side / 2)
    x2, y2 = min(W, xc + side / 2), min(H, yc + side / 2)
    return (int(x1), int(y1), int(x2), int(y2)), r


def random_bbox_like(box, img_wh, rng):
    """Random box of the same width/height as `box`, uniformly placed inside the image."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    W, H = img_wh
    nx1 = rng.integers(0, max(1, W - bw + 1))
    ny1 = rng.integers(0, max(1, H - bh + 1))
    return (int(nx1), int(ny1), int(nx1 + bw), int(ny1 + bh))


def expand2square(img, bg):
    W, H = img.size
    if W == H:
        return img, (0, 0)
    S = max(W, H)
    res = Image.new(img.mode, (S, S), bg)
    ox, oy = (S - W) // 2, (S - H) // 2
    res.paste(img, (ox, oy))
    return res, (ox, oy)


def box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# ----------------------------------------------------------------------------------------------
# Standard VQA accuracy (port of the official EvalAI / VQA eval normalisation)
# ----------------------------------------------------------------------------------------------
contractions = {"aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've", "couldnt": "couldn't",
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've", "hasnt": "hasn't", "havent": "haven't",
    "hed": "he'd", "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's", "howd": "how'd", "howll": "how'll",
    "hows": "how's", "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've", "isnt": "isn't", "itd": "it'd",
    "itd've": "it'd've", "it'dve": "it'd've", "itll": "it'll", "let's": "let's", "maam": "ma'am", "mightnt": "mightn't",
    "mightnt've": "mightn't've", "mightn'tve": "mightn't've", "mightve": "might've", "mustnt": "mustn't",
    "mustve": "must've", "neednt": "needn't", "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at", "shant": "shan't", "shed've": "she'd've",
    "she'dve": "she'd've", "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've", "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've", "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
    "somebodys": "somebody's", "someoned": "someone'd", "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's", "somethingd": "something'd",
    "somethingd've": "something'd've", "something'dve": "something'd've", "somethingll": "something'll",
    "thats": "that's", "thered": "there'd", "thered've": "there'd've", "there'dve": "there'd've",
    "therere": "there're", "theres": "there's", "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've",
    "theyll": "they'll", "theyre": "they're", "theyve": "they've", "twas": "'twas", "wasnt": "wasn't",
    "wed've": "we'd've", "we'dve": "we'd've", "weve": "we've", "werent": "weren't", "whatll": "what'll",
    "whatre": "what're", "whats": "what's", "whatve": "what've", "whens": "when's", "whered": "where'd",
    "wheres": "where's", "whereve": "where've", "whod": "who'd", "whod've": "who'd've", "who'dve": "who'd've",
    "wholl": "who'll", "whos": "who's", "whove": "who've", "whyll": "why'll", "whyre": "why're", "whys": "why's",
    "wont": "won't", "wouldve": "would've", "wouldnt": "wouldn't", "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've", "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
    "yall'd've": "y'all'd've", "y'alld've": "y'all'd've", "y'all'dve": "y'all'd've", "youd": "you'd",
    "youd've": "you'd've", "you'dve": "you'd've", "youll": "you'll", "youre": "you're", "youve": "you've"}
manualMap = {'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6',
             'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'}
articles = ['a', 'an', 'the']
periodStrip = re.compile(r"(?!<=\d)(\.)(?!\d)")
commaStrip = re.compile(r"(\d)(,)(\d)")
punct = [';', r"/", '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']


def processPunctuation(inText):
    outText = inText
    for p in punct:
        if (p + ' ' in inText or ' ' + p in inText) or (re.search(commaStrip, inText) is not None):
            outText = outText.replace(p, '')
        else:
            outText = outText.replace(p, ' ')
    outText = periodStrip.sub("", outText, re.UNICODE)
    return outText


def processDigitArticle(inText):
    outText = []
    for word in inText.lower().split():
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
    for wordId, word in enumerate(outText):
        if word in contractions:
            outText[wordId] = contractions[word]
    return ' '.join(outText)


def normalize_answer(a):
    a = a.replace('\n', ' ').replace('\t', ' ').strip()
    a = processPunctuation(a)
    a = processDigitArticle(a)
    return a


def vqa_accuracy(pred, answers):
    """Official VQA accuracy: average over the 10 leave-one-out subsets of min(1, #match/3)."""
    p = normalize_answer(pred)
    gts = [normalize_answer(g) for g in answers]
    accs = []
    for i in range(len(gts)):
        others = gts[:i] + gts[i + 1:]
        accs.append(min(1.0, sum(g == p for g in others) / 3.0))
    return float(np.mean(accs))


def clean_pred(s):
    s = s.strip().split('\n')[0].strip()
    if s.endswith('.'):
        s = s[:-1]
    return s


def append_jsonl(path, rec):
    with open(path, 'a') as f:
        f.write(json.dumps(rec) + '\n')
        f.flush(); os.fsync(f.fileno())


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    for l in open(path):
        l = l.strip()
        if l:
            try: out.append(json.loads(l))
            except Exception: pass
    return out
