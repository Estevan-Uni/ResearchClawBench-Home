import sys; sys.path.insert(0, '/workspace/code')
import pandas as pd; pd.set_option('display.width', 250)
import screen
M = ['random', 'gbdt', 'wl_lr', 'scratch', 'pre_frozen', 'pre_ft']
for tag in ['real', 'control']:
    R, H = screen.evaluate(tag, M)
    print(tag); print(R.round(3).to_string()); print(H[H.method.isin(['gbdt', 'pre_ft', 'pre_frozen', 'scratch'])].round(3).to_string())
