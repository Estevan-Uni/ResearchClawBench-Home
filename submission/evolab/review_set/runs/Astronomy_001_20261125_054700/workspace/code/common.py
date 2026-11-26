"""Shared utilities: parse the input data file."""
import re, ast, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data', 'DESI_EDE_Repro_Data.txt')
OUT = os.path.join(ROOT, 'outputs')
IMG = os.path.join(ROOT, 'report', 'images')

BLOCKS = ['lcdm_params', 'ede_params', 'w0wa_params',
          'desi_dvrd_points', 'desi_fap_points', 'sne_mu_points']

def load_data():
    txt = open(DATA).read()
    out = {}
    for b in BLOCKS:
        m = re.search(rf'{b}\s*=\s*(\{{.*?\}}|\[.*?\])', txt, re.S)
        assert m, b
        out[b] = ast.literal_eval(m.group(1))
    shared = ['omega_m', 'H0', 'sigma8', 'ns', 'ombh2', 'ln10As', 'tau']
    for k in ['lcdm_params', 'ede_params', 'w0wa_params']:
        assert all(p in out[k] for p in shared), k
    assert 'f_EDE' in out['ede_params'] and 'log10_ac' in out['ede_params']
    assert 'w0' in out['w0wa_params'] and 'wa' in out['w0wa_params']
    return out

SHARED = ['omega_m', 'H0', 'sigma8', 'ns', 'ombh2', 'ln10As', 'tau']
LABELS = {'omega_m': r'\Omega_\mathrm{m}', 'H0': r'H_0', 'sigma8': r'\sigma_8',
          'ns': r'n_\mathrm{s}', 'ombh2': r'\omega_\mathrm{b}',
          'ln10As': r'\ln(10^{10}A_\mathrm{s})', 'tau': r'\tau_\mathrm{reio}',
          'f_EDE': r'f_\mathrm{EDE}', 'log10_ac': r'\log_{10}a_\mathrm{c}',
          'w0': r'w_0', 'wa': r'w_a'}
MODELS = {'lcdm': 'lcdm_params', 'ede': 'ede_params', 'w0wa': 'w0wa_params'}
NICE = {'lcdm': r'$\Lambda$CDM', 'ede': 'EDE', 'w0wa': r'$w_0w_a$CDM'}
COLORS = {'lcdm': '#4C72B0', 'ede': '#DD5555', 'w0wa': '#2CA02C'}

if __name__ == '__main__':
    d = load_data()
    for k, v in d.items():
        print(k, v)
