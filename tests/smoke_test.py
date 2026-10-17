import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from qecbb.bbcodes import get_code_name_info
from qecbb.model import GAT_LIF_BBDecoder
info = get_code_name_info()
m = GAT_LIF_BBDecoder(in_channels=8, k_logical=info['K_LOGICAL'], spiking=True)
assert hasattr(m, 'forward')
print('OK')
