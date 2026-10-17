import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)),'src'))
from qecbb.config import CONFIG
from qecbb.bbcodes import get_code_name_info
info=get_code_name_info()
print('ablations stub', info['name'])
