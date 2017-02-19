import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/PyUserInput")

from pymouse import PyMouse
from pykeyboard import PyKeyboard

m = PyMouse()
k = PyKeyboard()

x_dim, y_dim = m.screen_size()
m.click(x_dim/2, y_dim/2, 1)
k.type_string('Hello, World!')
k.press_key('H')