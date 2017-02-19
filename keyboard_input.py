import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/PyUserInput")
import time

from pymouse import PyMouse
from pykeyboard import PyKeyboard

m = PyMouse()
k = PyKeyboard()

x_dim, y_dim = m.screen_size()
m.click(x_dim/2, y_dim/2, 1)

def accelerate():
	k.press_key('option')
	time.sleep(0.5)
	k.release_key('option')

def turnRight():
	k.press_key('J')
	time.sleep(0.1)
	k.release_key('J')

def turnLeft():
	k.press_key('G')
	time.sleep(0.1)
	k.release_key('G')

def powerUp():
	k.press_key('command')
	time.sleep(0.5)
	k.release_key('command')

