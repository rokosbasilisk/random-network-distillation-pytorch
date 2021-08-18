
import re
import nle
import gym 
import cv2
import random
import numpy as np
from PIL import Image, ImageDraw




env = gym.make("NetHackChallenge-v0")
env.reset()
random_action = random.randint(0,113)
obs = env.step(random_action)[0]

img = nle.nethack.tty_render(obs['tty_chars'], obs['tty_colors'], obs['tty_cursor'])

def process_frame(frame):
    blstats = frame['blstats']
    msg = frame['message']
    message = ''
    for c in msg:
        message = message+chr(c)
    img = nle.nethack.tty_render(frame['tty_chars'],frame['tty_colors'],frame['tty_cursor'])
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    img = ansi_escape.sub('', img).split('\n')[2:-2]
    frame = ''
    for l in img:
        frame = frame+l+'\n'
    img = Image.new(mode='RGB',size=(790,370))
    text = ImageDraw.Draw(img)
    text.text((0, 0),frame, fill=(255,255,255))
    return np.array(img),message.split('\x00')[0],blstats


