# coding: utf-8
import re
from ttyplay import TtyPlay
from tqdm import tqdm
import time
import sys

inf = 9999999
enc = 'cp437'

msg_pattern = r'\x1b\[1;\d{1,2}H|\x1b\[H(.*?)\x1b\[K(.*?)'

stats_patterns = {}
stats_patterns['strength'] = r'St:(\d*)'
stats_patterns['dexterity'] = r'Dx:(\d*)'
stats_patterns['constitution'] = r'Co:(\d*)'
stats_patterns['intelligence'] = r'In:(\d*)'
stats_patterns['wisdom'] = r'Wi:(\d*)'
stats_patterns['charisma'] = r'Ch:(\d*)'
stats_patterns['dungeonlvl'] = r'Dlvl:(\d*)'
stats_patterns['gold'] = r'\$:(\d*)'
stats_patterns['hitpoints'] = r'HP:(\d*)\((\d*)\)'
stats_patterns['power'] = r'Pw:(\d*)\((\d*)\)'
stats_patterns['armor'] = r'AC:(\d*)'
stats_patterns['experience'] = r'Exp:(\d*)'
stats_patterns['time'] = r'T:(\d*)'

def return_msg(msg_pattern,string):
    result = re.search(msg_pattern, string)
    return result.group(1)
    
def process_tty(tty_file):

    tty_info = {}
    ttyplay = TtyPlay(tty_file)
    frames = []

    durations = ttyplay.compute_framedelays()
    durations.append(inf)

    while(ttyplay.read_frame()):
        frames.append(ttyplay.display_frame())
    
    msgs = {}
    stats = {}

    # fill messages
    for idx in range(len(frames)):
        frame = frames[idx].decode(enc)
        _ = []
        try:
            msg = return_msg(msg_pattern,frame)
            if 2<len(msg)<80:
                msgs[idx] = msg
        except:
            pass

        # fill stats
        for key in list(stats_patterns.keys()):
            try:
                value = return_msg(stats_patterns[key],frame)
                _.append((key,value))
                if (key=='time'):
                    try:
                        _.append(('hunger',return_msg(r'T:'+str(value)+' (.*)',frame).split(' ')[0]))
                    except:
                        pass
                if (key=='charisma'):
                    try:
                        _.append(('alignment',return_msg(r'Ch:'+str(value)+'  (.*)',frame).split(' ')[0]))
                    except:
                        pass
            except:
                pass
            if len(_)>1:
                stats[idx] = _

    tty_info['frames'] = frames
    tty_info['duration'] = durations
    tty_info['messages'] = msgs
    tty_info['stats'] = stats
    return tty_info


