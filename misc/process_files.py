# coding: utf-8
from tqdm import tqdm 
import os 
import pickle
from process_tty import process_tty

        
for ttyfile in tqdm(sorted(os.listdir('tty/'))):
    processed_tty = process_tty('tty/'+ttyfile)
    with open('processed_ttyrec/'+ttyfile.split('.ttyrec')[0]+'.pickle', 'wb') as handle:
        pickle.dump(processed_tty, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

