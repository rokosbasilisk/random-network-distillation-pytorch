wget http://www.atarimania.com/roms/Roms.rar
pip install unrar
unrar x Roms.rar
mkdir rars
mv HC\ ROMS.zip   rars
mv ROMS.zip  rars
python -m atari_py.import_roms rars

