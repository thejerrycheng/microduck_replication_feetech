set -e
D=/Users/jerrycheng/Desktop/thejerrycheng.github.io/assets/projects/microduck
for n in walk steer run sit jump flip push emote; do python3 assemble.py takes ui_$n $D/app_$n.mp4 "scale=1600:900:flags=lanczos" 27; done
python3 assemble.py takes hero $D/hero.mp4 "crop=1600:900:0:0" 27
for d in takes/emo_*; do n=$(basename $d); python3 assemble.py takes $n $D/$n.mp4 "crop=900:900:350:40,scale=560:560:flags=lanczos" 27; done
