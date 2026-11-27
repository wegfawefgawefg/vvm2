#!/usr/bin/env bash
# Images through pointers that move on a grid and zoom (old/image7.py, look.py, shapes.py).
# The machine is the frozen core; the image device lays a 28 by 28 image out as 66 input
# slots at three levels and supplies six steps: right, left, down, up, zoom in, zoom out.
#   old/scripts/image7.sh class mnist 3000 seed=0     classify, an MLP of the same size beside it;
#                                                also kmnist, fashion, shapes (name the shape,
#                                                tested on held-out sizes and places)
#   old/scripts/image7.sh class mnist 3000 tune=1000  and then 1000 steps taught with whole steps
#   old/scripts/image7.sh look mnist seed=0           where the pointers go: text, and a picture in
#                                                docs/img/ (needs the class run first;
#                                                which=tuned or which=mix for the machine a
#                                                run with tune= or mix= saved)
#   old/scripts/image7.sh class shapes 1500 mix=0.5   half the training passes run with whole steps
#   old/scripts/image7.sh draw 2000                   draw a named shape onto a 49-slot canvas
#   old/scripts/image7.sh gen fashion 2000            generate clothes and bags from a class and
#                                                8 style numbers
#   old/scripts/image7.sh time                        seconds a training step, nothing scored
#   old/scripts/image7.sh --quick                     everything above, tiny, end to end
# Options after the steps: seed=0 ops=128 ticks=12 reads=4 scratch=8 mlp=1 (class);
# seed=0 ops=64 reads=2 ruler=1 (draw, gen). Data: scripts/get_mnist.sh, scripts/get_data.sh
# kmnist, scripts/get_data.sh fashion (fetched here if missing). nice 19; 4 threads, or
# IMAGE_THREADS=1 old/scripts/image7.sh ... for one (faster in total when the CPU is shared).
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -x .venv/bin/python ] || scripts/setup.sh
py="nice -n 19 .venv/bin/python"

need() {
    case "$1" in
    mnist) [ -f data/mnist/t10k-labels-idx1-ubyte ] || scripts/get_mnist.sh ;;
    kmnist | fashion) [ -f "data/$1/t10k-labels-idx1-ubyte" ] || scripts/get_data.sh "$1" ;;
    esac
}

what="${1:-class}"
case "$what" in
--quick)
    need mnist
    need fashion
    export IMAGE_PICTURES=out/image/quick      # seed 99 and this folder: nothing real is
    $py -m old.image7 class mnist 30 ticks=6 ops=16 tune=10 seed=99      # written over
    $py -m old.image7 look mnist seed=99
    $py -m old.image7 look mnist seed=99 which=tuned
    $py -m old.image7 class shapes 30 ticks=6 ops=16 mix=0.5 seed=99
    $py -m vvm2.shapes draw 10 ops=8 seed=99
    $py -m vvm2.shapes gen fashion 10 ops=8 seed=99
    ;;
class | look)
    need "${2:-mnist}"
    $py -m old.image7 "$@"
    ;;
time)
    $py -m old.image7 "$@"
    ;;
draw)
    $py -m vvm2.shapes "$@"
    ;;
gen)
    need "${2:-fashion}"
    $py -m vvm2.shapes "$@"
    ;;
*)
    sed -n '2,23p' "$0"
    exit 1
    ;;
esac
