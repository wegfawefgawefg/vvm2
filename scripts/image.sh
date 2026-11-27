#!/usr/bin/env bash
# Images through pointers that move on a grid and zoom (vvm2/image.py, look.py, shapes.py).
# The machine is the frozen core; the image device lays a 28 by 28 image out as 66 input
# slots at three levels and supplies six steps: right, left, down, up, zoom in, zoom out.
#   scripts/image.sh class mnist 3000 seed=0     classify, an MLP of the same size beside it;
#                                                also kmnist, fashion, shapes (name the shape,
#                                                tested on held-out sizes and places)
#   scripts/image.sh class mnist 3000 tune=1000  and then 1000 steps taught with whole steps
#   scripts/image.sh look mnist seed=0           where the pointers go: text, and a picture in
#                                                docs/img/ (needs the class run first;
#                                                tuned=1 for the machine tune= saved)
#   scripts/image.sh draw 2000                   draw a named shape onto a 49-slot canvas
#   scripts/image.sh gen fashion 2000            generate clothes and bags from a class and
#                                                8 style numbers
#   scripts/image.sh time                        seconds a training step, nothing scored
#   scripts/image.sh --quick                     everything above, tiny, end to end
# Options after the steps: seed=0 ops=128 ticks=12 reads=4 scratch=8 mlp=1 (class);
# seed=0 ops=64 reads=2 ruler=1 (draw, gen). Data: scripts/get_mnist.sh, scripts/get_data.sh
# kmnist, scripts/get_data.sh fashion (fetched here if missing). 4 threads, nice 19.
set -euo pipefail
cd "$(dirname "$0")/.."
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
    $py -m vvm2.image class mnist 30 ticks=6 ops=16 tune=10
    $py -m vvm2.image look mnist
    $py -m vvm2.image look mnist tuned=1
    $py -m vvm2.image class shapes 30 ticks=6 ops=16
    $py -m vvm2.shapes draw 10 ops=8
    $py -m vvm2.shapes gen fashion 10 ops=8
    ;;
class | look)
    need "${2:-mnist}"
    $py -m vvm2.image "$@"
    ;;
time)
    $py -m vvm2.image "$@"
    ;;
draw)
    $py -m vvm2.shapes "$@"
    ;;
gen)
    need "${2:-fashion}"
    $py -m vvm2.shapes "$@"
    ;;
*)
    sed -n '2,20p' "$0"
    exit 1
    ;;
esac
