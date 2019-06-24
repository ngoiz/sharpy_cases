
set style line 1 lc rgb "red"    pointsize 1 pointtype 7 linewidth 2
set style line 2 lc rgb "black" pointsize 1 pointtype 7 linewidth 2 dashtype 2
set style line 3 lc rgb "green"   pointsize 1 pointtype 7 linewidth 2
set style line 4 lc rgb "orange"  pointsize 1 pointtype 7 linewidth 2
set style line 5 lc rgb "cyan"  pointsize 1 pointtype 7 linewidth 2
set style line 6 lc rgb "violet"  pointsize 1 pointtype 7 linewidth 2
set style line 8 lc rgb "black"  pointsize 1 pointtype 7 linewidth 2

set style line 11 dt 2 lc rgb "red"     pointsize 1 pointtype 7 linewidth 2
set style line 12 dt 2 lc rgb "orange"  pointsize 1 pointtype 7 linewidth 2
set style line 13 dt 2 lc rgb "blue"    pointsize 1 pointtype 7 linewidth 2
set style line 14 dt 2 lc rgb "green"   pointsize 1 pointtype 7 linewidth 2
set style line 15 dt 2 lc rgb "cyan"  pointsize 1 pointtype 7 linewidth 2
set style line 16 dt 2 lc rgb "violet"  pointsize 1 pointtype 7 linewidth 2
set style line 18 dt 2 lc rgb "black"  pointsize 1 pointtype 7 linewidth 2

set term pngcairo enhanced size 1200,600 linewidth 2 fontscale 1.5
set grid

set key bottom left
dt = 0.001

# set xrange [0:0.2]
#################################################################################
set xlabel "time [s]"
set xrange [0:0.68]

set ylabel "x [m]"
set output "x.png"
p 'simulation.csv' u (dt*$30):27 w l ls 1 title "MB", \
  'wang_x.txt' u 1:2 w l ls 2 title "WANG"

set ylabel "z [m]"
set output "z.png"
p 'simulation.csv' u (dt*$30):29 w l ls 1 title "MB", \
  'wang_y.txt' u 1:2 w l ls 2 title "WANG"
