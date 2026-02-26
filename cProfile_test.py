import cProfile, pstats
from mandelbrot import mandelbrot_set_numpy, mandelbrot_set

cProfile.run('mandelbrot_set_numpy(-2.0, 1.0, -1.5, 1.5, 1024, 1024,100)', 'mandelbrot_numpy.prof')
cProfile.run('mandelbrot_set(-2.0, 1.0, -1.5, 1.5, 1024, 1024,100)', 'mandelbrot_set.prof')

for name in ('mandelbrot_numpy.prof', 'mandelbrot_set.prof'):
    stats = pstats.Stats(name)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
