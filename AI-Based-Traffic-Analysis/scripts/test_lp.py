from optimization.minmax_routing import solve_minmax

links = [(1, 2), (2, 4), (1, 3), (3, 4)]
paths = [[1, 2, 4], [1, 3, 4]]

# 1) no background, 8 Mbps on 10 Mbps links: must split 50/50, theta = 0.4
th, x, ms = solve_minmax(links, [dict(demand=8.0, paths=paths)], {}, 10.0)
print("test1 theta=%.3f x=%s (%.1f ms)" % (th, x[0].round(3), ms))
assert abs(th - 0.4) < 1e-3 and abs(x[0][0] - 0.5) < 0.02

# 2) path via node 2 already 50% loaded: all 4 Mbps should go via node 3, theta = 0.5 (bg dominates)
th, x, _ = solve_minmax(links, [dict(demand=4.0, paths=paths)], {(1, 2): 0.5}, 10.0)
print("test2 theta=%.3f x=%s" % (th, x[0].round(3)))
assert abs(th - 0.5) < 1e-3 and x[0][1] > 0.98

# 3) heavy background on one branch, partial shift: bg 0.6 on (1,2); 8 Mbps total
th, x, _ = solve_minmax(links, [dict(demand=8.0, paths=paths)], {(1, 2): 0.6}, 10.0)
print("test3 theta=%.3f x=%s" % (th, x[0].round(3)))
assert abs(th - 0.7) < 1e-2 and x[0][1] > 0.8
print("LP OK")