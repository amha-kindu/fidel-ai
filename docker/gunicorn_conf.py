import multiprocessing
workers = int(multiprocessing.cpu_count() / 2) or 1
bind = "0.0.0.0:2345"
accesslog = "-"
errorlog = "-"
loglevel = "info"
worker_tmp_dir = "/dev/shm"
# Tune keepalive to play nice with SSE
keepalive = 75