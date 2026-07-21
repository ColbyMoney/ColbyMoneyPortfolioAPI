import multiprocessing

max_requests = 1000
max_requests_jitter = 50
log_file = "-"
bind = "0.0.0.0:8000"
timeout = 300
num_cpus = multiprocessing.cpu_count()
workers = 1
worker_class = "uvicorn.workers.UvicornWorker"