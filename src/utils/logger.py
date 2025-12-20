class Logger:
    def __init__(self, log_file='training.log'):
        self.log_file = log_file

    def log(self, message):
        with open(self.log_file, 'a') as f:
            f.write(f"{message}\n")

    def log_metrics(self, metrics):
        with open(self.log_file, 'a') as f:
            for key, value in metrics.items():
                f.write(f"{key}: {value}\n")

    def clear_log(self):
        open(self.log_file, 'w').close()