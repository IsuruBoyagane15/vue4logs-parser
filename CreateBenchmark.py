from Vue4logsParser import *

if __name__ == '__main__':
    for dataset, setting in benchmark_settings.items():
        parser = Vue4Logs(0.78, dataset)
        parser.parse()