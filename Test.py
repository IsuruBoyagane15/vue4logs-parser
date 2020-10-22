from Vue4logsParser import *

if __name__ == '__main__':
        t = 0.1
        while (t<1):
            pas = []
            for dataset, setting in benchmark_settings.items():
                parser = Vue4Logs(t, dataset)
                pa = parser.parse()
                pas.append(pa)
                # print(dataset, t, pa)
            print(t, sum(pas)/16.0)
            t = round(t+0.1, 2)
            pas = []
