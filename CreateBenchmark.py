from Vue4logsParser import *

if __name__ == '__main__':
    type = sys.argv[1]
    pas = []
    for dataset, setting in benchmark_settings.items():
        if type == '1':
            parser = Vue4Logs(0.78, dataset)
        elif type == '0':
            parser = Vue4Logs(benchmark_settings[dataset]['threshold'], dataset)
        else:
            print("Error in arguments.")

        pa = parser.parse()
        pas.append(pa)

    print(sum(pas)/16.0)
