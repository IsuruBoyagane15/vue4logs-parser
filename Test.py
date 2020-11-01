from Vue4logsParser import *

if __name__ == '__main__':
    step = sys.argv[1]
    t = float(step)
    BENCHMARK = pd.DataFrame()
    BENCHMARK['Dataset'] = list(benchmark_settings.keys())
    while t < 1:
        PAs = []
        print("Threshold : ", t)
        for dataset, setting in benchmark_settings.items():
            parser = Vue4Logs(t, dataset)
            pa = parser.parse()
            PAs.append(pa)

        print(t, sum(PAs) / 16.0)
        BENCHMARK[t] = PAs
        t = round(t + float(step), 2)
    print(BENCHMARK)
    BENCHMARK.to_csv('results/' + sys.argv[2] + '.csv', index=False)
