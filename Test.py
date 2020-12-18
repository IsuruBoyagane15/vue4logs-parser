from Vue4logsParser import *

"""
Run experiments by with parameter update
"""
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
            ground_truth_df = 'ground_truth/' + dataset + '_2k.log_structured.csv'
            output = "results/" + str(t) + "/" + dataset + "_structured.csv"
            pa = evaluate(ground_truth_df, output)[1]
            PAs.append(pa)

        print(t, sum(PAs) / 16.0)
        BENCHMARK[t] = PAs
        t = round(t + float(step), 2)
    print(BENCHMARK)
    BENCHMARK.to_csv('results/' + sys.argv[2] + '.csv', index=False)
