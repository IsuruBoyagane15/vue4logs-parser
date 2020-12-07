from Vue4logsParser import *

if __name__ == '__main__':
    type = sys.argv[1]
    try:
        dataset_chosen = sys.argv[2]
    except IndexError:
        dataset_chosen = None
    pas = []
    for dataset, setting in benchmark_settings.items():
        if dataset_chosen is not None and dataset != dataset_chosen:
            continue
        if type == '1':
            threshold = 0.78
        elif type == '0':
            threshold = benchmark_settings[dataset]['threshold']
        else:
            print("Error in arguments.")
            sys.exit(0)

        parser = Vue4Logs(threshold, dataset)
        parser.parse()
        ground_truth_df = 'ground_truth/' + dataset + '_2k.log_structured.csv'
        output = "results/" + str(threshold) + "/" + dataset + "_structured.csv"
        pa = evaluate(ground_truth_df, output)[1]
        pas.append(pa)

    print(sum(pas)/16.0)
