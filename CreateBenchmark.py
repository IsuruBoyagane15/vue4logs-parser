from Vue4logsParser import *

import random
random.seed(3)

if __name__ == '__main__':
    type = sys.argv[1]
    try:
        dataset_chosen = sys.argv[2]
    except IndexError:
        dataset_chosen = None
    pas = []

    for dataset, setting in benchmark_settings.items():
        ground_truth_df = 'ground_truth/' + dataset + '_2k.log_structured.csv'

        if dataset_chosen is not None and dataset != dataset_chosen:
            continue

        if type == '1':
            threshold = 0.6
            parser = Vue4Logs(threshold, dataset)
            parser.parse()
            output = "results/" + str(threshold) + "/" + dataset + "_structured.csv"
            pa = evaluate(ground_truth_df, output)[1]
            pas.append(pa)

        elif type == '0':
            highest_pa_for_dataset = 0
            selected_threshold = None

            for i in range(10):
                random_threshold = round(random.random(), 2)
                parser = Vue4Logs(random_threshold, dataset)
                parser.parse()
                output = "results/" + str(random_threshold) + "/" + dataset + "_structured.csv"
                pa = evaluate(ground_truth_df, output)[1]
                if pa > highest_pa_for_dataset:
                    highest_pa_for_dataset = pa
                    selected_threshold = random_threshold

            selected_output = "results/" + str(selected_threshold) + "/" + dataset + "_structured.csv"
            selected_pa = evaluate(ground_truth_df, selected_output)[1]
            print(dataset, selected_pa)
            pas.append(selected_pa)

        else:
            print("Error in arguments.")
            sys.exit(0)

    print(sum(pas)/16.0)
