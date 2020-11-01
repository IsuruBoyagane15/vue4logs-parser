import pandas as pd
import scipy.special
import sys


def evaluate(ground_truth, parsed_result):
    """
    Given the output structured file and ground truth structured file, return the f1 measure of clustering and
    parsing accuracy of log parsing.
    :param ground_truth: ground truth file location
    :param parsed_result: output structured file location
    :return: f1 measure of clustering and parsing accuracy
    """
    df_ground_truth = pd.read_csv(ground_truth)
    df_parsed_log = pd.read_csv(parsed_result)

    # Remove invalid ground truth event Templates
    null_log_ids = df_ground_truth[~df_ground_truth['EventTemplate'].isnull()].index
    df_ground_truth = df_ground_truth.loc[null_log_ids]
    df_parsed_log = df_parsed_log.loc[null_log_ids]

    (precision, recall, f_measure, accuracy) = get_accuracy(df_ground_truth['EventTemplate'],
                                                            df_parsed_log['EventTemplate'])

    print('Precision: %.4f, Recall: %.4f, F1_measure: %.4f, Parsing_Accuracy: %.4f' % (
        precision, recall, f_measure, accuracy))
    return f_measure, accuracy


def get_accuracy(series_ground_truth, series_parsedlog, debug=False):
    """
    Given the ground truth and output templates as Pandas series calculate precision, recall, f1-measure and parsing
    accuracy
    :param series_ground_truth: ground truth templates as Pandas series
    :param series_parsedlog: output templates as Pandas series
    :param debug: configuration parameter to debug or not
    :return: precision, recall, f1-measure and parsing accuracy
    """
    series_ground_truth_value_counts = series_ground_truth.value_counts()

    real_pairs = 0  # number of all possible combination given the ground truth
    for count in series_ground_truth_value_counts:
        if count > 1:
            real_pairs += scipy.special.comb(count, 2)

    series_parsed_log_value_counts = series_parsedlog.value_counts()

    parsed_pairs = 0
    for count in series_parsed_log_value_counts:
        if count > 1:
            parsed_pairs += scipy.special.comb(count, 2)

    accurate_pairs = 0
    accurate_events = 0  # determine how many lines are correctly parsed
    for parsed_eventId in series_parsed_log_value_counts.index:
        log_ids = series_parsedlog[series_parsedlog == parsed_eventId].index # log ids that are given parsed_eventId in the results output file
        series_ground_truth_logId_value_counts = series_ground_truth[log_ids].value_counts()
        error_eventIds = (parsed_eventId, series_ground_truth_logId_value_counts.index.tolist())
        error = True
        if series_ground_truth_logId_value_counts.size == 1:
            ground_truth_event_id = series_ground_truth_logId_value_counts.index[0]
            if log_ids.size == series_ground_truth[series_ground_truth == ground_truth_event_id].size:
                accurate_events += log_ids.size
                error = False
        if error and debug:
            print('(parsed_eventId, ground_truth_event_id) =', error_eventIds, 'failed', log_ids.size, 'messages')
        for count in series_ground_truth_logId_value_counts:
            if count > 1:
                accurate_pairs += scipy.special.comb(count, 2)

    precision = float(accurate_pairs) / parsed_pairs
    recall = float(accurate_pairs) / real_pairs
    f_measure = 2 * precision * recall / (precision + recall)
    accuracy = float(accurate_events) / series_ground_truth.size
    return precision, recall, f_measure, accuracy


if __name__ == '__main__':
    DATASET = sys.argv[1]
    THRESHOLD = sys.argv[2]
    TYPE = sys.argv[3]

    ground_truth_df = 'ground_truth/' + DATASET + '_2k.log_structured.csv'
    output = "results/" + TYPE + "/" + THRESHOLD + "/" + DATASET + "_structured.csv" # results_ is for manual inverted index results
    evaluate(ground_truth_df, output)
