from Vue4logsParser import *
import time
benchmark_settings = {
#     'HDFS': {
#         'log_file': 'HDFS/HDFS',
#         'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
#         'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
#         'similarity_threshold' :0.54,
#         'sizes' : ['300K', '1M', '10M', '100M', '500M', '1G']
#
# },

    'BGL': {
        'log_file': 'BGL/BGL',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'regex': [r'core\.\d+'],
        'similarity_threshold' : 0.59,
        'sizes' : ['300K', '1M', '10M', '100M', '500M']

},

    # 'Android': {
    #     'log_file': 'Android/Android',
    #     'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
    #     'regex': [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+', r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b'],
    #     'similarity_threshold': 0.96,
    #     'sizes': ['300K', '1M', '10M', '100M']
    #
    # },
}

if __name__ == '__main__':

    for dataset, setting in benchmark_settings.items():
        with open('logs/'+dataset + ".log", "r") as f1:
            content = f1.read().split("\n")
        dataset_config = benchmark_settings[dataset]

        # android
        # for index, size in enumerate(dataset_config['sizes']):
        #     count = None
        #     if size == "1M":
        #         count = 9000
        #     if size == "10M":
        #         count = 85000
        #     if size == "100M":
        #         count = 810000
        #     if size == "500M":
        #         count = 3450000
        #     if size == "1G":
        #         count = 7160000
        #     if size == "300K":
        #         continue

        # bgl
        for index, size in enumerate(dataset_config['sizes']):
            count = None
            if size == "1M":
                count = 7000
            if size == "10M":
                count = 74000
            if size == "100M":
                count = 710000
            if size == "500M":
                count = 3450000
            if size == "1G":
                count = 7160000
            if size == "300K":
                continue


        # hdfs
        # for index, size in enumerate(dataset_config['sizes']):
        #     count = None
        #     if size == "1M":
        #         count = 7200
        #     if size == "10M":
        #         count = 70000
        #     if size == "100M":
        #         count = 712000
        #     if size == "500M":
        #         count = 3560000
        #     if size == "1G":
        #         count = 7160000
        #     if size == "300K":
        #         continue
        #


            with open("logs/" + dataset + "/" + dataset  + "_" + size + ".log", "w") as f2:
                f2.write(("\n").join(content[0:count]))
            print(size)

        indir = os.path.join(input_dir, os.path.dirname(dataset_config['log_file']))
        headers, regex = generate_logformat_regex(dataset_config['log_format'])
        threshold = dataset_config['similarity_threshold']
        parser = Vue4Logs(dataset, threshold)
        perfBench = {}
        print(dataset,)

        # for index, size in enumerate(dataset_config['sizes']):
        #     log_file = os.path.basename(dataset_config['log_file'] + "/" + dataset + "_" + size + ".log")
        #     df_log = log_to_dataframe(indir + '/' + log_file, regex, headers)
        #     start = time.time()
        #     parser.parse(df_log)
        #     stop = time.time()
        #     dur = (stop - start)/60.0
        #     print(size, dur)

