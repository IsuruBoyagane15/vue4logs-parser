from Vue4logsParser import *

if __name__ == '__main__':
        t = 0.1
        # s1 = {}
        # s2 = {}
        # s3 = {}
        BENCHMARK = pd.DataFrame()
        BENCHMARK['Dataset'] = list(benchmark_settings.keys())
        # d = 'Mac'
        while (t<0.3):
            pas = []
            for dataset, setting in benchmark_settings.items():
                # if dataset == d:
                parser = Vue4Logs(t, dataset)
                pa, s_1, s_2, s_3 = parser.parse()
                pas.append(pa)
                    # try:
                    #     s1[t].append(s_1)
                    #     s2[t].append(s_2)
                    #     s3[t].append(s_3)
                    # except KeyError:
                    #     s1[t] = s_1
                    #     s2[t] = s_2
                    #     s3[t] = s_3

                # print(dataset, t, pa)

            # print(s2)
            # print(s3)
            print(t, sum(pas)/16.0)
            t = round(t+0.1, 2)

            BENCHMARK[t] = pas
        BENCHMARK.to_csv('results/' + 'final' + '.csv', index=False)
        # print(s1)
        # df_1=pd.DataFrame(s1)
        # df_1.to_csv('stage_1/' + d + '.csv', index=False)
        # print(s2)
        # df_2 = pd.DataFrame(s2)
        # df_2.to_csv('stage_2/' + d + '.csv', index=False)
        # print(s3)
        # df_3 = pd.DataFrame(s3)
        # df_3.to_csv('stage_3/' + d + '.csv', index=False)