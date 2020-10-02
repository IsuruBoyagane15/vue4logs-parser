import os.path as path
import re
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from evaluate import *

TEMPLATES = {}
RESULTS = []
LAMBDA_1 = 0.85
LAMBDA_2 = 1 - LAMBDA_1
INVERTED_INDEX = {}
BENCHMARK_SETTINGS = {
    'HDFS': {
        'log_file': 'HDFS/HDFS_2k.log',
        'log_format': '<Date> <Time> <Pid> <Level> <Component>: <Content>',
        'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?'],
        'threshold': 0.27
    },

    'Hadoop': {
        'log_file': 'Hadoop/Hadoop_2k.log',
        'log_format': '<Date> <Time> <Level> \[<Process>\] <Component>: <Content>',
        'regex': [r'(\d+\.){3}\d+'],
        'threshold': 0.77
    },

    'Spark': {
        'log_file': 'Spark/Spark_2k.log',
        'log_format': '<Date> <Time> <Level> <Component>: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+'],
        'threshold': 0.67
    },

    'Zookeeper': {
        'log_file': 'Zookeeper/Zookeeper_2k.log',
        'log_format': '<Date> <Time> - <Level>  \[<Node>:<Component>@<Id>\] - <Content>',
        'regex': [r'(/|)(\d+\.){3}\d+(:\d+)?'],
        'threshold': 0.64
    },

    'BGL': {
        'log_file': 'BGL/BGL_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <Node> <Time> <NodeRepeat> <Type> <Component> <Level> <Content>',
        'regex': [r'core\.\d+'],
        'threshold': 0.43
    },

    'HPC': {
        'log_file': 'HPC/HPC_2k.log',
        'log_format': '<LogId> <Node> <Component> <State> <Time> <Flag> <Content>',
        'regex': [r'=\d+'],
        'threshold': 0.34
    },

    'Thunderbird': {
        'log_file': 'Thunderbird/Thunderbird_2k.log',
        'log_format': '<Label> <Timestamp> <Date> <User> <Month> <Day> <Time> <Location> <Component>(\[<PID>\])?: <Content>',
        'regex': [r'(\d+\.){3}\d+'],
        'threshold': 0.27
    },

    'Windows': {
        'log_file': 'Windows/Windows_2k.log',
        'log_format': '<Date> <Time>, <Level>                  <Component>    <Content>',
        'regex': [r'0x.*?\s'],
        'threshold': 0.67
    },

    'Linux': {
        'log_file': 'Linux/Linux_2k.log',
        'log_format': '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}'],
        'threshold': 0.42
    },

    'Android': {
        'log_file': 'Android/Android_2k.log',
        'log_format': '<Date> <Time>  <Pid>  <Tid> <Level> <Component>: <Content>',
        'regex': [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+', r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b'],
        'threshold': 0.78
    },

    'HealthApp': {
        'log_file': 'HealthApp/HealthApp_2k.log',
        'log_format': '<Time>\|<Component>\|<Pid>\|<Content>',
        'regex': [],
        'threshold': 0.34
    },

    'Apache': {
        'log_file': 'Apache/Apache_2k.log',
        'log_format': '\[<Time>\] \[<Level>\] <Content>',
        'regex': [r'(\d+\.){3}\d+'],
        'threshold': 0.21
    },

    'Proxifier': {
        'log_file': 'Proxifier/Proxifier_2k.log',
        'log_format': '\[<Time>\] <Program> - <Content>',
        'regex': [r'<\d+\ssec', r'([\w-]+\.)+[\w-]+(:\d+)?', r'\d{2}:\d{2}(:\d{2})*', r'[KGTM]B'],
        'threshold': 0.78
    },

    'OpenSSH': {
        'log_file': 'OpenSSH/OpenSSH_2k.log',
        'log_format': '<Date> <Day> <Time> <Component> sshd\[<Pid>\]: <Content>',
        'regex': [r'(\d+\.){3}\d+', r'([\w-]+\.){2,}[\w-]+'],
        'threshold': 0.59
    },

    'OpenStack': {
        'log_file': 'OpenStack/OpenStack_2k.log',
        'log_format': '<Logrecord> <Date> <Time> <Pid> <Level> <Component> \[<ADDR>\] <Content>',
        'regex': [r'((\d+\.){3}\d+,?)+', r'/.+?\s', r'\d+'],
        'threshold': 0.67
    },

    'Mac': {
        'log_file': 'Mac/Mac_2k.log',
        'log_format': '<Month>  <Date> <Time> <User> <Component>\[<PID>\]( \(<Address>\))?: <Content>',
        'regex': [r'([\w-]+\.){2,}[\w-]+'],
        'threshold': 0.68
    },
}


def filter_wildcards(processed_log):
    filtered_token_list = []
    for current_token in processed_log:
        if "<*>" not in current_token:
            filtered_token_list.append(current_token)

    return filtered_token_list


def search_index(query_log):
    # print("SEARCH QUERY : ", query_log)
    hits = []
    for token in query_log:
        if token in INVERTED_INDEX:
            hits += INVERTED_INDEX[token]
    hit_set = set(hits)
    return list(hit_set)


def index_doc(doc_id):
    new_template = TEMPLATES[doc_id]

    template_length = len(new_template)
    # print(new_template)

    for i in range(template_length):
        token = new_template[i]
        if token in INVERTED_INDEX:
            INVERTED_INDEX[token].append(doc_id)
        else:
            INVERTED_INDEX[token] = [doc_id]


def update_doc(tokens_to_remove, doc_id):
    for token in tokens_to_remove:
        if token in INVERTED_INDEX:
            if doc_id in INVERTED_INDEX[token]:
                INVERTED_INDEX[token].remove(doc_id)
        #     else:
        #         print("DOC TO UPDATE DOES NOT HAVE THE TOKEN :", token)
        # else:
        #     print("TOKEN TO REMOVE DOES NOT EXIST IN THE INDEX :", token)


def get_new_template(temp_template):
    if len(TEMPLATES.keys()) == 0:
        next_id = 0
    else:
        next_id = max(TEMPLATES.keys()) + 1
    # print("NEXT TEMPLATE ID :", next_id)
    TEMPLATES[next_id] = temp_template
    RESULTS.append(next_id)
    return next_id


def write_results():
    df = pd.read_csv('ground_truth/' + DATASET + '_2k.log_structured.csv')
    df['EventId'] = ["E" + str(i) for i in RESULTS]
    templates_df = []
    for j in RESULTS:
        if int(j) > 2000:
            print("Error in result")
            sys.exit(0)
        else:
            templates_df.append(TEMPLATES[j])
    df['EventTemplate'] = templates_df

    if not path.exists('results/' + BENCHMARK_NAME + '/' + str(THRESHOLD)):
        os.makedirs('results/' + BENCHMARK_NAME + '/' + str(THRESHOLD))
    df.to_csv('results/' + BENCHMARK_NAME + '/' + str(THRESHOLD) + '/' + DATASET + '_structured.csv')


def length(template, log_message):
    message_length = len(log_message)
    template_length = len(template)

    diff = abs(message_length - template_length)
    maximum = max(message_length, template_length)

    length_feature = 1 - float(diff) / maximum

    return length_feature


def jaccard(template, log_message):
    filtered_log_tokens = filter_wildcards(log_message)
    filtered_template_tokens = filter_wildcards(template)

    log_token_set = set(filtered_log_tokens)
    template_token_set = set(filtered_template_tokens)

    intersection = log_token_set.intersection(template_token_set)
    union = log_token_set.union(template_token_set)

    return float(len(intersection) / len(union))


def simSiq(template, log_message):
    log_line_token_list = log_message.split(" ")
    template_token_list = template.split(" ")
    decided_length = min(len(log_line_token_list), len(template_token_list))

    sim_seq_sum = 0

    for i in range(decided_length):
        if log_line_token_list[i] == template_token_list[i]:
            sim_seq_sum += 1
    sim_seq = sim_seq_sum / float(decided_length)
    return sim_seq


def similarity(template, log_message):
    length_feature = length(template, log_message)

    if TYPE == 'baseline':
        similarity = jaccard(template, log_message)
    elif TYPE == 'sim_seq':
        similarity = simSiq(template, log_message)
    else:
        print("Error in configs")
        sys.exit(0)

    return LAMBDA_1 * length_feature + LAMBDA_2 * similarity


def generate_logformat_regex(logformat):
    headers = []
    splitters = re.split(r'(<[^<>]+>)', logformat)
    regex = ''
    for k in range(len(splitters)):
        if k % 2 == 0:
            splitter = re.sub(' +', '\\\s+', splitters[k])
            regex += splitter
        else:
            header = splitters[k].strip('<').strip('>')
            regex += '(?P<%s>.*?)' % header
            headers.append(header)
    regex = re.compile('^' + regex + '$')
    return headers, regex


def log_to_dataframe(log_file, regex, headers):
    log_messages = []
    linecount = 0
    with open(log_file, 'r') as fin:
        for line in fin.readlines():
            try:
                match = regex.search(line.strip())
                message = [match.group(header) for header in headers]
                log_messages.append(message)
                linecount += 1
            except Exception as e:
                pass
    logdf = pd.DataFrame(log_messages, columns=headers)
    logdf.insert(0, 'LineId', None)
    logdf['LineId'] = [i + 1 for i in range(linecount)]
    return logdf


def preprocess(dataset, line):
    regex = BENCHMARK_SETTINGS[dataset]['regex']
    for currentRex in regex:
        line = re.sub(currentRex, '<*>', line)
    return line


def replace_alpha_nums(preprocessed_log):
    for i, token in enumerate(preprocessed_log):
        alpha_numeric_regex = r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$'
        is_alpha_numeric = re.search(alpha_numeric_regex, token)
        if is_alpha_numeric:
            pre_processed_log[i] = re.sub(alpha_numeric_regex, '<*>', token)

    return pre_processed_log


def my_tokenizer(text):
    return text


def get_tfidf(doc_ids, temp):
    corpus = [temp[i] for i in doc_ids]
    filtered_corpus = list(map(lambda x: filter_wildcards(x), corpus))
    vectorizer = TfidfVectorizer(lowercase=False, analyzer='word', stop_words=None, tokenizer=my_tokenizer,
                                 token_pattern=None)

    vectors = vectorizer.fit_transform(filtered_corpus).toarray()
    vectors = [vectors[i].tolist() for i in range(len(corpus))]
    return cosine_similarity(vectors)


if __name__ == '__main__':

    TYPE = sys.argv[1]
    BENCHMARK_NAME = TYPE
    BENCHMARK = pd.DataFrame()
    BENCHMARK['Dataset'] = list(BENCHMARK_SETTINGS.keys())
    input_dir = 'logs/'

    PAs = []

    for DATASET, setting in BENCHMARK_SETTINGS.items():

        indir = os.path.join(input_dir, os.path.dirname(setting['log_file']))
        log_file = os.path.basename(setting['log_file'])
        headers, regex = generate_logformat_regex(setting['log_format'])
        df_log = log_to_dataframe(indir + '/' + log_file, regex, headers)

        if TYPE == '0':
            THRESHOLD = BENCHMARK_SETTINGS[DATASET]['threshold']
        elif TYPE == '1':
            THRESHOLD = 0.6
        else:
            print("Error in configs")
            sys.exit(0)

        for idx, line in df_log.iterrows():
            logID = line['LineId']
            pre_processed_log = preprocess(DATASET, line['Content']).strip().split()
            # print(logID, pre_processed_log)
            pre_processed_log = replace_alpha_nums(pre_processed_log)

            log_line = replace_alpha_nums(pre_processed_log)
            log_line = filter_wildcards(log_line)
            # print("FILTERED LOG LINE :", log_line)

            hits = search_index(log_line)
            found = False
            if len(hits) > 0:
                for hit in hits:
                    if pre_processed_log == TEMPLATES[hit]:
                        # print("early catch")
                        RESULTS.append(hit)
                        found = True

            if found:
                continue
            # print("more rules")

            # IF NO CANDIDATE FOUND
            if len(hits) == 0:
                # print("NO HITS")
                new_id = get_new_template(pre_processed_log)
                index_doc(new_id)

            # IF THERE IS AT LEAST ONE CANDIDATE
            else:
                candidates = {key: TEMPLATES[key] for key in hits}
                length_filtered_candidates = {key: candidates[key] for key in candidates if
                                              len(candidates[key]) == len(pre_processed_log)}
                if len(length_filtered_candidates) == 0:
                    new_id = get_new_template(pre_processed_log)
                    index_doc(new_id)
                else:
                    candidate_found = False
                    for i in length_filtered_candidates:
                        if pre_processed_log == TEMPLATES[i]:
                            RESULTS.append(i)
                            candidate_found = True
                            break

                    if not candidate_found:
                        max_similarity = 0
                        selected_candidate_id = None
                        remaining_hits = list(length_filtered_candidates.keys())

                        TEMPLATES[-1] = pre_processed_log
                        doc_ids = [-1]
                        for hit in length_filtered_candidates:
                            doc_ids.append(hit)

                        similarity = get_tfidf(doc_ids, TEMPLATES)[0]

                        TEMPLATES[-1] = None
                        for i in range(len(similarity)):
                            if i == 0:
                                continue
                            else:
                                current_similarity = similarity[i]
                                if current_similarity > max_similarity:
                                    max_similarity = current_similarity
                                    selected_candidate_id = remaining_hits[i - 1]

                        if max_similarity < THRESHOLD:
                            new_id = get_new_template(pre_processed_log)
                            index_doc(new_id)
                        else:
                            selected_candidate = TEMPLATES[selected_candidate_id]

                            if pre_processed_log == selected_candidate:
                                # print("SELECTED TEMPLATE IS EQUAL TO LOG LINE")
                                RESULTS.append(selected_candidate_id)
                            else:
                                template_length = len(selected_candidate)
                                # print("SELECTED TEMPLATE IS not EQUAL TO LOG LINE")
                                temporary_tokens = []
                                changed_tokens = []

                                for index in range(template_length):
                                    # if log_line_token_list[position] == candidate_token_list[position]:
                                    if pre_processed_log[index] == selected_candidate[index] or \
                                            "<*>" in selected_candidate[index]:
                                        temporary_tokens.append(selected_candidate[index])
                                    else:
                                        changed_tokens.append(selected_candidate[index])
                                        temporary_tokens.append("<*>")
                                updated_template = temporary_tokens
                                update_doc(changed_tokens, selected_candidate_id)

                                TEMPLATES[selected_candidate_id] = updated_template
                                RESULTS.append(selected_candidate_id)

                assert len(RESULTS) == logID

            # print("==== TEMPLATES ====")
            # for t in TEMPLATES:
            #     print(TEMPLATES[t])
            # print("\n")

        write_results()

        ground_truth_df = 'ground_truth/' + DATASET + '_2k.log_structured.csv'
        output = "results/" + BENCHMARK_NAME + "/" + str(
            THRESHOLD) + "/" + DATASET + "_structured.csv"  # results_ is for manual inverted index results
        print(DATASET + ' :', end='')
        pa = evaluate(ground_truth_df, output)[1]
        PAs.append(pa)

        RESULTS = []
        INVERTED_INDEX = {}
        TEMPLATES = {}

    BENCHMARK['PA'] = PAs
    print(BENCHMARK)
    print("Average :", round(sum(PAs) / 16.0, 4))
    BENCHMARK.to_csv('results/' + BENCHMARK_NAME + '.csv', index=False)
