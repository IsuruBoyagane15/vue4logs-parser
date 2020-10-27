import os.path as path
import re
import os

from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from Evaluate import *
from inverted_index.BiWordInvertedIndex import BiWordInvertedIndex
from inverted_index.VanillaInvertedIndex import *

benchmark_settings = {
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
        'threshold': 0.23
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
input_dir = 'logs/'


def filter_wildcards(processed_log):
    filtered_token_list = []
    for current_token in processed_log:
        if "<*>" not in current_token:
            filtered_token_list.append(current_token)

    return filtered_token_list


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


def my_tokenizer(text):
    return text


def replace_alpha_nums(preprocessed_log):
    for i, token in enumerate(preprocessed_log):
        alpha_numeric_regex = r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$'
        is_alpha_numeric = re.search(alpha_numeric_regex, token)
        if is_alpha_numeric:
            preprocessed_log[i] = re.sub(alpha_numeric_regex, '<*>', token)

    return preprocessed_log


def check_numeric(token):
    return_token = ""
    for i in range(0, len(token)):
        if not token[i].isnumeric():
            return_token += token[i]
        else:
            return_token += '<*>'
    wildcard_check = re.compile('(?:\<\*\>)+')
    return re.sub(wildcard_check, '<*>', return_token)


def replace_nums(preprocessed_log):
    for i, token in enumerate(preprocessed_log):
        preprocessed_log[i] = check_numeric(token)
    return preprocessed_log


def replace_only_nums(preprocessed_log):
    for i, token in enumerate(preprocessed_log):
        if token.isnumeric():
            preprocessed_log[i] = '<*>'

    return preprocessed_log


def get_tfidf(doc_ids, temp):
    corpus = [temp[i] for i in doc_ids]
    filtered_corpus = list(map(lambda x: filter_wildcards(x), corpus))
    vectorizer = TfidfVectorizer(lowercase=False, analyzer='word', stop_words=None, tokenizer=my_tokenizer,
                                 token_pattern=None)

    vectors = vectorizer.fit_transform(filtered_corpus).toarray()
    vectors = [vectors[i].tolist() for i in range(len(corpus))]
    return cosine_similarity(vectors)


class Vue4Logs:
    def __init__(self, threshold, dataset):
        self.threshold = threshold
        self.templates = {}
        self.inverted_index = VanillaInvertedIndex()
        self.results = []
        self.dataset = dataset
        self.output_path = "results/" + str(threshold)

    def get_new_template(self, temp_template):
        if len(self.templates.keys()) == 0:
            next_id = 0
        else:
            next_id = max(self.templates.keys()) + 1
        # print("NEXT TEMPLATE ID :", next_id)
        self.templates[next_id] = temp_template
        self.results.append(next_id)
        return next_id

    def write_results(self):
        df = pd.read_csv('ground_truth/' + self.dataset + '_2k.log_structured.csv')
        df['EventId'] = ["E" + str(i) for i in self.results]
        templates_df = []
        for j in self.results:
            if int(j) > 2000:
                print("Error in result")
                sys.exit(0)
            else:
                templates_df.append(" ".join(self.templates[j]))
        df['EventTemplate'] = templates_df

        if not path.exists(self.output_path):
            os.makedirs(self.output_path)
        df.to_csv(self.output_path + '/' + self.dataset + '_structured.csv')

    def preprocess(self, line):
        regex = benchmark_settings[self.dataset]['regex']
        for currentRex in regex:
            line = re.sub(currentRex, '<*>', line)
        return line

    def get_bm25(self, doc_ids):
        logs = []
        for i in doc_ids:
            if i == -1:
                query = self.templates[i]
            else:
                logs.append(self.templates[i])
        bm25 = BM25Okapi(logs)
        doc_scores = bm25.get_scores(query)
        return doc_scores

    def parse(self):
        dataset_config = benchmark_settings[self.dataset]
        indir = os.path.join(input_dir, os.path.dirname(dataset_config['log_file']))
        log_file = os.path.basename(dataset_config['log_file'])
        headers, regex = generate_logformat_regex(dataset_config['log_format'])
        df_log = log_to_dataframe(indir + '/' + log_file, regex, headers)
        for idx, line in df_log.iterrows():
            log_id = line['LineId']
            pre_processed_log = self.preprocess(line['Content']).strip().split()
            # print(logID, pre_processed_log)

            pre_processed_log = replace_nums(pre_processed_log)

            hits = self.inverted_index.search_doc(pre_processed_log)

            if len(hits) == 0:
                new_id = self.get_new_template(pre_processed_log)
                self.inverted_index.index_doc(new_id, self.templates[new_id])

            else:
                candidates = {key: self.templates[key] for key in hits}
                length_filtered_candidates = {key: candidates[key] for key in candidates if
                                              len(candidates[key]) == len(pre_processed_log)}
                remaining_hits = list(length_filtered_candidates.keys())

                if len(length_filtered_candidates) == 0:
                    new_id = self.get_new_template(pre_processed_log)
                    self.inverted_index.index_doc(new_id, self.templates[new_id])
                else:

                    greedily_found = False
                    for hit in remaining_hits:
                        if pre_processed_log == self.templates[hit]:
                            # print("greedy catch")
                            self.results.append(hit)
                            greedily_found = True

                    if greedily_found:
                        continue
                    # print("more rules")

                    max_similarity = 0
                    selected_candidate_id = None

                    similarity_candidates = {key: self.templates[key] for key in length_filtered_candidates}
                    similarity_candidates[-1] = pre_processed_log
                    doc_ids = [-1] + list(length_filtered_candidates.keys())

                    similarity = get_tfidf(doc_ids, similarity_candidates)[0]
                    # similarity = self.get_bm25(doc_ids)

                    for i in range(len(similarity)):
                        if i == 0:
                            continue
                        else:
                            current_similarity = similarity[i]
                            if current_similarity > max_similarity:
                                max_similarity = current_similarity
                                selected_candidate_id = remaining_hits[i - 1]

                    if max_similarity < self.threshold:
                        new_id = self.get_new_template(pre_processed_log)
                        self.inverted_index.index_doc(new_id, self.templates[new_id])
                    else:
                        selected_candidate = self.templates[selected_candidate_id]
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
                        self.inverted_index.update_doc(selected_candidate_id, self.templates[selected_candidate_id], updated_template)

                        self.templates[selected_candidate_id] = updated_template
                        self.results.append(selected_candidate_id)
                assert len(self.results) == log_id
        print(self.dataset)
        self.write_results()
        ground_truth_df = 'ground_truth/' + self.dataset + '_2k.log_structured.csv'
        output = self.output_path + "/" + self.dataset + "_structured.csv"
        pa = evaluate(ground_truth_df, output)[1]
        # print(self.dataset, pa)
        # print(self.inverted_index.dict)
        # print(self.dataset, len(self.templates), "\n")
        return pa


