import os.path as path
import re
import os

from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from Evaluate import *
from inverted_index.VanillaInvertedIndex import *

# Configurations for benchmark datasets taken from https://github.com/logpai/logparser
benchmark_settings = {
    'HDFS': {
        'regex': [r'blk_-?\d+', r'(\d+\.){3}\d+(:\d+)?']
    },

    'Hadoop': {
        'regex': [r'(\d+\.){3}\d+']
    },

    'Spark': {
        'regex': [r'(\d+\.){3}\d+', r'\b[KGTM]?B\b', r'([\w-]+\.){2,}[\w-]+']
    },

    'Zookeeper': {
        'regex': [r'(/|)(\d+\.){3}\d+(:\d+)?']
    },

    'BGL': {
        'regex': [r'core\.\d+']
    },

    'HPC': {
        'regex': [r'=\d+']
    },

    'Thunderbird': {
        'regex': [r'(\d+\.){3}\d+']
    },

    'Windows': {
        'regex': [r'0x.*?\s']
    },

    'Linux': {
        'regex': [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}']
    },

    'Android': {
        'regex': [r'(/[\w-]+)+', r'([\w-]+\.){2,}[\w-]+', r'\b(\-?\+?\d+)\b|\b0[Xx][a-fA-F\d]+\b|\b[a-fA-F\d]{4,}\b']
    },

    'HealthApp': {
        'regex': []
    },

    'Apache': {
        'regex': [r'(\d+\.){3}\d+']
    },

    'Proxifier': {
        'regex': [r'<\d+\ssec', r'([\w-]+\.)+[\w-]+(:\d+)?', r'\d{2}:\d{2}(:\d{2})*', r'[KGTM]B']
    },

    'OpenSSH': {
        'regex': [r'(\d+\.){3}\d+', r'([\w-]+\.){2,}[\w-]+']
    },

    'OpenStack': {
        'regex': [r'((\d+\.){3}\d+,?)+', r'/.+?\s', r'\d+']
    },

    'Mac': {
        'regex': [r'([\w-]+\.){2,}[\w-]+']
    }
}
input_dir = 'logs/'


def generate_logformat_regex(logformat):
    """
    Extract the log message and headers from raw log line by using the configuration logformat

    :param logformat: Format of header fields present in the particular dataset
    :return: log message and headers
    """
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
    """
    Convert log data in a logfile into a Pandas dataframe

    :param log_file: raw log file location
    :param regex: Regex to seperate
    :param headers: headers present in a log line
    :return: pandas dataframe containing log messages
    """
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
    """
    Dummy tokenizer to be used in tf-idf
    :param text: pre-tokenized text
    :return: pre-tokenized text
    """
    return text


def replace_alpha_nums(preprocessed_log):
    """
    Replace numeric parts in a log message using a regex

    :param preprocessed_log: resultant log message after pre-processing
    :return: input text after replacing numeric parts with a wildcard
    """
    for i, token in enumerate(preprocessed_log):
        alpha_numeric_regex = r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$'
        is_alpha_numeric = re.search(alpha_numeric_regex, token)
        if is_alpha_numeric:
            preprocessed_log[i] = re.sub(alpha_numeric_regex, '<*>', token)

    return preprocessed_log


def check_numeric(token):
    """
    Replace numeric parts in a token  using character level check and merge

    :param token: token to be checked for numerical
    :return: input token after replacing numeric parts with a wildcard
    """
    return_token = ""
    for i in range(0, len(token)):
        if not token[i].isnumeric():
            return_token += token[i]
        else:
            return_token += '<*>'
    wildcard_check = re.compile('(?:\<\*\>)+')
    return re.sub(wildcard_check, '<*>', return_token)


def replace_nums(preprocessed_log):
    """
    Replace numeric parts in a log message by calling check_numeric on each token

    :param preprocessed_log: resultant log message after pre-processing
    :return: input text after replacing numeric parts with a wildcard
    """
    for i, token in enumerate(preprocessed_log):
        preprocessed_log[i] = check_numeric(token)
    return preprocessed_log


def replace_only_nums(preprocessed_log):
    """
    Replace nuemeric tokens with a wildcard
    :param preprocessed_log: resultant log message after pre-processing
    :return: input text after replacing numeric tokens with a wildcard
    """
    for i, token in enumerate(preprocessed_log):
        if token.isnumeric():
            preprocessed_log[i] = '<*>'

    return preprocessed_log


def get_cosine_similarity(doc_ids, temp):
    """
    Convert set of log messages into tf-idf representation and calculate cosine similarity
    :param doc_ids: list of ids of the log messages
    :param temp: templates dictionary
    :return: cosine similarity matrix
    """
    corpus = [temp[i] for i in doc_ids]
    filtered_corpus = list(map(lambda x: filter_wildcards(x), corpus))
    vectorizer = TfidfVectorizer(lowercase=False, analyzer='word', stop_words=None, tokenizer=my_tokenizer,
                                 token_pattern=None)

    vectors = vectorizer.fit_transform(filtered_corpus).toarray()
    vectors = [vectors[i].tolist() for i in range(len(corpus))]
    return cosine_similarity(vectors)


class Vue4Logs:
    """
    Parsing solution
    """

    def __init__(self, dataset, threshold=0.61):
        """
        Constructor

        :param dataset: Dataset name
        :param threshold: Selected threshold
        """
        self.threshold = threshold
        self.templates = {}
        self.inverted_index = VanillaInvertedIndex()
        self.results = []
        self.dataset = dataset
        self.output_path = "results/" + str(threshold)

    def get_new_template(self, temp_template):
        """
        Put new log message into template dictionary as a new template

        :param temp_template: log message
        :return: id of the created template
        """
        if len(self.templates.keys()) == 0:
            next_id = 0
        else:
            next_id = max(self.templates.keys()) + 1
        # print("NEXT TEMPLATE ID :", next_id)
        self.templates[next_id] = temp_template
        self.results.append(next_id)
        return next_id

    def write_results(self, input_dataframe):
        """
        Write parsed results to a file

        :return: None
        """
        input_dataframe['EventId'] = ["E" + str(i) for i in self.results]
        templates_df = []
        for j in self.results:
            if int(j) > 2000:
                print("Error in result")
                sys.exit(0)
            else:
                templates_df.append(" ".join(self.templates[j]))
        input_dataframe['EventTemplate'] = templates_df

        if not path.exists(self.output_path):
            os.makedirs(self.output_path)
        input_dataframe.to_csv(self.output_path + '/' + self.dataset + '_structured.csv')
        return input_dataframe

    def preprocess(self, line):
        """
        Preprocess log message using correspondent regex

        :param line: raw log message
        :return: preprocessed log message
        """
        regex = benchmark_settings[self.dataset]['regex']
        for currentRex in regex:
            line = re.sub(currentRex, '<*>', line)
        return line

    def get_bm25(self, doc_ids):
        """
        Get bm25 similarity of log messages
        :param doc_ids: list of ids of log messages
        :return: bm25 similarity
        """
        logs = []
        for i in doc_ids:
            if i == -1:
                query = self.templates[i]
            else:
                logs.append(self.templates[i])
        bm25 = BM25Okapi(logs)
        doc_scores = bm25.get_scores(query)
        return doc_scores

    def parse(self, input_dataframe):
        """
        Parsing algorithm

        :return: Parsing accuracy for the given data
        """
        for idx, line in input_dataframe.iterrows():
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

                    similarity = get_cosine_similarity(doc_ids, similarity_candidates)[0]
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
                        self.inverted_index.update_doc(selected_candidate_id, self.templates[selected_candidate_id],
                                                       updated_template)

                        self.templates[selected_candidate_id] = updated_template
                        self.results.append(selected_candidate_id)
                assert len(self.results) == log_id
        structured_df = self.write_results(input_dataframe)
        return structured_df
