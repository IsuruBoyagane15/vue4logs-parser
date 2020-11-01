from inverted_index.InvertetIndex import *


def filter(preprocessed_log):
    filtered_token_list = []
    for current_token in preprocessed_log:
        if "<*>" not in current_token:
            filtered_token_list.append(current_token)

    return filtered_token_list

def filter_wildcards(preprocessed_log):
    filtered_token_list = []
    for current_token in preprocessed_log:
        if "<*>" not in current_token:
            filtered_token_list.append(current_token)

    return filtered_token_list
class BiWordInvertedIndex(InvertedIndex):
    def __init__(self):
        self.dict = {}

    def search_doc(self, query_log):
        new_sequance = []
        template_length = len(query_log)
        for i in range(template_length - 1):
            new_sequance.append(query_log[i] + " " + query_log[i + 1])
        new_sequance = filter(new_sequance)

        hits = []
        for token in new_sequance:
            if token in self.dict:
                hits += self.dict[token]
        hit_set = set(hits)
        return list(hit_set)

    def index_doc(self, doc_id, new_template):
        new_sequance = []
        template_length = len(new_template)
        for i in range(template_length-1):
            new_sequance.append(new_template[i] + " " + new_template[i+1])

        bi_sequance = filter(new_sequance)
        bi_len = len(bi_sequance)

        # print(new_template)

        for i in range(bi_len):
            token = bi_sequance[i]
            if token in self.dict:
                self.dict[token].append(doc_id)
            else:
                self.dict[token] = [doc_id]

    def update_doc(self, doc_id, old_template, updated_template):
        for token in old_template:
            if token in self.dict:
                if doc_id in self.dict[token]:
                    self.dict[token].remove(doc_id)
        self.index_doc(doc_id, updated_template)
#
# a = BiWordInvertedIndex()
# a.index_doc(0,'my name is isuru boya'.split())