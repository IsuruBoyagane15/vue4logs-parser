from inverted_index.InvertetIndex import *


def filter(preprocessed_log):
    filtered_token_list = []
    for current_token in preprocessed_log:
        if "<*>" not in current_token:
            filtered_token_list.append(current_token)

    return filtered_token_list


class VanillaInvertedIndex(InvertedIndex):
    def __init__(self):
        self.dict = {}

    def search_doc(self, query_log):
        query_log = filter(query_log)

        hits = []
        for token in query_log:
            if token in self.dict:
                hits += self.dict[token]
        hit_set = set(hits)
        return list(hit_set)

    def index_doc(self, doc_id, new_template):
        new_template = filter(new_template)

        template_length = len(new_template)
        # print(new_template)

        for i in range(template_length):
            token = new_template[i]
            if token in self.dict:
                self.dict[token].append(doc_id)
            else:
                self.dict[token] = [doc_id]

    def update_doc(self, tokens_to_remove, doc_id):
        for token in tokens_to_remove:
            if token in self.dict:
                if doc_id in self.dict[token]:
                    self.dict[token].remove(doc_id)
