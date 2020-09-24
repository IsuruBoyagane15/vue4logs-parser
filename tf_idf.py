import math
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
# from rank_bm25 import BM25Okapi

pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import TfidfVectorizer


def filter_from_wildcards(processed_log):
    filtered_token_list = []
    for current_token in processed_log:
        if "<*>" not in current_token:
            filtered_token_list.append(current_token)

    return filtered_token_list


# def tf_idf(templates, log_message):
#     print(template)
#     print(log_message)
#     print('---')
#     bagOfWordsA = filter_from_wildcards(template).split(" ")
#     bagOfWordsB = filter_from_wildcards(log_message).split(" ")
#     print(bagOfWordsA)
#     print(bagOfWordsB)
#     print('---')
#     print("bow-count A", len(bagOfWordsA))
#     print("bow-count B", len(bagOfWordsB))
#     print('---')
#
#     uniqueWords = set(bagOfWordsA).union(set(bagOfWordsB))
#     print('#', len(uniqueWords))
#     numOfWordsA = dict.fromkeys(uniqueWords, 0)
#
#     numOfWordsB = dict.fromkeys(uniqueWords, 0)
#
#     for word in bagOfWordsA:
#         numOfWordsA[word] += 1
#
#     for word in bagOfWordsB:
#         numOfWordsB[word] += 1
#     print('---')
#     print(pd.DataFrame([numOfWordsA, numOfWordsB]))
#
#     tfFirst = computeTF(numOfWordsA, bagOfWordsA)
#     tfSecond = computeTF(numOfWordsB, bagOfWordsB)
#     tf = pd.DataFrame([tfFirst, tfSecond])
#     print(tf)
#
#     idfs = computeIDF([numOfWordsA, numOfWordsB])
#     print('====')
#     tfidfA = computeTFIDF(tfFirst, idfs)
#     tfidfB = computeTFIDF(tfSecond, idfs)
#     df = pd.DataFrame([tfidfA, tfidfB])
#     print(df)


# def computeTF(wordDict, bagOfWords):
#     tfDict = {}
#     bagOfWordsCount = len(bagOfWords)
#     for word, count in wordDict.items():
#         tfDict[word] = count / float(bagOfWordsCount)
#     return (tfDict)


# def computeIDF(documents):
#     N = len(documents)
#
#     idfDict = dict.fromkeys(documents[0].keys(), 0)
#     for document in documents:
#         for word, val in document.items():
#             if val > 0:
#                 idfDict[word] += 1
#
#     for word, val in idfDict.items():
#         idfDict[word] = math.log(N / float(val) + 1)
#     return idfDict


# def get_tfidf(doc_ids, temp):
#     TEMPLATES = temp
#     docs = []
#     for doc_id in doc_ids:
#         doc = TEMPLATES[doc_id]
#         docs.append(doc)
#         # print(doc)
#
#     temp_dict = {}
#
#     doc_bows = []
#     for i, doc_id in enumerate(doc_ids):
#         doc = TEMPLATES[doc_id]
#         doc_bow = filter_from_wildcards(doc)
#         doc_bows.append(doc_bow)
#         temp_dict[doc_id] = [doc_bow]
#
#     words = []
#     for j in doc_bows:
#         words += j
#     # print('words', len(words))
#     unique_words = set(words)
#     # print('unique words', len(unique_words))
#
#     for id in temp_dict:
#         numOfWords_doc = dict.fromkeys(unique_words, 0)
#         temp_dict[id].append(numOfWords_doc)
#
#     Tfs = []
#     for i in (temp_dict):
#         bow = temp_dict[i][0]
#         for word in bow:
#             temp_dict[i][1][word] += 1
#         Tfs.append(computeTF(temp_dict[i][1], bow))
#
#     # print(pd.DataFrame(Tfs))
#
#     word_counts = [temp_dict[j][1] for j in temp_dict]
#     idfs = computeIDF(word_counts)
#     tfIdfs = []
#
#     for tf in Tfs:
#         tfIdfs.append(computeTFIDF(tf, idfs))
#     tfidf = pd.DataFrame(tfIdfs)
#     c = cosine_similarity(tfidf)
#     # print(c)
#     # print("ccc", c.tolist())
#     return c.tolist()
#     # max_similarity = 0
#     # selected_candidate_id = None
#     # for t in temp_dict:
#     #     if t != -1:
#     #         current_similarity = temp_dict[t][2]
#     #         if current_similarity > max_similarity:
#     #             max_similarity = current_similarity
#     #             selected_candidate_id = t
#     # return selected_candidate_id, max_similarity

# def get_tfidf(docs):
#     filtered_corpus = list(map(lambda x: filter_from_wildcards(x), docs))
#     vectorizer = TfidfVectorizer(lowercase=False, analyzer='word', stop_words=None, tokenizer=my_tokenizer,
#                                  token_pattern=None)
#
#     vectors = vectorizer.fit_transform(filtered_corpus).toarray()
#     print("p", vectors)
#     vectors = [vectors[i].tolist() for i in range(len(docs))]
#     return vectors

# def computeTFIDF(tfBagOfWords, idfs):
#     tfidf = {}
#     for word, val in tfBagOfWords.items():
#         tfidf[word] = val * idfs[word]
#     return tfidf


def my_tokenizer(text):
    return text

def get_tfidf(doc_ids, temp):
    corpus = [temp[i] for i in doc_ids]
    filtered_corpus = list(map(lambda x: filter_from_wildcards(x), corpus))
    # print(corpus, end='')
    vectorizer = TfidfVectorizer(lowercase=False, analyzer='word', stop_words=None, tokenizer=my_tokenizer,
                                 token_pattern=None)
    # bm25 = BM25Okapi(filtered_corpus)
    # doc_scores = bm25.get_scores(filtered_corpus[0])
    # print(doc_scores)
    # return doc_scores


    vectors = vectorizer.fit_transform(filtered_corpus).toarray()
    vectors = [vectors[i].tolist() for i in range(len(corpus))]
    return cosine_similarity(vectors)

if __name__ == '__main__':
    # TEMPLATES = {
    #     1: 'Closed socket connection for client <*> (no session established for client)',
    #     2: 'Exception causing close of session 0x0 due to java.io.IOException: ZooKeeperServer not running',
    #     3: 'Interrupted while waiting for message on queue',
    #     4: 'Connection broken for id <*>, my id = <*>, error =',
    #     5: 'Closed socket connection for client <*> which had sessionid 0x14ed93111f<*>',
    #     6: 'Got user-level KeeperException when processing sessionid:0x34ed93485090001 type:create cxid:0x55b8bb0f zxid:0x100000010 txntype:<*> reqpath:n/a Error Path:/home/curi/.zookeeper Error:KeeperErrorCode = NodeExists for /home/curi/.zookeeper',
    #     7: 'Client attempting to renew session <*> at <*>',
    #     8: 'Client attempting to establish new session at <*>',
    #     9: 'Established session <*> with negotiated timeout <*> for client <*>',
    #     10: 'Processed session termination for sessionid: 0x14ede63a5a<*>',
    #     11: 'Expiring session <*> timeout of <*> exceeded',
    #     12: 'Established session 0x34edfaa9c22003a with negotiated timeout <*> for client <*>'
    # }/

    TEMPLATES = {
        1: ['Closed', 'socket', 'connection', 'for', 'client', '<*>', '(no', 'session', 'established', 'for',
            'client)'],
        2: ['Exception', 'causing', 'close', 'of', 'session', '0x0', 'due', 'to', 'java.io.IOException:',
            'ZooKeeperServer', 'not', 'running'],
        3: ['Interrupted', 'while', 'waiting', 'for', 'message', 'on', 'queue'],
        4: ['Connection', 'broken', 'for', 'id', '<*>,', 'my', 'id', '=', '<*>,', 'error', '='],
        5: ['Closed', 'socket', 'connection', 'for', 'client', '<*>', 'which', 'had', 'sessionid', '0x14ed93111f<*>'],
        6: ['Got', 'user-level', 'KeeperException', 'when', 'processing', 'sessionid:0x34ed93485090001', 'type:create',
            'cxid:0x55b8bb0f', 'zxid:0x100000010', 'txntype:<*>', 'reqpath:n/a', 'Error', 'Path:/home/curi/.zookeeper',
            'Error:KeeperErrorCode', '=', 'NodeExists', 'for', '/home/curi/.zookeeper'],
        7: ['Client', 'attempting', 'to', 'renew', 'session', '<*>', 'at', '<*>'],
        8: ['Client', 'attempting', 'to', 'establish', 'new', 'session', 'at', '<*>'],
        9: ['Established', 'session', '<*>', 'with', 'negotiated', 'timeout', '<*>', 'for', 'client', '<*>'],
        10: ['Processed', 'session', 'termination', 'for', 'sessionid:', '0x14ede63a5a<*>'],
        11: ['Expiring', 'session', '<*>', 'timeout', 'of', '<*>', 'exceeded'],
        12: ['Established', 'session', '0x34edfaa9c22003a', 'with', 'negotiated', 'timeout', '<*>', 'for', 'client',
             '<*>']
    }

    lista = [12, 7, 9]

    # similarities = get_tfidf(lista, TEMPLATES)
    # print(pd.DataFrame(similarities))
    # print("++++++++")

    corpus = [TEMPLATES[i] for i in lista]
    filtered_corpus = list(map(lambda x: filter_from_wildcards(x), corpus))
    # vectorizer = TfidfVectorizer(lowercase=False, analyzer='word', stop_words=None, tokenizer=my_tokenizer,
    #                              token_pattern=None)
    #
    # vectors = vectorizer.fit_transform(filtered_corpus).toarray()
    # vectors = [vectors[i].tolist() for i in range(len(corpus))]
    # print(cosine_similarity(vectors))

    # for i in TEMPLATES:
    #     print(i, ":", TEMPLATES[i].split() , ',')


    # print(list(get_tfidf(lista, TEMPLATES)))

    bm25 = BM25Okapi(filtered_corpus[1:])
    doc_scores = bm25.get_scores(filtered_corpus[0])
    print(doc_scores)