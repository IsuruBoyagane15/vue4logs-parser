from flask import Flask
import pandas as pd
from flask import json
import ast
from flask_cors import CORS, cross_origin
from flask import request
from Vue4logsParser import *
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


def list_logs(logs):
    out = []
    buff = []
    for c in logs:
        if c == '\n':
            out.append(''.join(buff))
            buff = []
        else:
            buff.append(c)
    else:
        if buff:
            out.append(''.join(buff))

    return out

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



def make_summary(conf, logs):
    conf = ast.literal_eval(conf)

    logs = logs.split("\n")
    headers, regex = generate_logformat_regex(conf['log_format'])
    log_messages = []
    linecount = 0
    for line in logs:
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

    parser = Vue4Logs(conf["log_file"])
    pa = parser.parse(logdf)
    thisdict = list(pa.T.to_dict().values())

    return thisdict

def save_summary(res, fileName):
    df_final = pd.DataFrame()
    for i in list(res.values()):
        df = pd.DataFrame(i)
        df_final = df_final.append(df[['headers', 'Content', 'EventTemplate', 'Log_line']], ignore_index=True)
    df_final.to_csv('results/'+fileName+'.csv')
    return "Saves successfully"

@app.route("/", methods=['GET'])
@cross_origin()
def hello():
    response = app.response_class(
        response='Hi',
        status=200,
        mimetype='application/json'
    )
    return response


@app.route("/submit", methods=['POST'])
@cross_origin()
def parseLog():
    req = request.get_json()
    
    data = make_summary(req['conf'], req['logs'])
    for item in data:
        # item['EventTemplate'] = item['EventTemplate']
        a = item
        item['headers'] = " ".join([a[key] for key in  a.keys() - ['LineId', 'EventTemplate', 'EventId', 'Content']])
        item['Log_line'] = " ".join([a[key] for key in  a.keys() - ['LineId', 'EventTemplate', 'EventId']])


    response = app.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
    )
    return response

@app.route("/save", methods=['POST'])
@cross_origin()
def saveLog():
    req = request.get_json()
    data = save_summary(req['logs'],req['fileName'])
    response = app.response_class(
        response=json.dumps(data),
        status=200,
        mimetype='application/json'
    )
    return response

if __name__ == '__main__':
    app.run(debug=True)
