import re
import pandas as pd


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
        logs = fin.readlines()
        logs = [j.strip() for j in logs]

    for line in logs:
        try:
            line = line.strip()
            match = regex.search(line.strip())
            message = [match.group(header) for header in headers]
            log_messages.append(message)
            linecount += 1
        except Exception as e:
            print(e)
            pass
    logdf = pd.DataFrame(log_messages, columns=headers)
    logdf.insert(0, 'LineId', None)

    logdf['LineId'] = [i + 1 for i in range(linecount)]
    return logdf


def separate_forensic_headers(log_file):
    base_log_format = '\[<program_pid>\] : <Content>'
    headers, r = generate_logformat_regex(base_log_format)
    df_log = log_to_dataframe(log_file, r, headers)

    possible_headers = {
        'Level': '\<<Level>\> <Content>',
        'Duration': '\[<Duration>\] <Content>'
    }

    for i in possible_headers:
        log_format = possible_headers[i]
        headers, regex = generate_logformat_regex(log_format)
        log_messages = []
        linecount = 0
        for line in df_log['Content']:
            try:
                match = regex.search(line.strip())
                message = [match.group(header) for header in headers]
            except Exception as e:
                message = ["N/A", line]
            finally:
                linecount += 1
                log_messages.append(message)

        logdf = pd.DataFrame(log_messages, columns=headers, index=None)
        if i == "Duration":
            logdf['LineId'] = [j + 1 for j in range(linecount)]
        df_log[i] = logdf[i]
        df_log['Content'] = logdf['Content']
    return df_log

