"""
Parse the neuprintHTTP log file and export as CSV, or just print the most recent messages.

Examples:

    parse-neuprint-log /data1/neuprintlog/neuprint-cns/log.json
    parse-neuprint-log -d -u /data1/neuprintlog/neuprint-cns/log.json
    parse-neuprint-log /data1/neuprintlog/neuprint-cns/log.json recent-requests.csv
    parse-neuprint-log -t 1_000_000 /data1/neuprintlog/neuprint-cns/log.json more-requests.feather
    parse-neuprint-log -t 0 /data1/neuprintlog/neuprint-cns/log.json all-requests.feather

Notes:

    The log file contains lines like these:

    {"uri": "/api/dbmeta/datasets", "status": 200, "bytes_in": 0, "bytes_out": 92282, "duration": 41.93, "time": 1697898606, "user": "foobar@gmail.com", "category": "dbmeta/datasets", "debug": ""}
    {"uri": "/api/custom/custom", "status": 200, "bytes_in": 526, "bytes_out": 1050222, "duration": 64.31, "time": 1697898606, "user": "foobar@gmail.com", "category": "custom/custom", "debug": "
        MATCH (m:Meta)
        WITH m as m,
            apoc.convert.fromJsonMap(m.roiInfo) as roiInfo,
            apoc.convert.fromJsonMap(m.roiHierarchy) as roiHierarchy,
            apoc.convert.fromJsonMap(m.neuroglancerInfo) as neuroglancerInfo,
            apoc.convert.fromJsonList(m.neuroglancerMeta) as neuroglancerMeta,
            apoc.convert.fromJsonMap(m.statusDefinitions) as statusDefinitions
        RETURN m as meta, roiInfo, roiHierarchy, neuroglancerInfo, neuroglancerMeta, statusDefinitions
    "}

    As you can see, the log file is ALMOST '.jsonl' format, except:

    - It splits the cypher query ('debug') across multiple lines, which isn't valid JSON.
    - Special characters are inconsistently escaped in the query, so it can't be trivially
      parsed as valid JSON even if you fix the line endings.

    This script parses the file and converts it to CSV or Apache Feather
    format, with human-readable timestamps.

Warning:
    At the time of this writing, this script requires enough RAM to
    hold the entire output file in memory before it is written to disk.
    In theory, that could be improved.
"""
import os
import re
import sys
import argparse
import ujson

from tqdm import tqdm
import pandas as pd
import pyarrow.feather as feather


# Identify escape sequences that aren't valid in JSON (so we can turn them into non-escape sequences).
# Use a 'positive lookbehind assertion' to make sure we're not looking at characters following a double-\.
# For example, replace \L with \\L, but don't replace \\L with \\\L.
# Technically, even this isn't enough, since it doesn't handle \\\L correctly,
# but it seems to be good enough to parse the existing logs.
INVALID_JSON_ESCAPE = re.compile(rb'(?<=[^\\])\\([^bfnrt"\\])')


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--tail-bytes', '-t', type=int, default=10_000_000,
        help='Instead of parsing the whole file, parse only the last N bytes, according to this setting. '
             'By default, only the last 10MB is processed (10_000_000).'
             'To process the whole file, use the special value of 0.'
    )
    parser.add_argument(
        '--format', '-f', choices=['csv', 'feather', 'pretty'], required=False,
        help='Output format. The default is "pretty" if no output file was specified. '
             'Otherwise, the default is inferred from the output file extension.')
    parser.add_argument(
        '--users-last-messages', '-u', action='store_true',
        help='Show only the single most recent message from each user in the log.'
    )
    parser.add_argument(
        '--discard-cypher', '-d', action='store_true',
        help='Before exporting, discard the cypher queries in the log (the "debug" column)'
    )
    parser.add_argument(
        '--timezone', '-z', default='US/Eastern',
        help='Specify the timezone to use when exporting human-readable timestamps.'
    )
    parser.add_argument(
        '--chunk-size', '-c', type=int, default=10_000,
        help="To avoid loading millions of JSON objects into RAM, the file is converted to DataFrames in chunks. "
             "This sets the number of messages per chunk. (You can probably leave this alone.)"
    )
    parser.add_argument('log_file', help='source log file')
    parser.add_argument(
        'output_file', nargs='?',
        help='Destination of the exported file. If not given, write to stdout.'
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    log_df = parse_file(
        args.log_file,
        args.chunk_size,
        args.tail_bytes,
        args.timezone,
        args.discard_cypher,
        args.users_last_messages
    )

    if not args.format:
        if not args.output_file:
            args.format = 'pretty'
        elif args.output_file.endswith('.csv'):
            args.format = 'csv'
        else:
            args.format = 'feather'

    if not args.output_file and args.format == 'feather':
        raise RuntimeError("Feather format requries an output file.")

    # For the 'pretty' format (just print the dataframe)
    pd.set_option('display.max_rows', len(log_df))
    pd.set_option('display.max_columns', len(log_df.columns))
    pd.set_option('display.width', 1000)

    if args.format == 'pretty':
        log_df = log_df.drop(columns=['time', 'category'])

    if args.output_file:
        if args.format == 'feather':
            feather.write_feather(log_df, args.output_file)
        elif args.format == 'csv':
            log_df.to_csv(args.output_file, index=False, header=True)
        elif args.format == 'pretty':
            with open(args.output_file, 'w') as f:
                f.write(log_df)
    else:
        if args.format == 'csv':
            print(log_df.to_csv(index=False, header=True))
        elif args.format == 'pretty':
            print(log_df)


def parse_file(log_file, chunk_size, tail_bytes, timezone, discard_cypher, deduplicate_users):
    """
    Parse the entire contents of the log file.
    Return the data as pd.DataFrame.
    """
    filesize = os.path.getsize(log_file)
    progress = tqdm(total=filesize)
    with progress, open(log_file, 'rb') as f:
        if tail_bytes:
            tail_bytes = min(tail_bytes, filesize)
            f.seek(filesize - tail_bytes)
            progress.total = tail_bytes

        # Skip any incomplete message portion
        # (in case we are working with just the tail of the file).
        while f and f.peek(2)[:2] != b'{"':
            line = f.readline()
            progress.update(len(line))
            if not line:
                raise RuntimeError("File contains no complete messages.")

        chunk_dfs = []
        while chunk := parse_chunk(f, chunk_size):
            msgs, bytes_read = zip(*chunk)
            df = pd.DataFrame(msgs)
            if discard_cypher:
                del df['debug']
            chunk_dfs.append(df)
            progress.update(sum(bytes_read))

    log_df = pd.concat(chunk_dfs)

    # Convert the unix timestamp to human-readable datetime.
    t = pd.to_datetime(log_df['time'], unit='s', utc=True).dt.tz_convert(timezone)
    log_df.insert(0, 'datetime', t)

    if deduplicate_users:
        log_df = log_df.drop_duplicates('user', keep='last')

    return log_df


def parse_chunk(f, chunk_size):
    """
    Parse several messages and return a list of
    [(msg_data, num_bytes), (msg_data, num_bytes), ...]
    """
    msgs = []
    while f and len(msgs) < chunk_size:
        if (msg := parse_msg(f)):
            msgs.append(msg)
        else:
            break
    return msgs


def parse_msg(f):
    """
    Read the next message from the file, parse it as JSON,
    and return the parsed data along with the number of bytes read.
    """
    lines = []
    line = b''
    bytes_read = 0
    while line[-2:] != b'"}' and not (line[:2] == b'{"' and line[-1] == b'}'):
        line = f.readline()
        if not line:
            if lines:
                raise RuntimeError("Incomplete message")
            return None
        bytes_read += len(line)
        line = line.rstrip()
        lines.append(line)

    text = b'\\n'.join(lines)
    try:
        # Sadly, due to poor handling of special characters in the server
        # logging, the log string may contain escape sequences that aren't
        # valid when decoding as JSON, e.g. '\L'.
        # The only valid escape codes in JSON are: \b \f \n \r \t \" \\
        # So if we see some other escape code, we need to UN-escape it by
        # escaping its backslash, leaving the subsequent character out of
        # the escape sequence: \L -> \\L
        # And we fix quotes: replace \" with \\"
        data = ujson.loads(
            INVALID_JSON_ESCAPE.sub(rb'\\\\\1', text)
            .replace(rb'\\"', rb'\\\"')
        )
        return data, bytes_read
    except ujson.JSONDecodeError:
        sys.stderr.write(f"WARNING: Couldn't parse:\n{text}\n")


if __name__ == "__main__":
    main()
