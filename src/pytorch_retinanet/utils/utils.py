'''misc utils'''

import io
import sys

def load_label_map(label_map_filename):
    with open(label_map_filename, 'r') as f:
        content = f.read().splitlines()
        f.close()

    assert content is not None, 'cannot find label map'

    temp = list()
    for line in content:
        line = line.strip()
        if (len(line) > 2 and
                (line.startswith('id') or
                 line.startswith('name'))):
            temp.append(line.split(':')[1].strip())

    label_dict = dict()
    for idx in range(0, len(temp), 2):
        item_id = int(temp[idx])
        item_name = temp[idx + 1][1:-1]
        label_dict[item_id] = item_name

    return label_dict

class StdoutSilencer:
    """
    Silences STDOUT. Usage is as follows:

        with StdoutSilencer():
            print('This will not print')
        print('This will print')
    """

    def __enter__(self):
        self.old_stdout = sys.stdout
        sys.stdout = io.StringIO()

    def __exit__(self, type, value, traceback):
        sys.stdout = self.old_stdout
