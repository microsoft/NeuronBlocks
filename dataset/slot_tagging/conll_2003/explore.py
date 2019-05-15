import sys

def convert_tsv(in_file_path, out_file_path):
    words, labels = [], []
    with open(in_file_path) as f:
        fins = f.readlines()
    fout = open(out_file_path, 'w')
    for line in fins:
        if len(line) < 3:
            if len(words) > 0:
                sent_text = " ".join(words)
                sent_labels = " ".join(labels)
                fout.write(sent_text + '\t' + sent_labels + '\n')
                words, labels = [], []
        else:
            pair = line.strip('\n').split()
            words.append(pair[0])
            labels.append(pair[-1])
    if len(words) > 0:
        sent_text = " ".join(words)
        sent_labels = " ".join(labels)
        fout.write(sent_text + '\t' + sent_labels + '\n')
    fout.close()

if __name__ == '__main__':
    convert_tsv(sys.argv[1], sys.argv[2])

