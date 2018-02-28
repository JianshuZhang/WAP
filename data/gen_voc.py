import sys
import os
def gen_voc(infile, vocfile):
    vocab=set()
    with open(infile) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                print 'illegal line: ', line
                continue
            (title,label) = parts
            for w in label.split():
                if w not in vocab:
                    vocab.add(w)
    with open(vocfile,'w') as fout:
        for i, w in enumerate(vocab):
            fout.write('{}\t{}\n'.format(w,i+1))
        fout.write('<eol>\t0\n')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'gen_voc infile outfile'
        sys.exit(0)
    gen_voc(sys.argv[1], sys.argv[2])