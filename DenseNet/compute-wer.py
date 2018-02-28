import sys
import os
import numpy
def cmp_result(label,rec):
    dist_mat = numpy.zeros((len(label)+1, len(rec)+1),dtype='int32')
    dist_mat[0,:] = range(len(rec) + 1)
    dist_mat[:,0] = range(len(label) + 1)
    for i in range(1, len(label) + 1):
        for j in range(1, len(rec) + 1):
            hit_score = dist_mat[i-1, j-1] + (label[i-1] != rec[j-1])
            ins_score = dist_mat[i,j-1] + 1
            del_score = dist_mat[i-1, j] + 1
            dist_mat[i,j] = min(hit_score, ins_score, del_score)

    dist = dist_mat[len(label), len(rec)]
    return dist, len(label)

def process(recfile, labelfile, resultfile):
    total_dist = 0
    total_label = 0
    total_line = 0
    total_line_rec = 0
    rec_mat = {}
    label_mat = {}
    with open(recfile) as f_rec:
        for line in f_rec:
            tmp = line.split()
            key = tmp[0]
            latex = tmp[1:]
            rec_mat[key] = latex
    with open(labelfile) as f_label:
        for line in f_label:
            tmp = line.split()
            key = tmp[0]
            latex = tmp[1:]
            label_mat[key] = latex
    for key_rec in rec_mat:
        label = label_mat[key_rec]
        rec = rec_mat[key_rec]
        dist, llen = cmp_result(label, rec)
        total_dist += dist
        total_label += llen
        total_line += 1
        if dist == 0:
            total_line_rec += 1
    wer = float(total_dist)/total_label
    sacc = float(total_line_rec)/total_line

    f_result = open(resultfile,'w')
    f_result.write('WER {}\n'.format(wer))
    f_result.write('ExpRate {}\n'.format(sacc))
    f_result.close()

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print 'compute-wer.py recfile labelfile resultfile'
        sys.exit(0)
    process(sys.argv[1], sys.argv[2], sys.argv[3])