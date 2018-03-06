import numpy

import cPickle as pkl
import gzip


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)

def dataIterator(feature_file,label_file,dictionary,batch_size,batch_Imagesize,maxlen,maxImagesize):
    
    fp=open(feature_file,'rb')
    features=pkl.load(fp)
    fp.close()

    fp2=open(label_file,'r')
    labels=fp2.readlines()
    fp2.close()

    targets={}
    # map word to int with dictionary
    for l in labels:
        tmp=l.strip().split()
        uid=tmp[0]
        w_list=[]
        for w in tmp[1:]:
            if dictionary.has_key(w):
                w_list.append(dictionary[w])
            else:
                print 'a word not in the dictionary !! sentence ',uid,'word ', w
                sys.exit()
        targets[uid]=w_list



    imageSize={}
    for uid,fea in features.iteritems():
        imageSize[uid]=fea.shape[1]*fea.shape[2]

    imageSize= sorted(imageSize.iteritems(), key=lambda d:d[1]) # sorted by sentence length,  return a list with each triple element


    feature_batch=[]
    label_batch=[]
    feature_total=[]
    label_total=[]
    uidList=[]

    batch_image_size=0
    biggest_image_size=0
    i=0
    for uid,size in imageSize:
        if size>biggest_image_size:
            biggest_image_size=size
        fea=features[uid]
        lab=targets[uid]
        batch_image_size=biggest_image_size*(i+1)
        if len(lab)>maxlen:
            print 'sentence', uid, 'length bigger than', maxlen, 'ignore'
        elif size>maxImagesize:
            print 'image', uid, 'size bigger than', maxImagesize, 'ignore'
        else:
            uidList.append(uid)
            if batch_image_size>batch_Imagesize or i==batch_size: # a batch is full
                feature_total.append(feature_batch)
                label_total.append(label_batch)

                i=0
                biggest_image_size=size
                feature_batch=[]
                label_batch=[]
                feature_batch.append(fea)
                label_batch.append(lab)
                batch_image_size=biggest_image_size*(i+1)
                i+=1
            else:
                feature_batch.append(fea)
                label_batch.append(lab)
                i+=1

    # last batch
    feature_total.append(feature_batch)
    label_total.append(label_batch)

    print 'total ',len(feature_total), 'batch data loaded'

    return zip(feature_total,label_total),uidList