import numpy
import os
import sys


from nmt import train

def main(job_id, params):
    print params
    validerr = train(saveto=params['model'][0],
                     bn_saveto=params['bn_model'][0],
                     reload_=params['reload'][0],
                     dim_word=params['dim_word'][0],
                     kernel_Convenc=params['kernel_Convenc'],
                     dim_ConvBlock=params['dim_ConvBlock'],
                     layersNum_block=params['layersNum_block'],
                     dim_dec=params['dim_dec'][0], 
                     dim_attention=params['dim_attention'][0],
                     dim_coverage=params['dim_coverage'][0],
                     kernel_coverage=params['kernel_coverage'],
                     dim_target=params['dim_target'][0],
                     input_channels=params['input_channels'][0],
                     decay_c=params['decay-c'][0],
                     clip_c=params['clip-c'][0],
                     lrate=params['learning-rate'][0],
                     optimizer=params['optimizer'][0], 
                     patience=15,
                     maxlen=params['maxlen'][0],
                     maxImagesize=params['maxImagesize'][0],
                     batch_Imagesize=500000,
                     valid_batch_Imagesize=500000,
                     batch_size=8,
                     valid_batch_size=8,
                     validFreq=-1,
                     dispFreq=100,
                     saveFreq=-1,
                     sampleFreq=-1,
          datasets=['../data/offline-train.pkl',
                    '../data/train_caption.txt'],
          valid_datasets=['../data/offline-test.pkl',
                    '../data/test_caption.txt'],
          dictionaries=['../data/dictionary.txt'],
          valid_output=['./result/valid_decode_result.txt'],
          valid_result=['./result/valid.wer'],
                         use_dropout=params['use-dropout'][0])
    return validerr

if __name__ == '__main__':
    
    modelDir=sys.argv[1]
    maxlen=[200]
    maxImagesize=[500000]
    dim_word=[256]
    dim_dec=[256]
    dim_attention=[128]
    dim_coverage=[128]
    kernel_coverage=[5,5]
    kernel_Convenc=[3,3]
    dim_ConvBlock=[32,64,64,128]
    layersNum_block=[4,4,4,4]
    

        
    main(0, {
        'model': [modelDir+'attention_maxlen'+str(maxlen)+'_dimWord'+str(dim_word[0])+'_dim'+str(dim_dec[0])+'.npz'],
        'bn_model': [modelDir+'bn_params.npz'],
        'dim_word': dim_word,
        'dim_dec': dim_dec,
        'dim_attention': dim_attention,
        'dim_coverage': dim_coverage,
        'kernel_coverage': kernel_coverage,
        'kernel_Convenc': kernel_Convenc,
        'dim_ConvBlock': dim_ConvBlock,
        'layersNum_block': layersNum_block,
        'dim_target': [111], 
        'input_channels': [1], 
        'optimizer': ['adam'],
        'decay-c': [1e-4], 
        'clip-c': [100.], 
        'use-dropout': [True],
        'learning-rate': [2e-4],
        'maxlen': maxlen,
        'maxImagesize': maxImagesize,
        'reload': [False]})


