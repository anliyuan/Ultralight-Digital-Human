# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang, Xiaoyu Chen, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import copy
import logging
import os
import sys

import torch
import yaml
#from torch.utils.data import DataLoader
#from wenet.dataset.dataset import Dataset
from wenet.transformer.asr_model_streaming import init_asr_model
from wenet.utils.checkpoint import load_checkpoint
from wenet.utils.file_utils import read_symbol_table
from wenet.utils.config import override_config

import torchaudio
import torchaudio.compliance.kaldi as kaldi

import time

def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--data_type',
                        default='raw',
                        choices=['raw', 'shard'],
                        help='train and cv data type')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--dict', required=True, help='dict file')
    parser.add_argument('--beam_size',
                        type=int,
                        default=10,
                        help='beam size for search')
    parser.add_argument('--penalty',
                        type=float,
                        default=0.0,
                        help='length penalty')
    parser.add_argument('--result_file', required=True, help='asr result file')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16,
                        help='asr result file')
    parser.add_argument('--mode',
                        choices=[
                            'attention', 'ctc_greedy_search',
                            'ctc_prefix_beam_search', 'attention_rescoring'
                        ],
                        default='attention',
                        help='decoding mode')
    parser.add_argument('--ctc_weight',
                        type=float,
                        default=0.0,
                        help='ctc weight for attention rescoring decode mode')
    parser.add_argument('--decoding_chunk_size',
                        type=int,
                        default=-1,
                        help='''decoding chunk size,
                                <0: for decoding, use full chunk.
                                >0: for decoding, use fixed chunk size as set.
                                0: used for training, it's prohibited here''')
    parser.add_argument('--num_decoding_left_chunks',
                        type=int,
                        default=-1,
                        help='number of left chunks for decoding')
    parser.add_argument('--simulate_streaming',
                        action='store_true',
                        help='simulate streaming inference')
    parser.add_argument('--reverse_weight',
                        type=float,
                        default=0.0,
                        help='''right to left weight for attention rescoring
                                decode mode''')
    parser.add_argument('--bpe_model',
                        default=None,
                        type=str,
                        help='bpe model for english part')
    parser.add_argument('--override_config',
                        action='append',
                        default=[],
                        help="override yaml config")

    args = parser.parse_args()
    print(args)
    return args

def feature_extraction(waveform, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0, sample_rate=16000):
    num_mel_bins=80 #23
    frame_length=25
    frame_shift=10
    dither=0.0 #0.1
    #print(feat)
    feat = kaldi.fbank(waveform, 
                      num_mel_bins=num_mel_bins,
                      frame_length=frame_length,
                      frame_shift=frame_shift,
                      dither=dither,
                      energy_floor=0.0,
                      sample_frequency=sample_rate).unsqueeze(0)#.to(device)
                      #sample_frequency=sample_rate).to(device) #unsqueeze(0).to(device)
    feat_length = torch.IntTensor([feat.size()[1]])#.to(device)
    #print('sigle wave size and  feat', feat.size(), feat_length)
    return feat, feat_length

def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    if args.mode in ['ctc_prefix_beam_search', 'attention_rescoring'
                     ] and args.batch_size > 1:
        logging.fatal(
            'decoding mode {} must be running with batch_size == 1'.format(
                args.mode))
        sys.exit(1)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    if len(args.override_config) > 0:
        configs = override_config(configs, args.override_config)

    symbol_table = read_symbol_table(args.dict)
    test_conf = copy.deepcopy(configs['dataset_conf'])

    test_conf['filter_conf']['max_length'] = 102400
    test_conf['filter_conf']['min_length'] = 0
    test_conf['filter_conf']['token_max_length'] = 102400
    test_conf['filter_conf']['token_min_length'] = 0
    test_conf['filter_conf']['max_output_input_ratio'] = 102400
    test_conf['filter_conf']['min_output_input_ratio'] = 0
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['shuffle'] = False
    test_conf['sort'] = False
    test_conf['fbank_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_size'] = args.batch_size
    
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    wave_file = '/data_4T/speech/cn-asr/aishell/data_aishell/wav/test/S0764/BAC009S0764W0136.wav'
    waveform, sample_rate = torchaudio.load(wave_file)
    waveform = waveform * (1 << 15)
    #print(len(waveform[0])/sample_rate)
    waveform_chunks = []
    for i in range(0, len(waveform), sample_rate):
        chunk = waveform[i, i+sample_rate]
        waveform_chunks = waveform_chunks.append(chunk)
    #assert 0==1

    feat, feat_length = feature_extraction(waveform, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0, sample_rate=16000)
    feat, feat_length = feat.to(device), feat_length.to(device)
    # Load dict
    char_dict = {}
    with open(args.dict, 'r') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            char_dict[int(arr[1])] = arr[0]
    eos = len(char_dict) - 1

    #load trained model
    # Init asr model from configs
    model = init_asr_model(configs)
    load_checkpoint(model, args.checkpoint)
    model = model.to(device)
    model.eval()


    with torch.no_grad(),  open(args.result_file, 'w') as fout:
        t1 = time.time()
        hyps, _ = model.recognize(
                feat,
                feat_length,
                beam_size=args.beam_size,
                decoding_chunk_size=16, #args.decoding_chunk_size,
                num_decoding_left_chunks=args.num_decoding_left_chunks,
                simulate_streaming=True)  #args.simulate_streaming)
        hyps = [hyp.tolist() for hyp in hyps[0]]
        print(hyps)
        content = ''
        for w in hyps:
            if w == eos:
                break
            content += char_dict[w]
        logging.info('{}'.format(content))
        t2 = time.time()
        print('recognition time: ',t2-t1)

if __name__ == '__main__':
    main()







'''
        for batch_idx, batch in enumerate(test_data_loader):
            #print(batch[1].size(), batch[2].size(), batch[3], batch[4])
            #torch.Size([1, 502, 80]) torch.Size([1, 38]) tensor([502], dtype=torch.int32) tensor([38], dtype=torch.int32)
            #assert 0==1
            keys, feats, target, feats_lengths, target_lengths = batch
            feats = feats.to(device)
            target = target.to(device)
            feats_lengths = feats_lengths.to(device)
            target_lengths = target_lengths.to(device)
            #print(feats)
            if args.mode == 'attention':
                hyps, _ = model.recognize(
                    feats,
                    feats_lengths,
                    beam_size=args.beam_size,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    simulate_streaming=args.simulate_streaming)
                hyps = [hyp.tolist() for hyp in hyps]
            elif args.mode == 'ctc_greedy_search':
                hyps, _ = model.ctc_greedy_search(
                    feats,
                    feats_lengths,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    simulate_streaming=args.simulate_streaming)
            # ctc_prefix_beam_search and attention_rescoring only return one
            # result in List[int], change it to List[List[int]] for compatible
            # with other batch decoding mode
            elif args.mode == 'ctc_prefix_beam_search':
                assert (feats.size(0) == 1)
                hyp, _ = model.ctc_prefix_beam_search(
                    feats,
                    feats_lengths,
                    args.beam_size,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    simulate_streaming=args.simulate_streaming)
                hyps = [hyp]
            elif args.mode == 'attention_rescoring':
                assert (feats.size(0) == 1)
                hyp, _ = model.attention_rescoring(
                    feats,
                    feats_lengths,
                    args.beam_size,
                    decoding_chunk_size=args.decoding_chunk_size,
                    num_decoding_left_chunks=args.num_decoding_left_chunks,
                    ctc_weight=args.ctc_weight,
                    simulate_streaming=args.simulate_streaming,
                    reverse_weight=args.reverse_weight)
                hyps = [hyp]
            for i, key in enumerate(keys):
                content = ''
                for w in hyps[i]:
                    if w == eos:
                        break
                    content += char_dict[w]
                logging.info('{} {}'.format(key, content))
                fout.write('{} {}\n'.format(key, content))

if __name__ == '__main__':
    main()

'''
