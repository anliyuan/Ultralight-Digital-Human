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
from wenet.utils.mask import (make_pad_mask, mask_finished_preds,
        mask_finished_scores, subsequent_mask)

from wenet.utils.common import (IGNORE_ID, add_sos_eos,
        remove_duplicates_and_blank, th_accuracy,reverse_pad_list)

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
    #print(args)
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

def encode_chunk(model, ck_feat, offset, required_cache_size, subsampling_cache, elayers_output_cache, conformer_cnn_cache, device):
    print('encoder inpout size', ck_feat.size())
    (encoder_out, subsampling_cache, elayers_output_cache,
    conformer_cnn_cache) = model.encoder.forward_chunk(
                               ck_feat, 
                               offset, 
                               required_cache_size,
                               subsampling_cache,
                               elayers_output_cache,
                               conformer_cnn_cache
                               )
    #ffset += encoder_out.size(1) 
    print('encoder output size: ',encoder_out.size()) #500ms一个chunk的话，一个chunk的size为[1, 11, 256]
    #assert 0==1
    encoder_mask = torch.ones(1, encoder_out.size(1), device=encoder_out.device, dtype=torch.bool)
    return encoder_out, encoder_mask, subsampling_cache, elayers_output_cache, conformer_cnn_cache



def decode_chunk(model, encoder_out, encoder_mask, beam_size, cache, device, hyps, scores, end_flag):
    batch_size = encoder_out.size(0)
    chunk_length = encoder_out.size(1)
    encoder_dim = encoder_out.size(2)
    running_size = batch_size * beam_size
    encoder_out = encoder_out.unsqueeze(1).repeat(1, beam_size, 1, 1).view(
                    running_size, chunk_length, encoder_dim)  # (B*N, maxlen, encoder_dim)
    encoder_mask = encoder_mask.unsqueeze(1).repeat(
                    1, beam_size, 1, 1).view(running_size, 1, chunk_length)  # (B*N, 1, max_len)
    print('wave encoder ou tsize :', encoder_out.shape)

    for i in range(1, chunk_length + 1):  # 64个framedecoder后的长度为11
        if end_flag.sum() == running_size:
            break
        hyps_mask = subsequent_mask(i).unsqueeze(0).repeat(running_size, 1, 1).to(device)  # (B*N, i, i)
        logp, cache = model.decoder.forward_one_step(encoder_out, encoder_mask, hyps, hyps_mask, cache)
        #print(logp.shape)
        #print('***cache', len(cache), cache[0].shape)   #6
        #assert 0==1
        #print(logp.size()) #[10, 11008]
        #'''
        # 2.2 First beam prune: select topk best prob at current time
        top_k_logp, top_k_index = logp.topk(beam_size)  # (B*N, N)
        top_k_liogp = mask_finished_scores(top_k_logp, end_flag)
        top_k_index = mask_finished_preds(top_k_index, end_flag, model.eos)
        #print(top_k_logp)
        #assert 0==1

        # 2.3 Second beam prune: select topk score with history
        scores = scores + top_k_logp  # (B*N, N), broadcast add
        scores = scores.view(batch_size, beam_size * beam_size)  # (B, N*N)
        scores, offset_k_index = scores.topk(k=beam_size)  # (B, N)
        scores = scores.view(-1, 1)  # (B*N, 1)
        # 2.4. Compute base index in top_k_index,
        # regard top_k_index as (B*N*N),regard offset_k_index as (B*N),
        # then find offset_k_index in top_k_index
        base_k_index = torch.arange(batch_size, device=device).view(
                -1, 1).repeat([1, beam_size])  # (B, N)
        base_k_index = base_k_index * beam_size * beam_size
        best_k_index = base_k_index.view(-1) + offset_k_index.view(-1) #(B*N)
        # 2.5 Update best hyps
        best_k_pred = torch.index_select(top_k_index.view(-1),dim=-1,index=best_k_index)  # (B*N)
        best_hyps_index = best_k_index // beam_size
        last_best_k_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)  # (B*N, i)
        hyps = torch.cat((last_best_k_hyps, best_k_pred.view(-1, 1)),dim=1)  # (B*N, i+1)
        print(hyps)
        # 2.6 Update end flag 
        end_flag = torch.eq(hyps[:, -1], model.eos).view(-1, 1)
        #'''
    '''
    # 3. Select best of best
    scores = scores.view(batch_size, beam_size)
    best_scores, best_index = scores.max(dim=-1)
    best_hyps_index = best_index + torch.arange(
            batch_size, dtype=torch.long, device=device) * beam_size
    best_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)
    best_hyps = best_hyps[:, 1:]
    print(best_hyps, best_scores)
    #assert 0==1
    '''
    #return best_hyps, best_scores, new_cache
    return cache, hyps
            

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
    #device = torch.device('cpu') #
    device = torch.device('cuda' if use_cuda else 'cpu')

    #wave_file = '/data_4T/speech/cn-asr/aishell/data_aishell/wav/test/S0764/BAC009S0764W0136.wav'
    wave_file = '/data_4T/speech/cn-asr/tel_conversation/seg/wav-16k/A7222_3_242.wav'
    waveform, sample_rate = torchaudio.load(wave_file)
    waveform = waveform * (1 << 15)
    #print(waveform.shape)
    print('yes '*20)
    #print(len(waveform[0])/sample_rate)
    waveform_chunks = []
    ck_len = int(sample_rate * 0.5)
    for i in range(0, len(waveform[0]), ck_len):  # 500ms  equals a  chunk
        end = min(i+ck_len, len(waveform[0]))
        #chunk = waveform[0,i:i+sample_rate//2]
        chunk = waveform[0,i:end]
        if len(chunk)>1000:
            waveform_chunks.append(chunk.unsqueeze(0))
        #print(chunk.size())
    print('nums of chunks: ', len(waveform_chunks))
    #assert 0==1

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

    subsampling_cache: Optional[torch.Tensor] = None
    elayers_output_cache: Optional[List[torch.Tensor]] = None
    conformer_cnn_cache: Optional[List[torch.Tensor]] = None
    cache: Optional[List[torch.Tensor]] = None
    required_cache_size = -11 #-16
    outputs = []
    beam_size = args.beam_size
    offset = 0 
    hyps = torch.ones([beam_size, 1], dtype=torch.long,
                              device=device).fill_(model.sos)  # (B*N, 1)
    scores = torch.tensor([0.0] + [-float('inf')] * (beam_size - 1),
                                  dtype=torch.float)
    #scores = scores.to(device).repeat([batch_size]).unsqueeze(1).to(device)
    scores = scores.to(device).repeat([1]).unsqueeze(1).to(device)
    end_flag = torch.zeros_like(scores, dtype=torch.bool, device=device)

    with torch.no_grad(),  open(args.result_file, 'w') as fout:
        t1 = time.time()
        hyps_allchunks = []
        #'''
        for ck in waveform_chunks:     
            ck_feat, ck_feat_length = feature_extraction(ck, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0, sample_rate=16000)
            ck_feat, ck_feat_length = ck_feat.to(device), ck_feat_length.to(device)
            #print(ck_feat_length)
            #print(ck_feat.shape)
            encoder_out, encoder_mask, subsampling_cache, elayers_output_cache, conformer_cnn_cache = encode_chunk(model, 
                                                                ck_feat, offset, required_cache_size, 
                                                                subsampling_cache, elayers_output_cache, conformer_cnn_cache, device)
            #print('wave encoder ou tsize :', encoder_out.shape)
            #assert 0==1
            offset += encoder_out.size(1) 
            #assert 0==1
            #best_hyps, best_scores , cache= decode_chunk(model, encoder_out, encoder_mask, beam_size, cache, device)
            #cache: Optional[List[torch.Tensor]] = None
            #cache, hyps = decode_chunk(model, encoder_out, encoder_mask, beam_size, cache, device, hyps, scores, end_flag)
            #scores, hyps = greed_search_decode(model, encoder_out, encoder_mask)
            chunk_length = encoder_out.size(1)
            batch_size = encoder_out.size(0)
            encoder_out_lens = encoder_mask.squeeze(1).sum(1)
            ctc_probs = model.ctc.log_softmax(encoder_out)  # (B, maxlen, vocab_size)
            topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
            topk_index = topk_index.view(batch_size, chunk_length)  # (B, maxlen)
            mask = make_pad_mask(encoder_out_lens)  # (B, maxlen)
            topk_index = topk_index.masked_fill_(mask, model.eos)  # (B, maxlen)
            hyps = [hyp.tolist() for hyp in topk_index]
            scores = topk_prob.max(1)
            hyps = [remove_duplicates_and_blank(hyp) for hyp in hyps]
            #print(scores, hyps)
            hyps_allchunks += hyps[0]
            #assert 0==1
        #print(hyps_allchunks)
        #assert 0==1
        content = ''
        for w in hyps_allchunks:
            if w == eos:
                break
            content += char_dict[w]
        logging.info('{}'.format(content))
        
        
        #assert 0==1
        
        '''

        feat, feat_length = feature_extraction(waveform, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0, sample_rate=16000)
        feat, feat_length = feat.to(device), feat_length.to(device)
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
        #'''


def greed_search_decode(model, encoder_out, encoder_mask): #, device):
    chunk_length = encoder_out.size(1)
    batch_size = encoder_out.size(0)
    encoder_out_lens = encoder_mask.squeeze(1).sum(1)
    ctc_probs = model.ctc.log_softmax(encoder_out)  # (B, maxlen, vocab_size)
    topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, maxlen, 1)
    topk_index = topk_index.view(batch_size, chunk_length)  # (B, maxlen)
    mask = make_pad_mask(encoder_out_lens)  # (B, maxlen)
    topk_index = topk_index.masked_fill_(mask, model.eos)  # (B, maxlen)
    hyps = [hyp.tolist() for hyp in topk_index]
    scores = topk_prob.max(1)
    hyps = [remove_duplicates_and_blank(hyp) for hyp in hyps]
    return scores, hyps


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
