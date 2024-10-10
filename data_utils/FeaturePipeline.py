import numpy as np
import yaml
import torch
import torchaudio.compliance.kaldi as kaldi

class Feature_Pipeline():

    def __init__(self, engine_config):
        #with open(model_config_path, 'r') as fin:
        #    self.configs = yaml.load(fin, Loader=yaml.FullLoader)
        self.configs = engine_config
        self.num_mel_bins = self.configs['data_conf']['fbank_conf']['num_mel_bins'] # 80
        self.frame_length = self.configs['data_conf']['fbank_conf']['frame_length'] # 25
        # self.frame_shift = 1
        self.frame_shift = self.configs['data_conf']['fbank_conf']['frame_shift']   # 10
        self.dither = self.configs['data_conf']['fbank_conf']['dither'] # 0.0
        self.sample_rate = self.configs['engine_sample_rate_hertz'] # 16000
        #self.feature_queue_ = torch.tensor([[0]])
        self.first_wav_ = b''
        self.remained_wav_ = b''
        self._waveform = b'' #torch.tensor([[0]])
        self.exist_endpoint = False

    def AcceptWaveform(self, audio):  # audio: b''
        first, self.remained_wav_, ExitEndpoint = self.vad.endpoint_detect(audio)
        self._waveform += first
        #if ExitEndpoint:
        #self._waveform = self._waveform + self.first_wav_
        #else:
        #    self._waveform
        #self.remained_wav_ = second_wav
        '''
        feat, feat_length = self._extact_feature(waveform)
        if self.feature_queue_.shape[1]==1:
            self.feature_queue_ = feat
        else:
            self.feature_queue_ = torch.cat((self.feature_queue_, feat), 1)
        '''
        self.exist_endpoint = ExitEndpoint
        #self.mutex.release()
        #print('待计算音频长度：', len(self._waveform))
        return ExitEndpoint #, self.feature_queue_.shape[1]

    def _extract_feature(self, waveform_int16):
        #print(waveform_int16.shape)
        #assert max(waveform_int16.shape) > 512
        feat = kaldi.fbank(waveform_int16,
                          num_mel_bins=self.num_mel_bins,
                          frame_length=self.frame_length,
                          frame_shift=self.frame_shift,
                          dither=self.dither, 
                          energy_floor=0.0,
                          sample_frequency=self.sample_rate
                          )
        feat = feat.unsqueeze(0) #.to(device)
        feat_length = torch.IntTensor([feat.size()[1]])
        return feat, feat_length


    def Reset(self):
        self.remained_wav_ = b''
        self._waveform = b''
        #self.feature_queue_ = torch.tensor([[0]])

    def get_waveform_len(self):
        return len(self._waveform)

    def ReadFeats(self):
        if len(self._waveform) < 512:
            return None
        waveform = np.frombuffer(self._waveform, dtype=np.int16)
        waveform = torch.from_numpy(waveform).float().unsqueeze(0)
        feat, feat_length = self._extract_feature(waveform)
        #self._waveform = b''
        if self.exist_endpoint:
            self._waveform = self.remained_wav_
        else: self._waveform = b''
        return feat #, self.exist_endpoint

