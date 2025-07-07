import time
from typing import Tuple

import numpy as np
import openvino as ov
import torch
from funasr.models.ct_transformer.model import CTTransformer
from funasr.models.seaco_paraformer.model import SeacoParaformer
from funasr.models.transformer.utils.nets_utils import make_pad_mask, pad_list


class NewSeacoParaformer(SeacoParaformer):

    def __init__(self, base_object):
        self.__dict__ = base_object.__dict__


        core = ov.Core()


        print("ov loading encoder")
        encoder = core.read_model("openvino/encoder.xml")
        # self.ov_encoder = core.compile_model(encoder)
        self.ov_encoder = core.compile_model(model=encoder, device_name="GPU", config={"INFERENCE_PRECISION_HINT": "f32"})

        print("ov loading predictor")
        predictor_model = core.read_model("openvino/predictor.xml")
        self.ov_predictor = core.compile_model(predictor_model, device_name="GPU", config={"INFERENCE_PRECISION_HINT": "f32"})

        print("ov loading decoder1")
        decoder1_model = core.read_model("openvino/decoder_create_mask_first.xml")
        # self.ov_decoder1 = core.compile_model(decoder1_model)
        self.ov_decoder1 = core.compile_model(decoder1_model, device_name="GPU", config={"INFERENCE_PRECISION_HINT": "f32"})

        print("ov loading decoder2")
        decoder2_model = core.read_model("openvino/decoder_create_mask_second.xml")
        self.ov_decoder2 = core.compile_model(decoder2_model, device_name="GPU", config={"INFERENCE_PRECISION_HINT": "f32"})

        print("ov loading decoder3")
        decoder3_model = core.read_model("openvino/decoder_create_mask_third.xml")
        self.ov_decoder3 = core.compile_model(decoder3_model, device_name="GPU", config={"INFERENCE_PRECISION_HINT": "f32"})

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encoder. Note that this method is used by asr_inference.py
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                ind: int
        """
        print("ov encoder")
        # Forward encoder
        speech = torch.cat((speech, torch.zeros((speech.shape[0], 300-speech_lengths, speech.shape[2])).to(speech.device)), dim=1)
        enc_in_1 = speech.cpu()
        enc_in_2 = torch.Tensor([speech_lengths]).to(torch.int32).to("cpu")
        for i in range(10):
            start_t = time.time()
            ov_out = self.ov_encoder((enc_in_1, enc_in_2))
            end_t = time.time()
            elapsed = end_t - start_t
            if i!=0:
                print(f"encoder: {elapsed*1000} ms")

        encoder_out = torch.Tensor(ov_out[0]).to(speech.device)
        encoder_out_lens = speech_lengths

        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        return encoder_out, encoder_out_lens


    def calc_predictor(self, encoder_out, encoder_out_lens):
        encoder_out_mask = (~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]).to(encoder_out.device)

        print("ov predictor")

        pred_in_1 = encoder_out.cpu().numpy()
        pred_in_2 = encoder_out_mask.cpu().numpy()
        for i in range(10):
            start_t = time.time()
            ov_out = self.ov_predictor((pred_in_1, pred_in_2))
            end_t = time.time()
            elapsed = end_t - start_t
            if i!=0:
                print(f"predictor: {elapsed*1000} ms")

        ov_out = list(ov_out.values())
        l_test = np.zeros((1, 100, 512))
        keep_values = ov_out[3] >= 1
        l_test[:, :np.sum(keep_values), :] = ov_out[0][:, keep_values, :]
        ov_out[0] = l_test
        ov_out.append(torch.Tensor(np.nonzero(keep_values)[0]).to("cpu"))

        for i in range(len(ov_out)):
            ov_out[i] = torch.tensor(ov_out[i]).to(torch.float32).to(encoder_out.device)

        return ov_out


    def _seaco_decode_with_ASF(self,
                                encoder_out,
                                encoder_out_lens,
                                sematic_embeds,
                                ys_pad_lens,
                                hw_list,
                                nfilter=50,
                                seaco_weight=1.0):

        print("ov decoder1")

        dec_in_1 = encoder_out.cpu()
        dec_in_2 = torch.Tensor([encoder_out_lens]).to(torch.int32).cpu()
        dec_in_3 = sematic_embeds.cpu()
        dec_in_4 = torch.Tensor([ys_pad_lens]).to(torch.int32).cpu()
        for i in range(10):
            start_t = time.time()
            ov_out = self.ov_decoder1((dec_in_1, dec_in_2, dec_in_3, dec_in_4))
            end_t = time.time()
            elapsed = end_t - start_t
            if i!=0:
                print(f"dec 1: {elapsed*1000} ms")


        decoder_out = torch.Tensor(ov_out[0]).to(encoder_out.device)
        decoder_hidden = torch.Tensor(ov_out[1]).to(encoder_out.device)

        decoder_pred = torch.log_softmax(decoder_out, dim=-1)
        if hw_list is not None:
            hw_lengths = [len(i) for i in hw_list]
            hw_list_ = [torch.Tensor(i).long() for i in hw_list]
            hw_list_pad = pad_list(hw_list_, 0).to(encoder_out.device)
            selected = self._hotword_representation(hw_list_pad, torch.Tensor(hw_lengths).int().to(encoder_out.device))

            contextual_info = selected.squeeze(0).repeat(encoder_out.shape[0], 1, 1).to(encoder_out.device)
            num_hot_word = contextual_info.shape[1]
            _contextual_length = torch.Tensor([num_hot_word]).int().repeat(encoder_out.shape[0]).to(encoder_out.device)

            # ASF Core
            if nfilter > 0 and nfilter < num_hot_word:
                hotword_scores = self.seaco_decoder.forward_asf6(contextual_info, _contextual_length, decoder_hidden, ys_pad_lens)
                hotword_scores = hotword_scores[0].sum(0).sum(0)
                # hotword_scores /= torch.sqrt(torch.tensor(hw_lengths)[:-1].float()).to(hotword_scores.device)
                dec_filter = torch.topk(hotword_scores, min(nfilter, num_hot_word-1))[1].tolist()
                add_filter = dec_filter
                add_filter.append(len(hw_list_pad)-1)
                # filter hotword embedding
                selected = selected[add_filter]
                # again
                contextual_info = selected.squeeze(0).repeat(encoder_out.shape[0], 1, 1).to(encoder_out.device)
                num_hot_word = contextual_info.shape[1]
                _contextual_length = torch.Tensor([num_hot_word]).int().repeat(encoder_out.shape[0]).to(encoder_out.device)

            print(f"{contextual_info.shape = }")
            print(f"{_contextual_length = }")
            print(f"{sematic_embeds.shape = }")
            print(f"{decoder_hidden.shape = }")

            # SeACo Core
            print("ov decoder2")

            dec_in_1 = contextual_info.cpu()
            dec_in_2 = _contextual_length.to("cpu")
            dec_in_3 = sematic_embeds.cpu()
            dec_in_4 = torch.Tensor([ys_pad_lens]).to(torch.int32).to("cpu")
            for i in range(10):
                start_t = time.time()
                ov_out2 = self.ov_decoder2((dec_in_1, dec_in_2, dec_in_3, dec_in_4))
                end_t = time.time()
                elapsed = end_t - start_t
                if i!=0:
                    print(f"dec 2: {elapsed*1000} ms")
            cif_attended = torch.Tensor(ov_out2[0]).to(encoder_out.device)

            print("ov decoder3")

            dec_in_1 = contextual_info.cpu()
            dec_in_2 = _contextual_length.to("cpu")
            dec_in_3 = decoder_hidden.cpu()
            dec_in_4 = torch.Tensor([ys_pad_lens]).to(torch.int32).to("cpu")
            for i in range(10):
                start_t = time.time()
                ov_out3 = self.ov_decoder3((dec_in_1, dec_in_2, dec_in_3, dec_in_4 ))
                end_t = time.time()
                elapsed = end_t - start_t
                if i!=0:
                    print(f"dec 3: {elapsed*1000} ms")
            dec_attended = torch.Tensor(ov_out3[0]).to(encoder_out.device)

            merged = self._merge(cif_attended, dec_attended)

            dha_output = self.hotword_output_layer(merged)  # remove the last token in loss calculation
            dha_pred = torch.log_softmax(dha_output, dim=-1)
            def _merge_res(dec_output, dha_output):
                lmbd = torch.Tensor([seaco_weight] * dha_output.shape[0])
                dha_ids = dha_output.max(-1)[-1]# [0]
                dha_mask = (dha_ids == self.NO_BIAS).int().unsqueeze(-1)
                a = (1 - lmbd) / lmbd
                b = 1 / lmbd
                a, b = a.to(dec_output.device), b.to(dec_output.device)
                dha_mask = (dha_mask + a.reshape(-1, 1, 1)) / b.reshape(-1, 1, 1)
                logits = dec_output * dha_mask + dha_output[:,:,:] * (1-dha_mask)
                return logits

            merged_pred = _merge_res(decoder_pred, dha_pred)
            return merged_pred
        else:
            return decoder_pred


class NewCTTransformer(CTTransformer):
    def __init__(self, base_object):
        self.__dict__ = base_object.__dict__

        print("ov load punc")
        core = ov.Core()
        ov_model = core.read_model("openvino/punc_model.xml")
        self.punc = core.compile_model(ov_model, device_name="GPU", config={"INFERENCE_PRECISION_HINT": "f32"})


    def punc_forward(self, text: torch.Tensor, text_lengths: torch.Tensor, **kwargs):
        """Compute loss value from buffer sequences.

        Args:
            input (torch.Tensor): Input ids. (batch, len)
            hidden (torch.Tensor): Target ids. (batch, len)

        """
        x = self.embed(text)

        original_text_lengths = text_lengths
        print("punc_model")

        punc_in_1 = torch.cat((x.to("cpu"), torch.zeros((x.shape[0], 300-text_lengths, x.shape[2])).to("cpu")), dim=1)
        punc_in_2 = text_lengths.to("cpu")
        for i in range(10):
            start_t = time.time()
            ov_out = self.punc((punc_in_1, punc_in_2))
            end_t = time.time()
            elapsed = end_t - start_t
            if i!=0:
                print(f"punc: {elapsed*1000} ms")
        y = torch.tensor(ov_out[0][:, :original_text_lengths, :]).to("cpu")

        return y, None
