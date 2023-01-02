# /usr/bin/env python3

from typing import Dict
import torch
import numpy as np
import nilearn as nl
from numpy import ndarray
# import os, psutil
from cnnFeature.firstCnnModel import Enc
import cnnFeature.configs
from torch.autograd import Variable
from torchsummary import summary


class Connectome():

    def __init__(
            self,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.name = 'CSMEmbedder'
        self.training_style = 'CSM'
        assert self.training_style in {'CSM', 'decoding'}, f'{self.training_style} not supported'
        self._root_training_style = 'CSM'
        self.msk_embed = torch.nn.Parameter(
            torch.empty(
                size=(1, 1, self.in_dim)
            )
        )
        self.cls_embed = torch.nn.Parameter(
            torch.empty(
                size=(1, 1, self.in_dim)
            )
        )
        self._embeds = [
            self.msk_embed,
            self.cls_embed
        ]
        self._init_embeds()
        # print('cseinit')

    def connectome_batch(
            self,
            batch: Dict[str, torch.tensor],
            regionOfInterests,
            fmriData,
            sample

    ) -> ndarray:
        if batch==None:
            batch = {
                # "fmriData": None,
                # "labels": None,
        }
        batch_out = dict(batch)
        labels = torch.clone(batch['labels']) if 'labels' in batch else None

        if labels is not None:
            batch_out['labels'] = labels

        seqLen=(fmriData.shape)[0]
        if regionOfInterests is None:
            regionOfInterests=range(seqLen)

        #TODO subtrack some of the time to match the other data for csm training

        fmriData=fmriData.data
        #fmriData=np.pad(fmriData, [(0,1)], mode='constant', constant_values=0)
        # print('convConnectome feature sizes')
        # print(sample)
        # print(fmriData.shape)
        #fmriData=fmriData.transpose()

        #connectome=fmriData[:, 0:1000]
        print('sample issue')
        print(fmriData.shape)
        print(int(0))
        fmriData=fmriData[:,int(sample)]
        #fmriData=[0,fmriData]

        fmriData=torch.tensor(fmriData).cuda()
        #fmriData=[0,fmriData]
        img=fmriData.clone()
        img=torch.nn.functional.normalize(img, p=2.0, dim=0)
        img = Variable(torch.tensor(img)).cuda()
        #fmriData=(1,fmriData,20484)

        model = Enc().cuda()
        model.load_state_dict(torch.load(cnnFeature.configs.resultPath + "/enc_epoch_4_loss_0.004465451929718256")['model_state_dict'])
        #model.load_state_dict(torch.load(cnnFeature.configs.resultPath + "/enc_epoch_115_loss_0.000305522873532027"))
        model.eval()
        # print('input for enc')
        # print(img.size())
        # print(img)
        img=torch.unsqueeze(img,dim=0)
        #img = fmriData.clone
        #summary(model,(1,20484))
        connectome = model(img)
        connectome=connectome.cpu().detach().numpy()

        return connectome

    def _init_embeds(self):

        for embed in self._embeds:
            torch.nn.init.normal_(
                tensor=embed,
                mean=0.0,
                std=1.0,
            )

    def prep_batch(
            self,
            batch: Dict[str, torch.tensor],
    ) -> Dict[str, torch.tensor]:
        batch_out = dict(batch)
        labels = torch.clone(batch['labels']) if 'labels' in batch else None
        # print(' prep_batch')
        # print(batch.keys())
        # print(batch['inputs'].size())
        # print(batch['inputs'])

        if self.training_style != 'decoding':
            return self.mask_inputs(batch=batch_out)
        print('after mskin')
        batch_out = self.add_cls_embed(batch=batch_out)

        if labels is not None:
            batch_out['labels'] = labels

        return batch_out

    def mask_inputs(
            self,
            batch: Dict[str, torch.tensor],
    ) -> Dict[str, torch.tensor]:

        inputs_key = 'inputs' if 'inputs_embeds' not in batch else 'inputs_embeds'
        assert inputs_key in batch, f'{inputs_key} not found in batch'
        input_shape = batch[inputs_key].size()
        device = batch[inputs_key].device
        # print('mskinput')
        masking_i = torch.cat(
            [
                torch.randint(
                    low=1,  # at least one seq value before mask!
                    high=sum(batch['attention_mask'][i] == 1),  # high is exclusive, so this accounts for 0-indexing
                    size=(1,),
                    device=device
                )
                for i in range(input_shape[0])
            ],
            dim=0
        )
        # print('masking_i')
        # print(masking_i)
        modelling_mask = torch.zeros_like(
            batch[inputs_key],
            device=device
        )
        modelling_mask[torch.arange(input_shape[0]), masking_i] = 1
        batch['modelling_mask'] = modelling_mask.to(torch.long)
        batch['masked_inputs'] = torch.masked_select(
            input=batch[inputs_key],
            mask=batch['modelling_mask'].to(torch.bool)
        ).detach().clone()
        batch['inputs_embeds'] = torch.where(
            batch['modelling_mask'] == 1,
            self.msk_embed.repeat(
                input_shape[0],
                input_shape[1],
                1
            ),
            batch[inputs_key].to(torch.float)
        )
        # print('input_embeds')
        # print(batch['inputs_embeds'])
        batch['attention_mask'] = torch.cat(
            [
                torch.cat(
                    (
                        torch.ones(
                            (
                                1,
                                i + 1  # to account for 0-indexing in python
                            ),
                            device=device
                        ),
                        torch.zeros(
                            (
                                1,
                                input_shape[1] - i - 1  # to account for 0-indexing in python
                            ),
                            device=device
                        )
                    ),
                    dim=1
                )
                for i in masking_i
            ],
            dim=0
        ).to(torch.long)
        # re-mask inputs
        attention_mask_expanded = torch.unsqueeze(
            batch['attention_mask'],
            dim=2
        ).repeat(
            1,
            1,
            self.in_dim
        )
        batch["inputs_embeds"] = torch.where(
            attention_mask_expanded == 1,
            batch['inputs_embeds'],
            torch.zeros_like(batch['inputs_embeds'])
        )
        print('further inputs')
        print(batch.keys())
        print(batch['masked_inputs'].size())
        print(batch['inputs_embeds'].size())
        print(batch['attention_mask'].size())
        print(batch['modelling_mask'].size())

        return batch

    def add_cls_embed(
            self,
            batch: Dict[str, torch.tensor]
    ) -> Dict[str, torch.tensor]:
        inputs_key = 'inputs' if 'inputs_embeds' not in batch else 'inputs_embeds'
        assert inputs_key in batch, f'{inputs_key} not found in batch'
        batch_size = batch[inputs_key].size()[0]
        sequence_lengths = batch['attention_mask'].sum(dim=1)
        inputs_embeds = []
        print('cls embed')
        if 't_rs' in batch:
            t_rs = []
        one = []
        for i in range(len(sequence_lengths)):
            inputs_embeds.append(
                torch.cat(
                    [
                        batch[inputs_key][i, :sequence_lengths[i], :],
                        self.cls_embed[0],
                        batch[inputs_key][i, sequence_lengths[i]:, :]
                    ],
                    dim=0
                )
            )
            # print('t_rs' in batch)
            if 't_rs' in batch:
                print(batch['t_rs'][i, :sequence_lengths[i]])
                t_rs.append(
                    torch.cat(
                        [
                            batch['t_rs'][i, :sequence_lengths[i]],
                            torch.ones(1, device=batch['t_rs'].device) * -1,
                            batch['t_rs'][i, sequence_lengths[i]:]
                        ],
                        dim=0
                    )
                )
            one.append(
                torch.cat(
                    [
                        torch.ones(batch['t_rs'][i, :sequence_lengths[i]].size(), device=batch['t_rs'].device),
                        torch.ones(1, device=batch['t_rs'].device) * -1,
                        torch.ones(batch['t_rs'][i, sequence_lengths[i]:].size(), device=batch['t_rs'].device)
                    ],
                    dim=0
                )
            )

        batch['inputs_embeds'] = torch.stack(
            inputs_embeds,
            dim=0
        )
        batch['one'] = torch.stack(
            one,
            dim=0
        )
        if 't_rs' in batch:
            batch['t_rs'] = torch.stack(
                t_rs,
                dim=0
            )

        if 'token_type_ids' in batch:
            batch['token_type_ids'] = self._pad_tensor_left_by_n(
                tensor=batch['token_type_ids'],
                n=1,
                pad_value=0
            )

        if 'modelling_mask' in batch:
            batch['modelling_mask'] = self._pad_tensor_left_by_n(
                tensor=batch['modelling_mask'],
                n=1,
                pad_value=0
            )

        if 'attention_mask' in batch:
            batch['attention_mask'] = self._pad_tensor_left_by_n(
                tensor=batch['attention_mask'],
                n=1,
                pad_value=1
            )

        return batch

    def masking_loss(
            self,
            masked_inputs,
            outputs,
            modelling_mask
    ) -> Dict[str, torch.tensor]:
        print('mskloss')
        return {
            'masking_loss': self.reconstruction_loss(
                input=torch.masked_select(outputs, modelling_mask.to(torch.bool)),
                target=masked_inputs
            )['reconstruction_loss']
        }

    def _root_loss(
            self,
            masked_inputs,
            outputs,
            modelling_mask,
            **kwargs
    ) -> Dict[str, torch.tensor]:
        print('rpptloss')
        return self.masking_loss(
            masked_inputs=masked_inputs,
            outputs=outputs,
            modelling_mask=modelling_mask
        )
