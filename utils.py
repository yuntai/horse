# from transformers repo
# https://github.com/huggingface/transformers.git
# transformers/src/transformers/modeling_xml.py

# Copyright 2019-present, Facebook, Inc and the HuggingFace Inc. team.
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


def create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()
    out.requires_grad = False


def get_masks(slen, lengths, causal, padding_mask=None):                           
    """                                                                            
    Generate hidden states mask, and optionally an attention mask.                 
    """                                                                            
    alen = torch.arange(slen, dtype=torch.long, device=lengths.device)             
    if padding_mask is not None:                                                   
        mask = padding_mask                                                        
    else:                                                                          
        assert lengths.max().item() <= slen                                        
        mask = alen < lengths[:, None]                                             
                                                                                   
    # attention mask is the same as mask, or triangular inferior attention (causal)
    bs = lengths.size(0)                                                           
    if causal:                                                                     
        attn_mask = alen[None, None, :].repeat(bs, slen, 1) <= alen[None, :, None] 
    else:                                                                          
        attn_mask = mask                                                           
                                                                                   
    # sanity check                                                                 
    assert mask.size() == (bs, slen)                                               
    assert causal is False or attn_mask.size() == (bs, slen, slen)                 
                                                                                   
    return mask, attn_mask                                                         

