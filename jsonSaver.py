# We can now implement the TMRL ActorModule interface that we are supposed to submit for this competition.

# During training, TMRL will regularly save our trained ActorModule in the TmrlData/weights folder.
# By default, this would be done using the torch (i.e., pickle) serializer.
# However, while saving and loading your own pickle files is fine,
# it is highly dangerous to load other people's pickle files.
# Therefore, the competition submission does not accept pickle files.
# Instead, we can submit our trained weights in the form of a human-readable JSON file.
# The ActorModule interface defines save() and load() methods that we will override with our own JSON serializer.

import json
import torch


class TorchJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for torch tensors, used in the custom save() method of our ActorModule.
    """
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return json.JSONEncoder.default(self, obj)


class TorchJSONDecoder(json.JSONDecoder):
    """
    Custom JSON decoder for torch tensors, used in the custom load() method of our ActorModule.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):
        for key in dct.keys():
            if isinstance(dct[key], list):
                dct[key] = torch.Tensor(dct[key])
        return dct