import torch


class Conversation:
    def __init__(self, type: str, system_text=None) -> None:
        self.type = type
        self.exchanges = []
        self.system_text = system_text
    
    def add_exchange(self, input_text: str, output_text: str):
        self.exchanges.append({
            "input": input_text,
            "output": output_text
        })


@torch.no_grad() 
def get_casual_mask(size: int) -> torch.Tensor:
    # Lower triangular matrix
    # [[
    #   [True, False, ... , False],
    #   [True, True,  ... , False],
    #   [True, True,  ... , False],
    #   [True, True,  ... , True ]
    # ]]
    # 1 x size x size
    idx = torch.arange(size, dtype=torch.int)
    return (idx[None, :, None] >= idx[None, None, :]) # mask[i, j] = True if i ≥ j, else False.