from transformers import DecisionTransformerModel

class MultiRewardTransformerModel(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)
        # self.encoder is a GPT2 model without position embeddings
        self.embed_return = torch.nn.Linear(config.reward_dim, config.hidden_size)
