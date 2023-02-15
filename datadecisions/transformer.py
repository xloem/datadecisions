from transformers import DecisionTransformerModel, DecisionTransformerConfig

class MultiRewardTransformerModel(DecisionTransformerModel):
    # add multiple rewards via config.reward_dim
    def __init__(self, config):
        super().__init__(config)
        # self.encoder is a GPT2 model without position embeddings
        self.embed_return = torch.nn.Linear(config.reward_dim, config.hidden_size)
    # add loss, from hf example
    def forward(self, **kwparams):
        output = super().forward(**kwparams)
        action_targets = kwparams['actions']
        attention_mask = kwparams['attention_mask']
        act_dim = output.action_preds.shape[2]
        # i'm not sure why this works, isn't it matching the predicted with the recorded that generated it? shouldn't it be shifted by 1?
        action_preds = output.action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_targets = action_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        output['loss'] = torch.mean((action_preds - action_targets) ** 2)
        return output
Model = MultiRewardTransformerModel

class MultiRewardTransformerConfig(DecisionTransformerConfig):
    def __init__(self, reward_dim=1, **kpwarams):
        self.reward_dim = reward_dim
        super().__init__(**kwparams)
Config = MultiRewardTransformerConfig
