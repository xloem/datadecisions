from .transformer import Config, Model

class TransformerDecisions:
  def __init__(self, model):
    self.model = model
  @classmethod
  def load(cls, pathname):
    # could also load config first, create, and use an instance method
    return cls(Model.from_pretrained(pathname))
  @classmethod
  def create(cls, observations, actions, rewards=1, steps=4096, embeds=128, heads=1, layers=3):
    return cls(Model(Config(
        state_dim=observations,
        act_dim=actions,
        reward_dim=rewards,
        hidden_size=embeds,
        max_ep_len=steps,
        n_layer=layers,
        n_head=heads,
    )))
