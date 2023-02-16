class TransformerDecisions:
  def __init__(self, model):
    self.model = model
  @classmethod
  def create(cls, observations, actions, rewards=1, steps=4096, embeds=128, heads=12, layers=12):
  
