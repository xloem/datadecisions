
idea?
- some code to facilitate training and using transformer models as agents in novel environment in ways similar to existing

notes:
- hf decisiontransformer with multiple rewards
- environment callback that simplifies openai gym into a function with self-documenting action and observation parameters
- process recorded data

missing:
- primarily, actually _training_ a decision transformer. it would be fun to adapt a language model around properties of its outputs.
      but it would likely make sense to go through the huggingface tutorial on decision transformers.
- some things


### more notes

#### trainer
Loss should be the first element of tuple output and is only used if a "labels" argument is provided (according to docs).

#### datasets
Formally the datasets used by normative example code are arraylikes inheriting from `torch.utils.data.Dataset`.
They need only implement `__getitem__` and `__len__` and items can be tuples or dicts, analogous to csv or jsonlines files.

Contrariwise the DataLoader selects and batches items together for training.

Judging by the tutorial example, there's a reasonable chance that each subitem is stacked into a batch and forwarded as a named or ordered parameter directly to the model's `.forward` function.
