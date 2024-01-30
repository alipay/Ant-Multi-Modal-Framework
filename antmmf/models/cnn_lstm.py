# Copyright (c) 2023 Ant Group and its affiliates.
import torch

from torch import nn

from antmmf.common.registry import registry
from antmmf.models.base_model import BaseModel
from antmmf.modules.layers import ConvNet


_TEMPLATES = {
    "question_vocab_size": "{}_text_vocab_size",
    "number_of_answers": "{}_num_final_outputs",
}

_CONSTANTS = {"hidden_state_warning": "hidden state (final) should have 1st dim as 2"}


@registry.register_model("cnn_lstm")
class CNNLSTM(BaseModel):
    """CNNLSTM is a simple model for vision and language tasks. CNNLSTM is supposed to act
    as a baseline to test out your stuff without any complex functionality. Passes image
    through a CNN, and text through an LSTM and fuses them using concatenation. Then, it finally
    passes the fused representation from a MLP to generate scores for each of the possible answers.

    Args:
        config (Configuration): Configuration node containing all of the necessary config required
                             to initialize CNNLSTM.

    Inputs: sample_list (SampleList)
        - **sample_list** should contain image attribute for image, text for question split into
          word indices, targets for answer scores
    """

    def __init__(self, config):
        super().__init__(config)
        self._datasets = []
        for _, attr in registry.get("config").task_attributes.items():
            for dataset in attr.dataset_attributes:
                self._datasets.append(dataset)

    def build(self):
        assert len(self._datasets) > 0
        num_question_choices = registry.get(
            _TEMPLATES["question_vocab_size"].format(self._datasets[0])
        )
        num_answer_choices = registry.get(
            _TEMPLATES["number_of_answers"].format(self._datasets[0])
        )

        self.text_embedding = nn.Embedding(
            num_question_choices, self.config.text_embedding.embedding_dim
        )
        self.lstm = nn.LSTM(**self.config.lstm)

        layers_config = self.config.cnn.layers
        conv_layers = []
        for i in range(len(layers_config.input_dims)):
            conv_layers.append(
                ConvNet(
                    layers_config.input_dims[i],
                    layers_config.output_dims[i],
                    kernel_size=layers_config.kernel_sizes[i],
                )
            )
        self.cnn = nn.Sequential(*conv_layers)

        self.classifier = nn.Linear(
            self.config.classifier.input_dim, num_answer_choices
        )

    def forward(self, sample_list):
        self.lstm.flatten_parameters()

        question = sample_list.text
        image = sample_list.image

        # Get (h_n, c_n), last hidden and cell state
        _, hidden = self.lstm(self.text_embedding(question))
        # X x B x H => B x X x H where X = num_layers * num_directions
        hidden = hidden[0].transpose(0, 1)

        # X should be 2 so we can merge in that dimension
        assert hidden.size(1) == 2, _CONSTANTS["hidden_state_warning"]

        hidden = torch.cat([hidden[:, 0, :], hidden[:, 1, :]], dim=-1)
        image = self.cnn(image)
        if image.dim() > 1:
            image = torch.flatten(image, 1)

        # Fuse into single dimension
        fused = torch.cat([hidden, image], dim=-1)
        scores = self.classifier(fused)
        return {"logits": scores}

    def get_adv_parameters(self):
        # return parameters for adversarial genration and training
        # only do adversarial on the word-embeddings, not on others such as
        # positional embedding
        adv_param_names = ["text_embedding.weight"]

        params = []
        for n, v in self.named_parameters():
            if n in adv_param_names:
                params.append(v)
        ret = [
            {"params": p, "name": n, "modality": "text"}
            for n, p in zip(adv_param_names, params)
        ]
        return ret

    def init_adv_train(self):
        r"""
        during adversarial training, the model needs to be in train mode
        because cudnn RNN backward can only be called in training mode
        this is a bug in cudnn.

        in principle, it should use eval mode.
        """
        if torch.cuda.is_available():
            self.train()
        else:
            self.eval()
