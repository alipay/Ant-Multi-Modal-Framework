# Copyright (c) 2023 Ant Group and its affiliates.
from typing import List
import torch

from antmmf.common.registry import registry
from antmmf.utils import tensor_utils
from antmmf.common.constants import SS_GRAD_INPUT
from .saliency_interpreter import Interpreter


@registry.register_interpreter("simple-gradient")
class SimpleGradientInterpreter(Interpreter):
    """
    Registered as "simple-gradient".
    """

    def __init__(self, predictor, config) -> None:
        super().__init__(predictor, config)
        self.module = config.get("module", [])

    def _interpret(self, inputs, module_name):
        """
        Interprets the model's prediction for inputs.
        Gets the gradients of the loss with respect
        to the input and returns those gradients normalized and sanitized.
        """
        self._predictor.eval()
        instances_with_grads = dict()

        # List of embedding inputs, used for multiplying gradient by the input for normalization
        embeddings_list: List[torch.Tensor] = []

        # Hook used for saving embeddings
        embedding_layer = tensor_utils.find_embedding_layer(
            self._predictor, module_name
        )
        if embedding_layer is not None:
            handles = self._register_hooks(embeddings_list, embedding_layer)

        embedding_gradients, outputs = self.get_gradient(inputs, embedding_layer)

        for handle in handles:
            handle.remove()

        grad_dict = dict()
        key = SS_GRAD_INPUT
        grad_dict[key] = embedding_gradients[0].detach().cpu().numpy()

        # Gradients come back in the reverse order that they were sent into the network
        embeddings_list.reverse()
        embeddings_list = self._aggregate_token_embeddings(embeddings_list)

        grad_dict = self._compute_saliency(grad_dict, embeddings_list)

        instances_with_grads = grad_dict
        instances_with_grads.update(outputs)
        """ for example
        ret {'instance_1': {'grad_input_1': [0.4725051675965148, 0.07694429933375777, 0.16593637800930253, 0.0], 
                            'grad_input_2': [0.09523809523809523, 0.0, 0.2857142857142857, 0], 
                            'grad_input_3': [1.0], 
                            'grad_input_4': [1.0]}}
        """
        return instances_with_grads

    def interpret(self, inputs):
        """
        Interprets the model's prediction for inputs for each module
        """

        ret = {}
        for module in self.module:
            module.defrost()  # make the module be writable
            name = module["name"]
            ret[name] = module
            ret[name]["attributions"] = self._interpret(inputs, name)

        return ret

    def _register_hooks(self, embeddings_list, embedding_layer):
        """
        Finds all of the image/text embeddings, and registers a forward hook onto them. When forward()
        is called, embeddings_list is filled with the embedding values. This is necessary because
        our normalization scheme multiplies the gradient by the embedding value.
        """

        def forward_hook(module, inputs, output):
            embeddings_list.append(output.squeeze(0).clone().detach())

        # Register the hooks
        handles = []
        handles.append(embedding_layer.register_forward_hook(forward_hook))
        return handles

    def _register_embedding_gradient_hooks(self, embedding_gradients, embedding_layer):
        """
        Registers a backward hook on the embedding layer of the model.  Used to save the gradients
        of the embeddings for use in get_gradients()
        When there are multiple inputs (e.g., a passage and question), the hook
        will be called multiple times. We append all the embeddings gradients
        to a list.
        """

        def hook_layers(module, grad_in, grad_out):
            grads = grad_out[0]
            embedding_gradients.append(grads)

        hooks = []
        if embedding_layer is not None:
            hooks.append(embedding_layer.register_backward_hook(hook_layers))

        return hooks
