# Copyright (c) 2023 Ant Group and its affiliates.
import numpy
from antmmf.common.registry import registry
from antmmf.utils import tensor_utils
from antmmf.common.constants import SS_GRAD_INPUT
from .simple_gradient import SimpleGradientInterpreter


@registry.register_interpreter("integrated-gradient")
class IntegratedGradientInterpreter(SimpleGradientInterpreter):
    """
    Registered as "simple-gradient".
    """

    def __init__(self, predictor, config) -> None:
        super().__init__(predictor, config)
        self._steps = config.get("steps", 10)

    def _interpret(self, inputs, module_name):
        """
        Interprets the model's prediction for inputs.
        Gets the gradients of the loss with respect
        to the input and returns those gradients normalized and sanitized.
        """
        self._predictor.eval()
        instances_with_grads = dict()

        epsilon = 1e-10

        # Use 10 terms in the summation approximation of the integral in integrated grad
        steps = self._steps

        # Hook used for saving embeddings
        embedding_layer = tensor_utils.find_embedding_layer(
            self._predictor, module_name
        )
        # List of embedding inputs, used for multiplying gradient by the input for normalization
        embeddings_list = []
        grad_dict = dict()
        # Exclude the endpoint because we do a left point integral approximation
        idx = 0
        for alpha in numpy.linspace(0, 1.0, num=steps, endpoint=False):
            if embedding_layer is not None:
                handles = self._register_hooks(alpha, embeddings_list, embedding_layer)

            embedding_gradients, outputs = self.get_gradient(inputs, embedding_layer)

            for handle in handles:
                handle.remove()

            if SS_GRAD_INPUT not in grad_dict:
                grad_dict[SS_GRAD_INPUT] = embedding_gradients[0].detach().cpu().numpy()
            else:
                grad_dict[SS_GRAD_INPUT] += (
                    embedding_gradients[0].detach().cpu().numpy()
                )

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

    def _register_hooks(self, alpha, embeddings_list, embedding_layer):
        """
        Register a forward hook on the embedding layer which scales the embeddings by alpha. Used
        for one term in the Integrated Gradients sum.
        We store the embedding output into the embeddings_list when alpha is zero.  This is used
        later to element-wise multiply the input by the averaged gradients.
        """

        def forward_hook(module, inputs, output):
            # Save the input for later use. Only do so on first call.
            if alpha == 0:
                embeddings_list.append(output.squeeze(0).clone().detach())

            # Scale the embedding by alpha
            output.mul_(alpha)

        # Register the hooks
        handles = []
        handles.append(embedding_layer.register_forward_hook(forward_hook))
        return handles
