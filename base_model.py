import tensorflow.compat.v1 as tf
import tensorflow as tfv2
from tensorflow.python.framework import ops

# class FlipGradientBuilder(object):
#     def __init__(self):
#         self.num_calls = 0
#
#     def __call__(self, x, l=1.0):
#         grad_name = "FlipGradient%d" % self.num_calls
#         @ops.RegisterGradient(grad_name)
#         def _flip_gradients(op, grad):
#             # 梯度翻转，通过将梯度乘以 -l 来实现的
#             return [tf.negative(grad)*l]
#
#         g=tf.get_default_graph()
#         # gradient_override_map来覆盖默认的梯度。这里将 "Identity" 操作的梯度替换为 grad_name 对应的梯度。
#         with g.gradient_override_map({"Identity":grad_name}):
#             y=tf.identity(x)
#
#         self.num_calls+=1
#         return y


class FlipGradientBuilder:
    def __init__(self):
        self.num_calls = 0

    def __call__(self, x, l=1.0):
        @tfv2.custom_gradient
        def flip_gradient(x):
            def grad(dy):
                return -l * dy  # 梯度翻转，乘以负的 l 值
            return x, grad

        self.num_calls += 1
        return flip_gradient(x)
