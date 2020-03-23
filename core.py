import numpy as np
import tensorflow as tf
from util import store_args, nn
from normalizer import Normalizer

EPS = 1e-6
LOG_STD_MAX = 2
LOG_STD_MIN = -20

def mlp(x, layers_sizes, reuse=None, flatten=False, name=""):
    for i,size in enumerate(layers_sizes):
        activation = tf.nn.relu if i < len(layers_sizes) - 1 else None
        x = tf.layers.dense(inputs=x, units=size,
                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                            reuse=reuse,activation=activation,
                            name=name + '_' + str(i))
        if activation:
            x = activation(x)
    if flatten:
        assert layers_sizes[-1] == 1
        x = tf.reshape(x, [-1])
    return x

def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)


def mlp_gaussian_policy(x, a, dimu, layers_sizes,output_activation):
    act_dim = dimu
    net = mlp(x, layers_sizes)
    mu = tf.layers.dense(net, act_dim, activation=output_activation)

    log_std = tf.layers.dense(net, act_dim, activation=tf.tanh)
    log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
    # log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return mu, pi, logp_pi

def apply_squashing_func(mu, pi, logp_pi):
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)
    return mu, pi, logp_pi


class mlp_actor_critic:
    @store_args
    def __init__(self, inputs_tf, dimo, dimg, dimu, max_u, o_stats, g_stats, hidden, layers,
                 **kwargs):
        """The actor-critic network and related training code.

        Args:
            inputs_tf (dict of tensors): all necessary inputs for the network: the
                observation (o), the goal (g), and the action (u)
            dimo (int): the dimension of the observations
            dimg (int): the dimension of the goals
            dimu (int): the dimension of the actions
            max_u (float): the maximum magnitude of actions; action outputs will be scaled
                accordingly
            o_stats (baselines.her.Normalizer): normalizer for observations
            g_stats (baselines.her.Normalizer): normalizer for goals
            hidden (int): number of hidden units that should be used in hidden layers
            layers (int): number of hidden layers
        """
        self.o_tf = inputs_tf['o']
        self.g_tf = inputs_tf['g']
        self.u_tf = inputs_tf['u']

        # Prepare inputs for actor and critic.
        o = self.o_stats.normalize(self.o_tf)
        g = self.g_stats.normalize(self.g_tf)
        input_pi = tf.concat(axis=1, values=[o, g])  # for actor

        # Networks.
        with tf.variable_scope('pi'):
            self.mu_tf, self.pi_tf, self.logp_pi_tf = mlp_gaussian_policy(input_pi, self.u_tf, dimu,
                                                                          layers_sizes=[self.hidden] * self.layers + [self.dimu],
                                                                          output_activation=None)
            self.mu_tf, self.pi_tf, self.logp_pi_tf = apply_squashing_func(self.mu_tf, self.pi_tf, self.logp_pi_tf)

        # vf_mlp = lambda x: tf.squeeze(mlp(x, layers_sizes=[self.hidden] * self.layers + [1],
        #                                   ), axis=1)
        # vf_mlp1 = lambda x: tf.squeeze(mlp(x, layers_sizes=[self.hidden] * self.layers + [1],
        #                                   reuse=True), axis=1)

        with tf.variable_scope('q1'):
            self.q1_pi_tf = mlp(tf.concat(axis=1, values=[o, g, self.pi_tf]),
                                layers_sizes=[self.hidden] * self.layers + [1])
            self.q1_tf = mlp(tf.concat(axis=1, values=[o, g, self.u_tf]),
                             layers_sizes=[self.hidden] * self.layers + [1], reuse=True)
        with tf.variable_scope('q2'):
            self.q2_pi_tf = mlp(tf.concat(axis=1, values=[o, g, self.pi_tf]),
                                layers_sizes=[self.hidden] * self.layers + [1])
            self.q2_tf = mlp(tf.concat(axis=1, values=[o, g, self.u_tf]),
                             layers_sizes=[self.hidden] * self.layers + [1], reuse=True)
        with tf.variable_scope('v'):
            self.v_tf = mlp(input_pi,layers_sizes=[self.hidden] * self.layers + [1])

