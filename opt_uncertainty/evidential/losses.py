import tensorflow as tf
import numpy as np


def Dirichlet_SOS(y, alpha):
    def KL(alpha):
        beta = tf.ones((1, alpha.shape[1]), dtype=tf.float32)
        S_alpha = tf.reduce_sum(alpha, axis=1, keepdims=True)
        S_beta = tf.reduce_sum(beta, axis=1, keepdims=True)
        lnB = tf.math.lgamma(S_alpha) - tf.reduce_sum(tf.math.lgamma(alpha),axis=1,keepdims=True)
        lnB_uni = tf.reduce_sum(tf.math.lgamma(beta), axis=1, keepdims=True) - tf.math.lgamma(S_beta)

        dg0 = tf.math.digamma(S_alpha)
        dg1 = tf.math.digamma(alpha)

        kl = tf.reduce_sum((alpha - beta) * (dg1-dg0), axis=1, keepdims=True) + lnB + lnB_uni
        return kl

    S = tf.reduce_sum(alpha, axis=1, keepdims=True)
    evidence = alpha - 1
    m = alpha / S

    A = tf.reduce_sum((y-m)**2, axis=1, keepdims=True)
    B = tf.reduce_sum(alpha*(S-alpha)/(S*S*(S+1)), axis=1, keepdims=True)

    # annealing_coef = tf.minimum(1.0,tf.cast(global_step/annealing_step,tf.float32))

    # prob = alpha/tf.reduce_sum(alpha, 1, keepdims=True)
    alp = evidence * (1-y) + 1

    # C = tf.reduce_mean(alp, axis=1)
    C =  1 * KL(alp)

    return tf.reduce_mean(A + B + C)

def Softmax_CE(y, logits):
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
    return tf.reduce_mean(loss)
