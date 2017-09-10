from PIL import Image,ImageFilter
from matplotlib import pyplot as plt
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import sigmoid
import contextlib
from tensorflow.python.util import nest
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from random import shuffle 
from tqdm import tqdm 
from tensorflow.contrib import rnn
from numpy import array, int32
from random import randint
import re
import numpy
import numpy as np
import os
import cv2                 # working with, mainly resizing, images *
import tensorflow as tf

_BIAS_VARIABLE_NAME = "biases"
_WEIGHTS_VARIABLE_NAME = "weights"
@contextlib.contextmanager
def _checked_scope(cell, scope, reuse=None, **kwargs):
  if reuse is not None:
    kwargs["reuse"] = reuse
  with vs.variable_scope(scope, **kwargs) as checking_scope:
    scope_name = checking_scope.name
    if hasattr(cell, "_scope"):
      cell_scope = cell._scope  # pylint: disable=protected-access
      if cell_scope.name != checking_scope.name:
        raise ValueError(
            "Attempt to reuse RNNCell %s with a different variable scope than "
            "its first use.  First use of cell was with scope '%s', this "
            "attempt is with scope '%s'.  Please create a new instance of the "
            "cell if you would like it to use a different set of weights.  "
            "If before you were using: MultiRNNCell([%s(...)] * num_layers), "
            "change to: MultiRNNCell([%s(...) for _ in range(num_layers)]).  "
            "If before you were using the same cell instance as both the "
            "forward and reverse cell of a bidirectional RNN, simply create "
            "two instances (one for forward, one for reverse).  "
            "In May 2017, we will start transitioning this cell's behavior "
            "to use existing stored weights, if any, when it is called "
            "with scope=None (which can lead to silent model degradation, so "
            "this error will remain until then.)"
            % (cell, cell_scope.name, scope_name, type(cell).__name__,
               type(cell).__name__))
    else:
      weights_found = False
      try:
        with vs.variable_scope(checking_scope, reuse=True):
          vs.get_variable(_WEIGHTS_VARIABLE_NAME)
        weights_found = True
      except ValueError:
        pass
      if weights_found and reuse is None:
        raise ValueError(
            "Attempt to have a second RNNCell use the weights of a variable "
            "scope that already has weights: '%s'; and the cell was not "
            "constructed as %s(..., reuse=True).  "
            "To share the weights of an RNNCell, simply "
            "reuse it in your second calculation, or create a new one with "
            "the argument reuse=True." % (scope_name, type(cell).__name__))

    # Everything is OK.  Update the cell's scope and yield it.
    cell._scope = checking_scope  # pylint: disable=protected-access
    yield checking_scope
def _linear(args, output_size, bias, bias_start=0.0):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    weights = vs.get_variable(
        _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size], dtype=dtype)
    if len(args) == 1:
      res = math_ops.matmul(args[0], weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), weights)
    if not bias:
      return res
    with vs.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      biases = vs.get_variable(
          _BIAS_VARIABLE_NAME, [output_size],
          dtype=dtype,
          initializer=init_ops.constant_initializer(bias_start, dtype=dtype))
    return nn_ops.bias_add(res, biases)    
class GRUCell(RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self, num_units, input_size=None, activation=tf.nn.tanh, reuse=None):
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._activation = activation
    self._reuse = reuse

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    with _checked_scope(self, scope or "gru_cell", reuse=self._reuse):
      with vs.variable_scope("gates"):  # Reset gate and update gate.
        # We start with bias of 1.0 to not reset and not update.
        value = sigmoid(_linear(
          [inputs, state], 2 * self._num_units, True, 1.0))
        r, u = array_ops.split(
            value=value,
            num_or_size_splits=2,
            axis=1)
# initialization are from the paper RECURRENT BATCH NORMALIZATION
        r_mean,r_var=tf.nn.moments(r,[1],name="r_moments",keep_dims=True)
        u_mean,u_var=tf.nn.moments(r,[1],name="u_moments",keep_dims=True)
        with vs.variable_scope("r_beta") as rn:
          try:
            rbeta=tf.get_variable("rbeta",r.get_shape()[1],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
            rgamma=tf.get_variable("rgamma",r.get_shape()[1],dtype=tf.float32,initializer= tf.constant_initializer(0.1))
          except ValueError:
            rn.reuse_variables()
            rbeta=tf.get_variable("rbeta",r.get_shape()[1],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
            rgamma=tf.get_variable("rgamma",r.get_shape()[1],dtype=tf.float32,initializer= tf.constant_initializer(0.1))
        with vs.variable_scope("u_beta") as un:
          try:
            ubeta=tf.get_variable("ubeta",r.get_shape()[1],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
            ugamma=tf.get_variable("ugamma",r.get_shape()[1],dtype=tf.float32,initializer= tf.constant_initializer(0.1))
          except ValueError:
            un.reuse_variables()
            ubeta=tf.get_variable("ubeta",r.get_shape()[1],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
            ugamma=tf.get_variable("ugamma",r.get_shape()[1],dtype=tf.float32,initializer= tf.constant_initializer(0.1))
        r=tf.nn.batch_normalization(r,r_mean,r_var,rbeta,rgamma,0.000001)
        u=tf.nn.batch_normalization(u,u_mean,u_var,ubeta,ugamma,0.000001)
      with vs.variable_scope("candidate"):
        #c = self._activation(_linear([inputs, r * state],
         #                            self._num_units, True))
        c=_linear([inputs,r*state],self._num_units, True)
        c_mean,c_var=tf.nn.moments(r,[1],name="c_moments",keep_dims=True)
        with vs.variable_scope("c_beta") as cn:
          try:
            cbeta=tf.get_variable("cbeta",c.get_shape()[1],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
            cgamma=tf.get_variable("cgamma",c.get_shape()[1],dtype=tf.float32,initializer= tf.constant_initializer(0.1))
          except ValueError:
            cn.reuse_variables()
            cbeta=tf.get_variable("cbeta",c.get_shape()[1],dtype=tf.float32,initializer=tf.constant_initializer(0.0))
            cgamma=tf.get_variable("cgamma",c.get_shape()[1],dtype=tf.float32,initializer= tf.constant_initializer(0.1))
        c=self._activation(tf.nn.batch_normalization(c,c_mean,c_var,cbeta,cgamma,0.000001))
      new_h = u * state + (1 - u) * c
    return new_h, new_h
b=100
dim=32
def images_matrix(index):
  
  image_list=[]
  # if index==0:
  #   for image in tqdm (os.listdir('C:/Users/yaseralwattar/Desktop/faces/lfwcrop_grey/lfwcrop_grey/faces')):
  #     path=os.path.join('C:/Users/yaseralwattar/Desktop/faces/lfwcrop_grey/lfwcrop_grey/faces',image)
  #     image_reader=cv2.imread(path,0)     
  #     modified_image=cv2.resize(image_reader,(dim,dim))   
  #     image_list.append([np.array(modified_image)])
  # else:

  #   for image in tqdm (os.listdir('C:/Users/yaseralwattar/Desktop/faces/lfwcrop_grey/lfwcrop_grey/smallest faces')):
  #     path=os.path.join('C:/Users/yaseralwattar/Desktop/faces/lfwcrop_grey/lfwcrop_grey/smallest faces',image)
  #     image_reader=cv2.imread(path,0)
      
  #     modified_image=cv2.resize(image_reader,(dim,dim))
    
  #     image_list.append([np.array(modified_image)])
  if index==0:
    for image in tqdm (os.listdir('C:/Users/yaseralwattar/Desktop/cat faces/clipped faces')):
      path=os.path.join('C:/Users/yaseralwattar/Desktop/cat faces/clipped faces',image)
      image_reader=cv2.imread(path,0)     
      modified_image=cv2.resize(image_reader,(dim,dim))   
      image_list.append([np.array(image_reader)])
  else:

    for image in tqdm (os.listdir('C:/Users/yaseralwattar/Desktop/cat faces/clipped faces')):
      path=os.path.join('C:/Users/yaseralwattar/Desktop/cat faces/clipped faces',image)
      image_reader=cv2.imread(path,0)
      
      modified_image=cv2.resize(image_reader,(dim,dim))
    
      image_list.append([np.array(image_reader)])
  
  
  shuffle(image_list)
  
  image_list=np.array([i[0] for i in image_list]).reshape(-1,dim*dim)

  #image_list=np.array([np.array(i) for i in image_list]).reshape(-1,A*B).astype(np.float32)
  #image_list=np.stack(image_list).reshape(-1,A*B)
  

  image_list=image_list.astype(np.float32)/255

  cv2.destroyAllWindows()
  np.save('image_matrix.npy',image_list)
  return image_list
max_train=150

feed_in=tf.placeholder(tf.float32,shape=(b,dim*dim))
ts=np.arange(0,10,1,np.int32)
num_units=128
gru_in=GRUCell(num_units)
gru_out=GRUCell(num_units)
#h_in0=gru_in.zero_state(b,tf.float32)
h_out0=gru_out.zero_state(b,tf.float32)
canvas=[0.0 for i in range(10)]
e_image=[0.0 for i in range(10)]
read=[0.0 for i in range(10)]
new_in1=[0.0 for i in range(10)]
new_in2=[0.0 for i in range(10)]
new_out=[0.0 for i in range(10)]
mean=[0.0 for i in range(10)]
variance=[0.0 for i in range(10)]
smp=[0.0 for i in range(10)]

for t in ts:
  print("draw %s" %t)
  if t==0:
    e_image[t]=feed_in-tf.sigmoid(tf.zeros([b,dim*dim],dtype=tf.float32,name="canvas0"))
  else:
    e_image[t]=feed_in-tf.sigmoid(canvas[t-1])
  read[t]=tf.concat([feed_in,e_image[t]],1)
  
  with tf.variable_scope("gruin") as in1:

    if t==0.0:
      new_in1[t],new_in2[t]=gru_in(tf.concat([read[t],\
      gru_out.zero_state(b,tf.float32)],1),gru_in.zero_state(b,tf.float32))
    else :
    #except ValueError:
      
      in1.reuse_variables()
      new_in1[t],new_in2[t]=gru_in(tf.concat([read[t],new_out[t-1]],1),new_in2[t-1])

  mean[t]=tf.matmul(new_in1[t],tf.get_variable(name="weightmean%s"%t,shape=[num_units,10]\
  ,dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer()))+\
  tf.Variable(tf.zeros([10]),dtype=tf.float32,name="biasmean%s"%t)
  variance[t]=tf.exp(tf.matmul(new_in1[t],tf.get_variable(name="weightvar%s"%t,shape=[num_units,10]\
  ,dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer()))+\
  tf.Variable(tf.zeros([10]),dtype=tf.float32,name="biasvar%s"%t))
  loss2=tf.reduce_sum(mean[t]*mean[t]+variance[t]*variance[t]-2*tf.log(variance[t]),1)
  smp[t]=mean[t]+tf.multiply(tf.random_normal((b,10)),variance[t])

  with tf.variable_scope("gruout") as out:
    try:
      if t==0:
        new_out[t],_=gru_out(smp[t],h_out0)
      else:
        new_out[t],_=gru_out(smp[t],new_out[t-1])
    except ValueError:
      out.reuse_variables()
      new_out[t],new_out[t]=gru_out(smp[t],new_out[t-1])

  if t==0:
    canvas[t]=tf.add(tf.zeros([b,dim*dim],dtype=tf.float32,name="canvas0"),tf.matmul(new_out[t],\
    tf.get_variable(name="weightc%s"%t,shape=[num_units,dim*dim],dtype=tf.float32,\
    initializer=tf.contrib.layers.xavier_initializer())+tf.Variable(tf.zeros([dim*dim]),\
    dtype=tf.float32,name="biasc%s"%t)))

  else:
    canvas[t]=tf.add(canvas[t-1],tf.matmul(new_out[t],\
    tf.get_variable(name="weightc%s"%t,shape=[num_units,dim*dim],dtype=tf.float32,\
    initializer=tf.contrib.layers.xavier_initializer())+tf.Variable(tf.zeros([dim*dim]),\
    dtype=tf.float32,name="biasc%s"%t)))
  loss2+=loss2
#####################################***********Sig fro last canvas
loss2=tf.reduce_mean(0.5*loss2-5.0)

face=images_matrix(0)
nb=int(len(face)/b)
mean1=tf.convert_to_tensor(mean)
variance1=tf.convert_to_tensor(variance)
print("mean %s var %s "% (mean1.get_shape(),variance1.get_shape()))


loss1=tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits\
(logits=canvas[t],labels=feed_in),1))
# mean2=tf.reduce_sum(mean1*mean1,1)
# variance2=tf.reduce_sum(variance1*variance1,1)
# lnvar=2*tf.reduce_sum(tf.log(variance1),1)
# loss2=tf.reduce_mean(0.5*(mean2+variance2-lnvar)-5)
# loss2=tf.reduce_mean(0.5*tf.reduce_sum(mean1*mean1+variance1*variance1-\
# 2.2*tf.log(variance1),0)-5.0)

loss=tf.add(loss1,loss2)
lr_o=tf.train.RMSPropOptimizer(0.00099).minimize(loss)
loss_val=[0.0 for i in range(max_train)]
loss1_val=[0.0 for i in range(max_train)]
loss2_val=[0.0 for i in range(max_train)]
sess=tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(max_train):
  sb=0
  for j in range(nb):
    print("j %s"%j)
    sface=face[j*b:j*b+b]
    sface1=np.array(sface).astype(np.float32)
    sb+=b
    ev_tuple=sess.run([loss,loss1,loss2,lr_o],{feed_in:sface1})
  loss_val[step],loss1_val[step],loss2_val[step],_=ev_tuple
  if step%2==0:
    
    print("round: %s loss: %s loss1: %s loss2: %s"%(step,loss_val[step],loss1_val[step],loss2_val[step]))#########################***************for evaluation only to be removed 

print("Hello")

test_face=images_matrix(1)

images=sess.run(canvas,{feed_in:test_face})

###############################
np.save("C:/Users/yaseralwattar/tensorflow-examples/draw/under_test.npy",[np.array(images)])
np.save("C:/Users/yaseralwattar/tensorflow-examples/draw/loss.npy",loss_val)
sess.close()

