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
dim=32
def spatial_attention(new_out,N,index,time,feed_in,e_image):
	###########important a ,b from 1 to before 33
	a=np.arange(0,dim,1,np.float32)
	b1=np.arange(0,dim,1,np.float32)
	patch_rows=np.arange(0,N,1,np.float32)
	patch_col=np.arange(0,N,1,np.float32)
	mean_x_p=[]
	mean_y_p=[]# filter centers
	varw=tf.matmul(new_out, tf.Variable(tf.random_normal([num_units,N],stddev=0.1),dtype=tf.float32,name="att_weights%s" %t))\
	+tf.Variable(tf.zeros([N]),dtype=tf.float32,name="att_biases%s"%t) #100 * 1024

	d_x=varw[:,0]
	d_x=tf.reshape(d_x,(100,1))
	#print("d_x%s "% d_x)
	d_y=varw[:,1]
	d_y=tf.reshape(d_y,(100,1))
	log_is_var=varw[:,2]
	log_is_var=tf.reshape(log_is_var,(100,1))
	log_d_str=varw[:,3]
	log_d_str=tf.reshape(log_d_str,(100,1))
	log_int=varw[:,4]
	log_int=tf.reshape(log_int,(100,1))
	x_center=tf.multiply((33.0/2.0),d_x+1)#100*1
	y_center=tf.multiply((33.0/2.0),d_y+1)# Equation 23
	stride=tf.multiply(31.0/(N-1.0),tf.exp(log_d_str))
	#print("log_d_str%s "% log_d_str)
	for i in patch_rows:
		#print("i %s" %(i-5/2-0.5))
		#mean_x_p1=x_center+tf.multiply(tf.to_float(i-5.0/2.0-0.5),tf.exp(stride))
		mean_x_p1=x_center+tf.multiply(tf.to_float(i-5.0/2.0-0.5),stride)
		mean_x_p.append(mean_x_p1)
	for j in patch_col:
		# mean_y_p1=y_center+tf.multiply(tf.to_float(j-5.0/2.0-0.5),tf.exp(stride))#100*1
		mean_y_p1=y_center+tf.multiply(tf.to_float(j-5.0/2.0-0.5),stride)
		mean_y_p.append(mean_y_p1)#100*5
	#print("mean_y_p %d " % len(mean_y_p))
	#print("a%s"%a)
	#print("mean_y_p%s"% mean_y_p[4])
	for i in range(5):
		filterbank_x1=tf.exp(tf.multiply((a-mean_x_p[i]),(a-mean_x_p[i]))/(-2*tf.exp(log_is_var)))
		filterbank_y1=tf.exp(tf.multiply((b1-mean_y_p[i]),(b1-mean_y_p[i]))/(-2*tf.exp(log_is_var)))

		if i==0:
			filterbank_x3=filterbank_x1
			filterbank_y3=filterbank_y1
		else:
			filterbank_x3=tf.concat([filterbank_x3,filterbank_x1],1)
			filterbank_y3=tf.concat([filterbank_y3,filterbank_y1],1)

	filterbank_x3=tf.reshape(filterbank_x3,(100,N,dim)) # Fx
	filterbank_y3=tf.reshape(filterbank_y3,(100,N,dim)) # Fy
	##########################??????????????????????????????????????@@@@@@@@@@@@!@@@@ maximum @@@
	sum_over_a= tf.reduce_sum(filterbank_x3,2)
	sum_over_b=tf.reduce_sum(filterbank_y3,2)
	sum_over_a=tf.reshape(sum_over_a,(100,N,1))
	sum_over_b=tf.reshape(sum_over_b,(100,N,1))
	#print("sum_over_a %s" % sum_over_a )
	if sum_over_a !=0:
		filterbank_x3=filterbank_x3/tf.clip_by_value(sum_over_a,0.1,1)

		
	if sum_over_b !=0:
		filterbank_y3=filterbank_y3/tf.clip_by_value(sum_over_b,0.1,1)
	
	if index==0:
		modified_feeder=feed_in
		modified_eimage=e_image
		modified_feeder=tf.reshape(modified_feeder,(100,dim,dim))# reshped input 100*32*32
		modified_eimage=tf.reshape(modified_eimage,(100,dim,dim))
		filterbanky_to_feedin=tf.matmul(filterbank_y3,modified_feeder)
		filterbanky_to_feedin_to_filterbankx=tf.matmul(filterbanky_to_feedin,tf.transpose(filterbank_x3,perm=[0,2,1])) # dim0->dim0 dim1->dim2 dim2->dim1
		filterbanky_to_eimage=tf.matmul(filterbank_y3,modified_eimage)
		filterbanky_to_eimage_to_filterbankx=tf.matmul(filterbanky_to_eimage,tf.transpose(filterbank_x3,perm=[0,2,1])) # dim0->dim0 dim1->dim2 dim2->dim1
		intensity=tf.exp(log_int)
		intensity=tf.reshape(intensity,(100,1,1))
		intensity_filterbankx=intensity*filterbanky_to_feedin_to_filterbankx
		intensity_filterbankeimage=tf.multiply(intensity,filterbanky_to_eimage_to_filterbankx)
		intensity_filterbankx=tf.reshape(intensity_filterbankx,(100,N*N))
		intensity_filterbankeimage=tf.reshape(intensity_filterbankeimage,(100,N*N))
		read_op=tf.concat([intensity_filterbankx,intensity_filterbankeimage],1)
		#print("read %s "% read_op)
		return read_op,sum_over_a,x_center,mean_x_p[0]
		#write
	else:
		writing_patch=tf.matmul(new_out, tf.Variable(tf.random_normal([num_units,N*N],stddev=0.1),dtype=tf.float32,name="att_w%s" %t))\
		+tf.Variable(tf.zeros([N*N]),dtype=tf.float32,name="att_b%s"%t) #100 * 1024
		writing_patch=tf.reshape(writing_patch,(100,N,N))
		reversed_intensity=1.0/tf.exp(log_int)
		reversed_intensity=tf.reshape(reversed_intensity,(100,1,1))
		filterbanky_to_feedin=tf.matmul(tf.transpose(filterbank_y3,perm=[0,2,1]),writing_patch)#100*32*5  * 100*5*5->100*32*5
		filterbanky_to_feedin_to_filterbankx=tf.matmul(filterbanky_to_feedin,filterbank_x3) # 100 *32*32
		reversed_intensity_filterbankx=tf.multiply(reversed_intensity,filterbanky_to_feedin_to_filterbankx)
		reversed_intensity_filterbankx=tf.reshape(reversed_intensity_filterbankx,(100,dim*dim))
		if t==0:
			print("write %s"%reversed_intensity_filterbankx)
		return reversed_intensity_filterbankx





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
  if index==0:
    for image in tqdm (os.listdir('C:/Users/yaseralwattar/Desktop/faces/lfwcrop_grey/lfwcrop_grey/small faces')):
      path=os.path.join('C:/Users/yaseralwattar/Desktop/faces/lfwcrop_grey/lfwcrop_grey/small faces',image)
      image_reader=cv2.imread(path,0)     
      modified_image=cv2.resize(image_reader,(dim,dim))   
      image_list.append([np.array(modified_image)])
  else:

    for image in tqdm (os.listdir('C:/Users/yaseralwattar/Desktop/faces/lfwcrop_grey/lfwcrop_grey/smallest faces')):
      path=os.path.join('C:/Users/yaseralwattar/Desktop/faces/lfwcrop_grey/lfwcrop_grey/smallest faces',image)
      image_reader=cv2.imread(path,0)
      
      modified_image=cv2.resize(image_reader,(dim,dim))
    
      image_list.append([np.array(modified_image)])
  # if index==0:
  #   for image in tqdm (os.listdir('C:/Users/yaseralwattar/Desktop/cat faces/small clipped faces')):
  #     path=os.path.join('C:/Users/yaseralwattar/Desktop/cat faces/small clipped faces',image)
  #     image_reader=cv2.imread(path,0)     
  #     modified_image=cv2.resize(image_reader,(dim,dim))   
  #     image_list.append([np.array(image_reader)])
  # else:

  #   for image in tqdm (os.listdir('C:/Users/yaseralwattar/Desktop/cat faces/clipped faces')):
  #     path=os.path.join('C:/Users/yaseralwattar/Desktop/cat faces/clipped faces',image)
  #     image_reader=cv2.imread(path,0)
      
  #     modified_image=cv2.resize(image_reader,(dim,dim))
    
  #     image_list.append([np.array(image_reader)])
  
  
  shuffle(image_list)
  
  image_list=np.array([i[0] for i in image_list]).reshape(-1,dim*dim)

  #image_list=np.array([np.array(i) for i in image_list]).reshape(-1,A*B).astype(np.float32)
  #image_list=np.stack(image_list).reshape(-1,A*B)
  

  image_list=image_list.astype(np.float32)/255

  cv2.destroyAllWindows()
  np.save('image_matrix.npy',image_list)
  return image_list
max_train=300

feed_in=tf.placeholder(tf.float32,shape=(b,dim*dim))
ts=np.arange(0,10,1,np.int32)
num_units=256
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
  #read[t]=tf.concat([feed_in,e_image[t]],1)
  if t==0:
  	read[t],sum_over_a,x_center,mean_x_p0=spatial_attention(h_out0,5,0,t,feed_in,e_image[t])
  else:
  	read[t],sum_over_a,x_center,mean_x_p0=spatial_attention(new_out[t-1],5,0,t,feed_in,e_image[t])
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
  	canvas[t]=tf.add(tf.zeros([b,dim*dim],dtype=tf.float32,name="canvas0"),\
  	spatial_attention(new_out[t],5,1,t,feed_in,e_image[t]))
  else:
  	canvas[t]=tf.add(canvas[t-1],spatial_attention(new_out[t],5,1,t,feed_in,e_image[t]))


  # if t==0:
  #   canvas[t]=tf.add(tf.zeros([b,dim*dim],dtype=tf.float32,name="canvas0"),tf.matmul(new_out[t],\
  #   tf.get_variable(name="weightc%s"%t,shape=[num_units,dim*dim],dtype=tf.float32,\
  #   initializer=tf.contrib.layers.xavier_initializer())+tf.Variable(tf.zeros([dim*dim]),\
  #   dtype=tf.float32,name="biasc%s"%t)))

  # else:
  #   canvas[t]=tf.add(canvas[t-1],tf.matmul(new_out[t],\
  #   tf.get_variable(name="weightc%s"%t,shape=[num_units,dim*dim],dtype=tf.float32,\
  #   initializer=tf.contrib.layers.xavier_initializer())+tf.Variable(tf.zeros([dim*dim]),\
  #   dtype=tf.float32,name="biasc%s"%t)))
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

