import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import numpy as np

x = tf.Variable(34,name='x')
y = tf.Variable(12,name='y')
z = tf.Variable(45,name='z')
f = pow(x,3)+pow(x,2)*y+y*z

a1 = tf.constant(3)
a2 = a1 + 12
a3 = a2 - 8

m = tf.ones((20,5))
n = tf.zeros((5,20))
k = tf.ones((5,5))
y = tf.matmul(n,m)+k

arr = np.array([[2.0,4.0,6.6],[6.6,8.2,1.1],[7.4,4.1,2.9]])
v = tf.matrix_determinant(arr)

inv_mat = tf.matrix_inverse(arr)

trans = tf.matrix_transpose(inv_mat)

diag = tf.matrix_diag(inv_mat)

dis_1 = tf.argmax([3,4,20,15],name='dis')

dis_2 = tf.argmin([4,3,20,15],name='dis')

arg = tf.argmin(arr,axis=0)

a = tf.random_uniform((90,),minval=0,maxval=1)
b = tf.random_uniform((90,),minval=0,maxval=1)
pred = tf.round(a,'pred')
y_true = tf.round(b,'y_true')
confusion_mat = tf.confusion_matrix(pred, y_true, name='confusion_Matrix')

A = tf.placeholder(tf.float32, shape=(None,3))
B = A + 5

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    init.run()
    result = f.eval()
    print('Operations with basic tensorflow variables')
    print(result)

    a = a3.eval()
    print('Operations with basic tensorflow constant')
    print(a)

    matrix = y.eval()
    print('Operations with basic tensorflow matrices')
    print(matrix)

    out = v.eval()
    print('Matrix determinant')
    print(out)

    inv = inv_mat.eval()
    print('Matrix inverse')
    print(inv)

    trans_n = trans.eval()
    print('Transpose of inv matrix')
    print(trans)

    diagonal = diag.eval()
    print('Diagonal matrix')
    print(diagonal)

    print('index of maximum of a few numbers')
    print(dis_1.eval())

    print('index of minimum of a few numbers')
    print(dis_2.eval())

    print('index maximum along the each column of arr')
    print(arg.eval())

    print('Confusion matrix')
    print(confusion_mat.eval())

    b_val = B.eval(feed_dict={A:[[1,2,3]]})
    b_val_u = B.eval(feed_dict= {A:[[4,5,6],[7,8,9]]})
    print('Placeholders')
    print(b_val)
    print(b_val_u)
    