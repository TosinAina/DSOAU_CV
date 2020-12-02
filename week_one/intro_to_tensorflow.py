import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

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

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    init.run()
    result = f.eval()
    print(result)

    a = a3.eval()
    print(a)

    matrix = y.eval()
    print(matrix)