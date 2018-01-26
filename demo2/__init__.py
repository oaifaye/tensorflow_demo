#-*-coding:UTF-8-*-
import tensorflow as tf

# 使用tensorflow在默认的图中创建节点，这个节点是一个变量。
state = tf.Variable(0)
print(state)
# state = tf.Variable(0 , name='counter')

# 此处调用了td的一个函数，用于创建常量。
one = tf.constant(1)

# 对常量与变量进行简单的加法操作，这点需要说明的是： 在TensoorFlow中，所有的操作op，变量都视为节点，tf.add() 的意思就是在tf的默认图中添加一个op，这个op是用来做加法操作的。
new_value = tf.add(state,one)

# 这个操作是：赋值操作。将new_value的值赋值给update变量。
update = tf.assign(state,new_value)

# 好了，到此为止。我们的“图flow”构建好了。
# 在这里，需要再次说明：我们此时只是定义好了图，并没有变量并没有初始化。目前只有state的值是1。

# 此处用于初始化变量。但是这句话仍然不会立即执行。需要通过sess来将数据流动起来 。
init = tf.initialize_all_variables()

# 切记：所有的运算都应在在session中进行：
# 此处自动开启一个session
with tf.Session() as sess:
    # 对变量进行初始化，执行（run）init语句
    sess.run(init)
    sess.run(state)
    for _ in range(3): 
        sess.run(update) 
        print(sess.run(state))
    print('ss:',sess.run(new_value))