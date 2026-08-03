"""Write down the observations from the plot for all the above functions in the code.
1.What are the min and max values for the functions?
ans: for the sigmoid function:
     the maximum value  1
     the minimum value  0
    2.)for tanh function:
     the maximum value +1
     the minimum value -1
    3.)for ReLU function:
     the maximum value infinity
     the minimum value 0
    4.)leaky relu function:
     the maximum value infinity
     the minimum value- negative infinity
    5.)Softmax function:
     the maximum value  1
     the minimum value  0

--------------------------------------------------------------------------------------------------------------------------------

2.Is the output of the function zero-centred?
An activation function provides a zero-centered output if its mathematical range is perfectly symmetrical around zero.
You find this by checking if the function is an odd function, meaning it satisfies:
f(-x)=-f(x)
 ----so for sigmoid function:-----------------
 f(-x) is not equal to -f(x)
1/1+e^z is not equal to -1/1+e^-z

------tanh function---------------
f(-x) is equal to -f(x)
f(-x) = e^-x - e^x / e^x + e^-x
-f(x) = e^-x - e^x / e^x + e^-x
hence it is zero centered

---------relu function--------------
f(-x) is not equal to -f(x)
hence relu function is not zero centered.

-------------------------------------------------------------------------------------------------------------------------------------

3.What happens to the gradient when the input values are too small or too big?
ans: when the input is extremly big(positive), or extremly small(negative)function , so depending on the
     function it can either vanishes i.e zero or might go to infinity.
------------------------------------------------------------------------------------------------------------------------------

4.What is the relationship between sigmoid and tanh?
 """

