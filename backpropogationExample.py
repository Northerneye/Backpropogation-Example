import numpy as np
training_rate = .1 #How fast or slow we want this to train
input_node = .5 #Value of the input
weight1 = .5 #Value of the weight from the first layer
hidden_node = 0 #the hidden node is initialized to zero
weight2 = .5 #Value of the weight from the second layer
output_node = 0 #Output is initially set to zero
expected_output = 1 #This is the expected value that should be obtained from the network
for j in range(1000):
    hidden_node = input_node*weight1
    output_node = hidden_node*weight2 #runs the network
    loss = output_node-expected_output #finds the error of the prediction
    weight2holder = weight2-loss*weight1*training_rate #updates the second weight
    loss2 = weight2holder-weight2 #finds the loss of the second weight
    weight2 = weight2holder
    weight1 = weight1-loss2*weight1*training_rate #updates the first weight
    
hidden_node = input_node*weight1 #runs the output again after updating weights
output_node = hidden_node*weight2
print("weight1: ",weight1)
print("weight2: ",weight2)
print("output: ",output_node)
print("loss: ",loss)


#       | output |          
#           ||         
#      | * weight2 |         (multiply hidden node by weight 2 to find output)
#           ||         
#     | Hidden node |               
#           ||         
#      | * weight1 |        (multiply input times weight1 to get hidden node) 
#           ||             
#        | Input |            
