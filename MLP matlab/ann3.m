function [weight_1_out,weight_2_out,weight_3_out,output,err] = ann3(input,weight_1,weight_2,weight_3,target_output,learning_rate)
% artificial neural network
% be sure to include an extra dimension for the codomain of the first input
% weights for the bias node. the other layers will take care of this
% automatically but be sure the cardinality of the matrices agree for each
% input/output. multiple node layers must be hard coded. default number of
% hidden layers is whatever i have it set to now.

input = [-1;input];

n = size(target_output);
n = n(2);

out1 = layer(input,weight_1);
out1 = sigmoid(out1,1);
out2 = layer(out1,weight_2);
out2 = sigmoid(out2,1);
out3 = layer(out2,weight_3);
out3 = sigmoid(out3,1);
output = out3;

% back propagation
err = norm(target_output - out3)/sqrt(n);
dout = out_error(target_output,out3);
dC = int_error(out2,weight_3,dout);
dB = int_error(out1,weight_2,dC);

% weight adjustment
weight_3_out = new_weights(out2,dout,learning_rate,weight_3);
weight_2_out = new_weights(out1,dC,learning_rate,weight_2);
weight_1_out = new_weights(input,dB,learning_rate,weight_1);
end
