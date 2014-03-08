function w = new_weights(input,err,learning_rate,weight)
% updates the weights

w = err*input'*learning_rate + weight;
end