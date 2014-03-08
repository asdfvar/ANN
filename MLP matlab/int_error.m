function di = int_error(output,weight,forward_error)
% delta terms for back propagation. not actual error

temp = weight'*forward_error;
temp = temp.*(1-output);
di = temp.*output;
% di = weight'*forward_error.*(1-output).*output;
end
