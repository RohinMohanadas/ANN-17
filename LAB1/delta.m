%function [ output_args ] = Untitled( input_args )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
close all
clear all
clc

nu = 0.005;
epochs = 20;

W = randn(1,3).*0.01;

sepdata; 

[insize, ndata] = size(patterns);
[outsize, ndata] = size(targets);

X = [patterns; ones(1,200)];

T = targets;

for i = 1:epochs
    
    dW = -1 .* nu .* (W*X - T) * X';
    
    
    
    p = W(1,1:2);
    k = -W(1, insize+1) / (p*p');
    l = sqrt(p*p');
    plot (patterns(1, find(targets>0)), ...
        patterns(2, find(targets>0)), '*', ...
        patterns(1, find(targets<0)), ...
        patterns(2, find(targets<0)), '+', ...
        [p(1), p(1)]*k + [-p(2), p(2)]/l, ...
        [p(2), p(2)]*k + [p(1), -p(1)]/l, '-');
        axis ([-2, 2, -2, 2], 'square');
    drawnow;
    
    
    W = W + dW;
    pause(0.2);
end

    


%end

