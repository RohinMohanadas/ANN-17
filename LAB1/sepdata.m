%function [ patterns, targets ] = sepdata()
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

classA(1,:) = randn(1,100) .* 0.5 + 1.0;
classA(2,:) = randn(1,100) .* 0.5 + 0.5;
classB(1,:) = randn(1,100) .* 0.5 - 1.0;
classB(2,:) = randn(1,100) .* 0.5 + 0.0;

patterns = [classA, classB];

targets = [ones(1,100) ones(1,100).*-1];

permute = randperm(200);
patterns = patterns(:, permute);
targets = targets(:, permute);

plot (patterns(1, find(targets>0)), ...
patterns(2, find(targets>0)), '*', ...
patterns(1, find(targets<0)), ...
patterns(2, find(targets<0)), '+');
%end

