addpath(genpath('utils'))

clear all;
close all;
clc
warning('off');

dataset_name = 'abu-airport-2';
file_path = 'datasets/';
load(join([file_path, dataset_name]));

