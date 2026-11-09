%% SIMULATION STUDY: loading recovery with TASD- and TPCA-initialized CP-ALS
clc;
clear;

%% Local paths
codeDir = fileparts(mfilename('fullpath'));
addpath(codeDir);
addpath(fullfile(codeDir, 'tensor_toolbox-v3.8'));

%% General parameters
config = struct;
config.rngSeed = 42;
config.Rtrue = 3;
config.p = [22 22 84 15];
config.lambdaBase = [5 3 1];
config.M = 100;                    % Independent simulated datasets
config.nStarts = 3;                % Repeated starts on each fixed dataset

%% ALS stopping rule and loading recovery threshold
config.alsTol = 1e-6;
config.maxit = 500;
config.recoveryTol = 0.05;

%% Simulation designs
config.baseXi = 0.2;
config.baseSigma = 1e-2;
config.sigmaGrid = [1e-9 1e-7 1e-5 1e-4 1e-2 1];
config.xiGrid = linspace(0, 0.9, 10);
config.RfitGrid = [config.Rtrue - 1, config.Rtrue, config.Rtrue + 1];

% Distinguish a uniformly weaker signal from one particularly weak component.
weakestLambda = config.lambdaBase;
weakestLambda(end) = 0.25 * weakestLambda(end);
config.strengthLambda = [ ...
    config.lambdaBase; ...
    0.25 * config.lambdaBase; ...
    weakestLambda];
config.strengthLabels = ["strong", "all weak", "one weak"];

%% Run, save and plot
% Runs are serial so elapsed times are not affected by competing workers.
% The runner saves a checkpoint after every completed simulation condition.
config.outputDir = fullfile(codeDir, 'Results_v3');
results = run_tensor_simulations(config);
plot_tensor_simulations(results, fullfile(config.outputDir, 'figures'));

disp(results.summaryTable);
