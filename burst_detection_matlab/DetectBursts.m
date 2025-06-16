%%% script to run burst detection on NARCO data
clear
clc
close all

DataFolder = 'D:\Data\ArthurNarcolepsy\SET';
DestinationFolder = 'D:\Data\ArthurNarcolepsy\DetectedBursts';
RerunAnalysis = false;
RunParallelBurstDetection = true; % true for faster processing

%%% criteria to find bursts in single channels
% irregular shaped bursts, few criteria, but needs more cycles. This is
% criteria set 3 in the paper. (I don't change the order now in the script,
% because it would inaccurately refect the indexing saved in the detected
% bursts).
Idx = 1; % this is to make it easier to skip some
CriteriaSets = struct();
CriteriaSets(Idx).PeriodConsistency = .6;
CriteriaSets(Idx).AmplitudeConsistency = .6;
CriteriaSets(Idx).MonotonicityInAmplitude = .6;
CriteriaSets(Idx).FlankConsistency = .6;
CriteriaSets(Idx).MinCyclesPerBurst = 5;
% % without periodneg, to capture bursts that accelerate/decelerate

% short bursts, strict monotonicity requirements. This is criteria set 2.
Idx = Idx+1;
CriteriaSets(Idx).PeriodNeg = true;
CriteriaSets(Idx).PeriodConsistency = .7;
CriteriaSets(Idx).FlankConsistency = .3;
CriteriaSets(Idx).MonotonicityInAmplitude = .9;
CriteriaSets(Idx).MinCyclesPerBurst = 3;

% relies on shape but low other criteria; gets most of the bursts (in the
% paper, this is referred to as criteria set 1, because it uses the most
% criteria, so was good for an initial explanation).
Idx = Idx+1;
CriteriaSets(Idx).PeriodNeg = true;
CriteriaSets(Idx).PeriodConsistency = .5;
CriteriaSets(Idx).AmplitudeConsistency = .4;
CriteriaSets(Idx).FlankConsistency = .5;
CriteriaSets(Idx).ShapeConsistency = .2;
CriteriaSets(Idx).MonotonicityInTime = .4;
CriteriaSets(Idx).MonotonicityInAmplitude = .4;
CriteriaSets(Idx).ReversalRatio = .6;
CriteriaSets(Idx).MinCyclesPerBurst = 4;

MinClusteringFrequencyRange = 1; % to cluster bursts across channels

Bands.ThetaLow = [2 6];
Bands.Theta = [4 8];
Bands.ThetaAlpha = [6 10];
Bands.Alpha = [8 12];


Files = deblank(string(ls(DataFolder)));
Files(~contains(Files, '.set')) = [];

if ~exist(DestinationFolder, 'dir')
    mkdir(DestinationFolder)
end

for FileIdx = 1:numel(Files)
    File = Files{FileIdx};
    DestinationFileCSV = [extractBefore(File, '.set'), '.csv'];

    if ~RerunAnalysis && exist(fullfile(DestinationFolder, DestinationFileCSV), 'file')
        disp(['already did ', File])
        continue
    end

    EEG = pop_loadset('filename', File, 'filepath', DataFolder);
    SampleRate = EEG.srate;
    Chanlocs = EEG.chanlocs;

    % filter data
    EEG = pop_eegfiltnew(EEG, 1);
    EEG = pop_eegfiltnew(EEG,  [], 40);

    % any timepoints marked as bad?
    KeepTimepoints = ones(1, size(EEG.data, 2));

    % filter data into narrowbands
    EEGNarrowbands = cycy.filter_eeg_narrowbands(EEG, Bands);

    %%
    % apply burst detection
    Bursts = cycy.detect_bursts_all_channels(EEG, EEGNarrowbands, Bands, ...
        CriteriaSets, RunParallelBurstDetection, KeepTimepoints);

    % aggregate bursts into clusters across channels (not really used)
    BurstClusters = cycy.aggregate_bursts_into_clusters(Bursts, EEG, MinClusteringFrequencyRange);

    % remove from Bursts all bursts that didn't make it into a cluster (means it was only in one channel)
    ClusteredBurstIndexes = unique([BurstClusters.ClusterBurstsIdx]);
    Bursts = Bursts(ClusteredBurstIndexes);

    KeepPoints = nnz(KeepTimepoints); % only if there's artifact data

    BurstsRedux = rmfield(Bursts, {'...'});
    BurstClustersRedux = rmfield(BurstClusters, {''});

    BurstsTable = struct2table(BurstsRedux);
    BurstClustersRedux = struct2table(BurstClustersRedux);
    % cycy.plot.plot_all_bursts(EEG, 20, BurstClusters, 'Band');

    save(fullfile(DestinationFolder, DestinationFileMAT), 'BurstClusters', 'Bursts', 'KeepPoints', 'Chanlocs', 'SampleRate')
    writetable(BurstsTable, fullfile(DestinationFolder, ['Bursts_', DestinationFileCSV]))
    writetable(BurstClustersTable, fullfile(DestinationFolder, ['BurstCluster_', DestinationFileCSV]))
end
