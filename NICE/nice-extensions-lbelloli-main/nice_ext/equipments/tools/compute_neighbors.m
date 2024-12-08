% Copyright (C) Federico Raimondo - All Rights Reserved
% Unauthorized copying of this file, via any medium is strictly prohibited
% Proprietary and confidential
% Written by Federico Raimondo <federaimondo@gmail.com>, October 2017

cfg = {};
cfg.layout = 'EEG1005.lay';

layout = ft_prepare_layout(cfg);

% names = {'Fp1'; 'Fpz'; 'Fp2'; 'AF1'; 'AFz'; 'AF2'; 'F7'; 'F5'; 'F1'; 'Fz'; 'F2';
% 'F6'; 'F8'; 'FT9'; 'FT7'; 'FC5'; 'FC3'; 'FC1'; 'FCz'; 'FC2'; 'FC4';
% 'FC6'; 'FT8'; 'FT10'; 'T7'; 'C5'; 'C3'; 'C1'; 'Cz'; 'C2'; 'C4'; 'C6';
% 'T8'; 'TP9'; 'TP7'; 'CP5'; 'CP3'; 'CP1'; 'CPz'; 'CP2'; 'CP4'; 'CP6';
% 'TP8'; 'TP10'; 'P9'; 'P7'; 'P3'; 'P1'; 'Pz'; 'P2'; 'P4'; 'P8'; 'P10';
% 'PO3'; 'POz'; 'PO4'; 'O1'; 'Oz'; 'O2'; 'Iz'};

% names = {'C3'; 'C4'; 'O1'; 'O2'; 'A1'; 'A2'; 'Cz'; 'F3'; 'F4'; 'F7';
%            'F8'; 'Fz'; 'Fp1'; 'Fp2'; 'P3'; 'P4'; 'Pz'; 'T7'; 'T8';
%            'P7'; 'P8'}

names = {
        'Fp1'; 'Fpz'; 'Fp2'; 'AF7'; 'AF3'; 'AFz'; 'AF4'; 'AF8'; 'F7'; 'F5',
        'F3'; 'F1'; 'Fz'; 'F2'; 'F4'; 'F6'; 'F8'; 'FT9'; 'FT7'; 'FC5'; 'FC3',
        'FC1'; 'FCz'; 'FC2'; 'FC4'; 'FC6'; 'FT8'; 'FT10'; 'T7'; 'C5'; 'C3',
        'C1'; 'Cz'; 'C2'; 'C4'; 'C6'; 'T8'; 'TP7'; 'CP5'; 'CP3'; 'CP1'; 'CP2',
        'CP4'; 'CP6'; 'TP8'; 'P9'; 'P7'; 'P5'; 'P3'; 'P1'; 'Pz'; 'P2'; 'P4',
        'P6'; 'P8'; 'P10'; 'PO9'; 'PO7'; 'PO3'; 'POz'; 'PO4'; 'PO8'; 'PO10',
        'O1'; 'O2'; 'I1'; 'Iz'; 'I2'; 'AFp3h'; 'AFp4h'; 'AFF5h'; 'AFF6h',
        'FFT7h'; 'FFC5h'; 'FFC3h'; 'FFC1h'; 'FFC2h'; 'FFC4h'; 'FFC6h'; 'FFT8h',
        'FTT9h'; 'FTT7h'; 'FCC5h'; 'FCC3h'; 'FCC1h'; 'FCC2h'; 'FCC4h'; 'FCC6h',
        'FTT8h'; 'FTT10h'; 'TTP7h'; 'CCP5h'; 'CCP3h'; 'CCP1h'; 'CCP2h',
        'CCP4h'; 'CCP6h'; 'TTP8h'; 'TPP9h'; 'TPP7h'; 'CPP5h'; 'CPP3h'; 'CPP1h',
        'CPP2h'; 'CPP4h'; 'CPP6h'; 'TPP8h'; 'TPP10h'; 'PPO9h'; 'PPO5h',
        'PPO6h'; 'PPO10h'; 'POO9h'; 'POO3h'; 'POO4h'; 'POO10h'; 'OI1h'; 'OI2h',
        'AFF1'; 'AFF2'; 'PPO1'; 'PPO2'; 'M1'; 'M2'}

fname = '../data/ant_124_neighbours.mat'

% T3 is T7; T4 is T8

cfg = {};
cfg.method = 'triangulation';
cfg.layout = 'EEG1005.lay';
cfg.channel = names;
cfg.feedback = 'yes';
neighbours = ft_prepare_neighbours(cfg)
save(fname, 'neighbours')
