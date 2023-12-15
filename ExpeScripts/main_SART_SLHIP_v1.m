%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Project SLHIP - SART + Probes
%%%%%
%%%%%
%%%%% Written by Thomas Andrillon
%%%%% Email: thomas.andrillon@gmail.com
%%%%% 
%%%%%   Adapted by Arthur Le Coz
%%%%% v1:
%%%%%   - Changing questions : ON, MW, Disturbed, Hallu, MB, Forgot
%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% init matlab
% clear all variables and close all figures
clear all;
close all;
% set rand function on the clock
rand('seed',sum(100*clock));
trai_res=[];
test_res=[];
probe_res=[];

% add the folder in the path
if ismac
    root_path='/Users/arthurlecoz/Desktop/A_Thesis/2023/Expe/SLHIP/SART/ExpeScripts';
    PTB_path='/Applications/Psychtoolbox/';
else
    root_path='C:\manips\SLHIP\ExpeFolder\ExpeScripts';
    PTB_path='C:\Toolbox\Psychtoolbox';
end
cd(root_path)
addpath(pwd)
% add PTB in the path
if exist('Screen')~=3
    addpath(genpath(PTB_path));
    fprintf('... adding PTB to the path\n')
end
all_GrandStart=GetSecs;

stim_path=[root_path filesep '..' filesep 'ExpeStim'];

% Select debug mode (1 or 0), EEG (1 or 0), Tobii (1 or 0)
flag_smallw     = 1;
answerdebug=input('Entrez 0 (non) ou 1 (oui, debug)');
flag_debug         = answerdebug;
flag_EEG           = 1; % EEG segments to be added in the future
flag_PPort         = 1; % Set to 1 if there is a parallel port to send triggers
flag_EyeLink       = 1;
flag_skiptraining  = 0;
flag_skipbaseline  = 1;
flag_2diodes       = 0;
flag_1diodes       = 0;
flag_bip           = 0;
flag_escp          = 0;

supervised_questions = {...
    'Choisissez une des options suivantes décrivant le mieux\n\n votre etat d''esprit juste avant l''interruption.\n\n\n - J''étais concentré-e sur la tâche : Appuyez sur 1\n\n - J''étais distrait·e par quelque chose d''interne (pensée, image, etc.) : Appuyez sur 2\n\n - J''étais distrait·e par quelque chose d''externe (environnement) : Appuyez sur 3\n\n - J''étais distrait·e par quelque chose de fictif (illusion, hallucination) : Appuyez sur 4\n\n -  - Je ne pensais à rien : Appuyez sur 5\n\n - Je ne me souviens pas : Appuyez sur 6',...
    'Avez-vous volontairement contrôlé\nsur quoi se portait votre attention ?\n\n Oui : Appuyez sur 1\n\n Non : Appuyez sur 2',...
    ...
    'Notez votre vigilance:\n\n\n - Extremement alerte : Appuyez sur 1\n\n - Très alerte : Appuyez sur 2\n\n - Alerte : Appuyez sur 3\n\n - Plutôt alerte : Appuyez sur 4\n\n - Ni alerte, ni fatigué·e : Appuyez sur 5\n\n - Quelques signes de fatigue : Appuyez sur 6\n\n - Fatigué-e, mais aucun effort\n\npour rester eveillé-e : Appuyez sur 7\n\n - Fatigué-e, des efforts pour rester eveillé-e : Appuyez sur 8\n\n - Tres fatigué-e, de gros efforts pour rester eveillé-e,\n\nluttant contre le sommeil : Appuyez sur 9'
    };

supervised_questions_headers={'Juste avant l''interruption',...
    'Juste avant l''interruption',...
    'Au cours des précédents essais'};

supervised_questions_acceptedanswers={[1:6],[1 2],[1:9]};

%% Enter subject info
if flag_debug
    subject         = 'debug';  % get subjects name
    subjectID       = '000';    % get subjects seed for random number generator
    sessionID       = 1;      % get session number
    expstart        = datestr(now,'ddmmmyyyy-HHMM');   % string with current date
    subjectGender   = NaN;
    subjectAge      = 'X';
    flicker_freqL   = 12;  % in Hertz of backgroud
    flicker_freqR   = 15;  % of box (try 16 or 22.4)
else
    subject         = input('Code Sujet:','s');      % get subjects name
    subjectID       = input('n° du sujet (ex 048):','s');       % get subjects seed for random number generator
    subjectGender   = input('Genre (H, F, A):','s');
    subjectAge      = input('Age (ex 23):','s');
    sessionID       = 1; %str2num(answerdlg{3});      % get session number
    expstart        = datestr(now,'ddmmmyyyy-HHMM');   % string with current date
    
    numSub=str2num(subjectID);
end
SubjectInfo.sub         = subject;
SubjectInfo.subID       = subjectID;
SubjectInfo.sessID      = sessionID;
SubjectInfo.Age         = subjectAge;
SubjectInfo.Gender      = subjectGender;
SubjectInfo.Date        = expstart;
SubjectInfo.FlagW       = flag_smallw;
SubjectInfo.FlagEEG     = flag_EEG;
SubjectInfo.FlagTobii   = flag_EyeLink;

%% EEG
% if flag_EEG
%     % Check that the MEX file io64 is in the path
%     if flag_PPort
%         addpath('eeg')
%         
%         object = io64;  %% initialize driver of parallel port
%         status = io64(object);  %% check status of driver (should be open now)
%         %%%  should say: 64-bit Windows
%         %%% 'InpOut32a driver is open'
%         %%% [status = 0]
%         port= hex2dec('2FD8');% Parallel port's address in computer // C465
%         %%% port= hex2dec('0378');  %% 0378 / D010 ;address of the parallel port (check under devices/Resources Setting)
%         data = io64(object,port);  %% read out data from port
%         data_in=io64(object,port); %% read out in matlab agai
%     end
%     answerdebug2=input('Entrez 1 si l''EEG enregistre (0 pour annuler):');
%     if answerdebug2==0
%         flag_escp=1;
%     else
%         WaitSecs(3);
%         
%         if flag_PPort
%             % Code for triggers (must be between 0 and 255)
%             %start/end recording
%             trig_start          =1; %S
%             trig_end            =2; %E
%             trig_startBlock     =3; %B
%             trig_endBlock       =4; %K
%             trig_startTrial     =5; %T
%             trig_startQuestion  =6; %Q
%             trig_probestart     =7; %P
%             trig_probeend       =8; %C
%             
%             trig_reset      =0;
%             % Send a first trigger
%             io64(object,port,trig_start); 
%             all_StartRecord=GetSecs;
%             WaitSecs(0.004);%% sends event code
%             fprintf('>>>>>> Vérifiez que le trigger a été envoyé!!!\n');
%             io64(object,port,0); %% set port to zero
%             
% %             outputSingleScan(s,trig_start);
% %             fprintf('>>>>>> CHECK START TRIGGER HAS BEEN SENT\n');
% %             io64(object,port,0); %% set port to zero
% %             WaitSecs(1);
% %             outputSingleScan(s,trig_reset);
%         end
%     end
% else
%     fprintf('>>>>>> EEG system won''t be used\n');
% end
% WaitSecs(1);

if flag_EEG
    % Check that the MEX file io64 is in the path
   OpenParPort( 64 );
    answerdebug2=input('Press 1 if EEG is recording (0 to abort):');
    if answerdebug2==0
        flag_escp=1;
    else
        WaitSecs(3);
%         SendTrigger( marqueurMessage );

        if flag_PPort
            % Code for triggers (must be between 0 and 255)
            %start/end recording
            trig_start          = 1; %S
            trig_end            = 11; %E
            trig_startBlock     = 2; %B
            trig_endBlock       = 22; %K
            trig_startTrial     = 64; %T
            trig_startQuestion  = 128; %Q
            trig_probestart     = 3; %P
            trig_probeend       = 33; %C
            trig_response       = 5; %C
            
            % Send a first trigger
            SendTrigger(trig_start);
            fprintf('>>>>>> CHECK START TRIGGER HAS BEEN SENT\n');
            WaitSecs(1);
        end
    end
else
    fprintf('>>>>>> EEG system won''t be used\n');
end
WaitSecs(1);

%% Audio
if flag_bip && flag_escp==0
    audiocheck=0;
    InitializePsychSound;
    freq = 44100;
    while audiocheck==0
        pahandle = PsychPortAudio('Open', [], [], 0, freq, 1);
        [beep,samplingRate] = MakeBeep(440,0.5,freq);
        PsychPortAudio('FillBuffer', pahandle, beep);
        PsychPortAudio('Start', pahandle, 1, 0, 1);
        answerdebug2=input('Press 1 if sound is playing (2, to retry, 0 to abort):');
        if answerdebug2==0
            flag_escp=1;
            audiocheck=1;
        elseif answerdebug2==1
            audiocheck=1;
        elseif answerdebug2==2
            audiocheck=0;
        end
        PsychPortAudio('Stop', pahandle);
        PsychPortAudio('Close', pahandle);
    end
end

%% init PTB
screenNumbers = Screen('Screens');
numscreen = max(screenNumbers);
% set up screen display
if flag_smallw
    Prop=12;
    %     w = Screen('OpenWindow', numscreen, 0, [0, 0, 1920*(Prop-1)/Prop, 1080*(Prop-1)/Prop]+[1920 1080 1920 1080]*1/Prop/2);
    Screen('Preference', 'SkipSyncTests', 1);
    w = Screen('OpenWindow', numscreen, 0, [0, 0, 1200, 700]);
    InstrFont=30;
else
    w = Screen('OpenWindow', 2, 0, []);
    InstrFont=30;
    HideCursor;
end
Screen('TextSize',w, InstrFont);
ifi = Screen('GetFlipInterval',w,100);
[wx, wy] = Screen('WindowSize', w);
vbl = Screen('Flip', w);
Screen('BlendFunction', w, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
Screen('FillRect',  w, 0);
Screen('Flip',w);

% Parameters drawing
cross_fixCrossDimPix = 16; % Arm width
% set parameters for fixation cross
cross_lineWidthPix = 5;
cross_xCoords = [-cross_fixCrossDimPix cross_fixCrossDimPix 0 0];
cross_yCoords = [0 0 -cross_fixCrossDimPix cross_fixCrossDimPix];
cross_colour = [255 0 0]; %red

if flag_2diodes || flag_1diodes
    squarewidth=80; %wy/50*2; % screen is rougly 50cm and we want a 2cm-width square
    startpos=[wx-squarewidth 0];
    din2_pos=repmat(startpos,5,1)+[0 0 ; squarewidth 0; squarewidth squarewidth; 0 squarewidth; 0 0];
    startpos=[wx-squarewidth wy-squarewidth];
    din1_pos=repmat(startpos,5,1)+[0 0 ; squarewidth 0; squarewidth squarewidth; 0 squarewidth; 0 0];
end

% Mask
% filename = [stim_path filesep 'Mask.jpg'];
% thispic=imread(filename);
% imgetex=Screen('MakeTexture', w, thispic);
% mask_index=imgetex;
% Rect
centerx = (wx/2);
centery = wy/2;
LeftRect=[centerx-0.2*wx, centery-0.2*wx, centerx, centery+0.2*wx];
RightRect=[centerx, centery-0.2*wx, centerx+0.2*wx, centery+0.2*wx];

dur_face_presentation=0.9;
dur_face_presentation_jitter=0.2;
TargetID=3;
letterSize=60;

KbIndexes=GetKeyboardIndices;
KbIndex=max(KbIndexes);
% % % % WaitSecs(1); [keyIsDown, secs, keyCode, deltaSecs] = KbCheck(-1); while ~keyIsDown; [keyIsDown, secs, keyCode, deltaSecs] = KbCheck(-1); end; find(keyCode)
% if length(KbIndexes)==1
if ismac
    myKeyMap=[30 31 32 33]; % for 1, 2, 3, 4
    AbortKey='Escape';
elseif IsWindows
    myKeyMap=[49:57]; % for 1, 2, 3, 4
    AbortKey='Esc';
else
    %     myKeyMap=[100 101 102 107]; % for 1, 2, 3, 4
    myKeyMap=[84 85 86 87]; % for 1, 2, 3, 4
    %     myKeyMap=[30 31 32 33]; % for 1, 2, 3, 4
    fprintf('!!!!! WARNING: only main keyboard recognized!!!!\nIn such case, unplug/plug USB cable and THEN quit and relaunch matlab\n')
end
% else
%     %     myKeyMap=[100 101 102 107];
%     myKeyMap=[84 85 86 87]; % for 1, 2, 3, 4
% end

%% EEG
WaitSecs(0.2);
if flag_EEG && flag_1diodes && flag_escp==0
    dinok=0;
    while dinok==0
        DrawFormattedText(w, 'checking sync...', 'center', 'center', [255 0 0]);
        Screen('FillPoly', w ,[1 1 1]*255, din2_pos);
        tempst=Screen('Flip',w);
        while GetSecs<tempst+0.2
            if dinok==0
                [keyIsDown, secs, keyCode, deltaSecs] = KbCheck(-1);
                if keyIsDown==1
                    dinok=1;
                end
            end
        end
        DrawFormattedText(w, 'checking sync...', 'center', 'center', [255 0 0]);
        tempst=Screen('Flip',w);
        while GetSecs<tempst+0.2
            if dinok==0
                [keyIsDown, secs, keyCode, deltaSecs] = KbCheck(-1);
                if keyIsDown==1
                    dinok=1;
                end
            end
        end
    end
end
DrawFormattedText(w, 'done!', 'center', 'center', [255 0 0]);
tempst=Screen('Flip',w);

%% Tobii
if flag_EyeLink && flag_escp==0
    if (Eyelink('initialize') ~= 0) %% to debug Eyelink('initializedummy')
        return;
    end
    
    %%%%%%%% EYELINK SETUP AND CALIBRATION
    window=w;
    el=EyelinkInitDefaults(window);%,par);
    el.backgroundcolour=255;
    
    if ~EyelinkInit(0, 1)
        fprintf('Eyelink Init aborted.\n');
        cleanup;  % cleanup function
        return;
    end
    % make sure that we get gaze data from the Eyelink
    Eyelink('Command', 'link_sample_data = LEFT,RIGHT,GAZE,AREA');
    
    % open file to record data to
    % Eyelink('Openfile', par.edfFile);
    saveEyeLink_eyet=sprintf('WIMs%s.edf',subjectID);
    Eyelink('OpenFile',saveEyeLink_eyet);
    % calibrate:
    EyelinkDoTrackerSetup(el); % Instructions come up on the screen. It seems Esc has to be pressed on the stim computer to exit at the end
    
    disp('FINISHED CALIBRATING')
    
    Eyelink('StartRecording');
    WaitSecs(0.1);
    % mark zero-plot time in data file
    Eyelink('Message', 'SYNCTIME');
    % figure out which eye is being tracked:
    eye_used = Eyelink('EyeAvailable');
    EyeLink_StartTime=GetSecs;
    SubjectInfo.EyeLink_StartTime=EyeLink_StartTime;
    
    nCalib=1;
    Eyelink('Message', sprintf('C_%g',nCalib));
    
    fprintf('>>>>>> EyeLink is up and running\n');
    Screen('FillRect',  w, 0);
    
else
    fprintf('>>>>>> EyeLink won''t be used\n');
end
Screen('Flip',w);
WaitSecs(1);
Screen('Flip',w);

%% BASELINE || TO DO
%if flag_skipbaseline==0 && flag_escp==0
%     Screen('TextSize',w, InstrFont);
%     DrawFormattedText(w, 'Nous allons maintenant enregistrer \nvotre activité cérébrale de repos\n\n  Appuyez sur une touche quand vous êtes prêt', 'center', 'center', [255 255 255]);    
%     Screen('Flip',w);
%     KbWait(-1);
%     KbReleaseWait(-1);
    
%     display_BaselineIM_v3;
% end
% Screen('Flip',w);
% WaitSecs(3);

%% TRAINING
if flag_escp==1
    Screen('TextSize',w, InstrFont);
    DrawFormattedText(w, 'Interruption de la session...', 'center', 'center', [255 255 255]);
    Screen('Flip',w);
    WaitSecs(1);
    Screen('Flip',w);
else
    ListenChar(2);
    trai_res=[];
    if flag_skiptraining==0
        if flag_PPort
            SendTrigger(trig_start);
        end
        
        KbReleaseWait(-1);
        Screen('TextSize',w, InstrFont);
        DrawFormattedText(w, 'Vous allez réaliser l''entraînement de la tâche.\n\n\nRappelez vous:\n\nAppuyez sur la barre espace apres tous les chiffres\n\nSAUF le chiffre 3\n\n\n\nAppuyez sur Espace quand vous etes prêt-e', 'center', 'center', [255 255 255]);
        Screen('Flip',w);
        KbWait(-1);
        KbReleaseWait(-1);
        Screen('Flip',w);
        this_blockcond=2;
        nblock=2;
        thiset=3;
        display_SART_training_SLHIP_v1;
        
        perfGO=100*(nanmean(trai_res(trai_res(:,1)==2,12)));
        perfNOGO=100*(nanmean(trai_res(trai_res(:,1)==2,11)));
        meanRT=(nanmean(trai_res(trai_res(:,1)==2,10)-(trai_res(trai_res(:,1)==2,8))));
        Screen('TextSize',w, InstrFont);
        DrawFormattedText(w,sprintf('Votre performance était:\n\n%2.1f %%(appui)\n\n%2.1f %%(non-appui)\n\n%1.2fs (temps de réponse)\n\nAppuyez sur une touche pour continuer',perfGO,perfNOGO,meanRT), 'center', 'center', [255 255 255]);%
        Screen('Flip',w);
        KbWait(-1);
        KbReleaseWait(-1);
        Screen('Flip',w);
        
        if flag_PPort
            SendTrigger(trig_end);
        end
    end
    Screen('Flip',w);
end
%% Randomize blocks and sequences
block_type      = [2 2 2 2]; % Modified to 4 blocks
set_images      = [3 3 3 3];
expe_sampling   = 1;
max_probe_jitter= 30;
min_probe_jitter= 40;
block_perm      = randperm(length(block_type));
if flag_debug==0
    number_probes   = 10;
    num_missprobes  = 0;
else
    number_probes   = 2;
    num_missprobes  = 0;
end
if flag_escp==1
    Screen('TextSize',w, InstrFont);
    DrawFormattedText(w, 'Interruption de la session...', 'center', 'center', [255 255 255]);
    Screen('Flip',w);
    WaitSecs(1);
    Screen('Flip',w);
else
    Screen('TextSize',w, InstrFont);
    DrawFormattedText(w, 'Prêt-e à commencer?\n\nAppuyez sur une touche quand vous êtes prêt·e', 'center', 'center', [255 255 255]);
    Screen('Flip',w);
    KbWait(-1);
    KbReleaseWait(-1);
    Screen('Flip',w);
end


%% Init Results variables
test_res=[];
probe_res=[];
nblock=0;
maxblock=length(block_perm);

if flag_PPort
            SendTrigger(trig_startBlock);
end 

%% HERE STARTS THE LOOP ACROSS BLOCKS. RERUN THIS SECTION IF CRASHES DURING TEST
while nblock < maxblock && flag_escp==0
    nblock=nblock+1;
    
    % start block
    Screen('Flip',w);
    this_block      = block_perm(nblock);
    this_blockcond  = block_type(block_perm(nblock));
    thiset          = set_images(block_perm(nblock));
    Screen('TextSize',w, InstrFont);
    DrawFormattedText(w, sprintf('Partie %g terminée %g\n\nAppuyez sur la barre d''espace quand vous etes prêt-e',nblock,length(block_perm)), 'center', 'center', [255 255 255]);
    Screen('Flip',w);
    KbWait(-1);
    KbReleaseWait(-1);
    Screen('Flip',w);
    all_tstartblock(nblock)=GetSecs;

    if expe_sampling==1
        probe_intervals = rand(1,number_probes+num_missprobes+2)*max_probe_jitter+min_probe_jitter;
        %         missingprobes=[ones(1,num_missprobes) zeros(1,number_probes+2)];
        %         missingprobes=missingprobes(randperm(length(missingprobes)));
        %         while flag_debug==0 && (missingprobes(1)==1 || missingprobes(end)==1 || diff(find(missingprobes))==1)
        %             missingprobes=missingprobes(randperm(length(missingprobes)));
        %         end
        probe_times = cumsum(probe_intervals);
        %         probe_times(missingprobes==1)=[];
        %         probe_intervals=diff([0 probe_times]); %(missingprobes==1)=[];
        this_probe=1; this_probetime=all_tstartblock(nblock)+probe_intervals(this_probe);
        this_probe_count=1;
    end
    
    %%%%%% call function for SART
    display_SART_SLHIP_v1;
    ListenChar(0);
    
    %     %%%%% Redo calibration every two blocks
    %     if ismember(nblock,2:2:length(block_perm)-1) && flag_EyeLink
    %         nCalib=nCalib+1;
    %         WaitSecs(1);
    %         DrawFormattedText(w, 'Press any key\n\nTo start the recalibration', 'center', 'center', [255 255 255]);
    %         Screen('Flip',w);
    %         KbWait(-1);
    %         KbReleaseWait(-1);
    %         Screen('Flip',w);
    %
    %         Eyelink('trackersetup');
    %         % do a final check of calibration using driftcorrection
    %         Eyelink('dodriftcorrect');
    %
    %         Eyelink('Message', sprintf('calib_%g',nCalib));
    %     end
end
all_tendexpe=GetSecs;
ListenChar(0);
Screen('Flip',w);
Screen('TextSize',w, InstrFont);
if  flag_escp==0
    DrawFormattedText(w, 'Bravo!\n\nVous avez terminé!\n\nMerci d''avoir participé!', 'center', 'center', [255 255 255]);
    Screen('Flip',w);
    WaitSecs(5);
end

%%
if flag_PPort
    SendTrigger(trig_end);
end 
% Save results and close Tobii/EEG
all_GrandEnd=GetSecs;

save_path=[pwd filesep '../ExpeResults/'];
if flag_escp==0
    save(sprintf('%s/wanderIM_behavres_s%s_%s',save_path,subjectID,expstart),'trai_res','test_res','probe_res','SubjectInfo','all_*');
else
    save(sprintf('%s/wanderIM_behavres_s%s_%s_ABORTED',save_path,subjectID,expstart),'trai_res','test_res','probe_res','SubjectInfo','all_*');
end
Screen('CloseAll')
if flag_EyeLink==1
    Eyelink('stoprecording');
    Screen(w,'close');
    Eyelink('closefile');
    if flag_escp==0
        newsaveEyeLink_eyet=sprintf('%s/wanderIM_eyelink_s%s_%s.edf',save_path,subjectID,expstart);
    else
        newsaveEyeLink_eyet=sprintf('%s/wanderIM_eyelink_s%s_%s_ABORTED.edf',save_path,subjectID,expstart);
    end
    Eyelink('ReceiveFile',saveEyeLink_eyet,newsaveEyeLink_eyet);
    
    Eyelink('shutdown');
else
    Screen('CloseAll')
end
ShowCursor;

%% Score end task 
% 
% fprintf('Performance sur la SART :\n\t - essais go correct: %g%%\n\t - essais nogo correct : %g %%\n\t - temps de réaction moyen : %g s\n\n'...
%     100*nanmean(test_res(:,12)),100*nanmean(test_res(:,11)),nanmean(test_res(~isnan(test_res(:,12)),10)-test_res(~isnan(test_res(:,12)),8)))
% 

