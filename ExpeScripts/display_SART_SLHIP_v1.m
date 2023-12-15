%% Randomise order of trials
all_seq=randperm(9);
while all_seq(1)==3
    all_seq=randperm(9);
end
for nseq=1:200
    rand_seq=randperm(9);
    while rand_seq(1)==3 || rand_seq(end)==3 || rand_seq(1)==all_seq(end)% not target on edge
        rand_seq=randperm(9);
    end
    all_seq=[all_seq rand_seq];
end
DrawFormattedText(w, 'Préparez-vous à commencer la tâche\n\n Placez votre main dominante sur la barre espace', 'center', 'center', [255, 255, 255]);
Screen('Flip',w);
WaitSecs(2);
if flag_PPort
    SendTrigger(trig_startBlock);
end
if flag_EyeLink
    Eyelink('Message', sprintf('B%g',nblock));
end
% imge_indexes=pertask_imge_indexes{thiset};

%% play SART
%%% Init
starttra=GetSecs;
ntrial=1;
resprec=0;
lastpress=0;
block_starttime(nblock)=GetSecs;
% Start with 3s of fixation cross
Screen('DrawLines', w, [cross_xCoords; cross_yCoords], cross_lineWidthPix, cross_colour, [wx, wy]./2, 2);
Screen('Flip',w);
while GetSecs<starttra+2
end
if flag_PPort
    SendTrigger(trig_startTrial+this_seq_trial);
end
if flag_EyeLink
    Eyelink('Message', sprintf('B%g_T%g_S%g',nblock,ntrial,this_seq_trial));
end
if flag_1diodes % at the begining of each probe, turn the din1 to white
    Screen('FillPoly', w ,[1 1 1]*255, din2_pos);
end
dur_face_presentation_rand=dur_face_presentation + dur_face_presentation_jitter*rand;

this_seq_trial=all_seq(ntrial);

Screen('TextSize',w, letterSize);

DrawFormattedText(w,num2str(this_seq_trial),'center','center',cross_colour,[],[],[],[],[]);
Screen('Flip',w);
stimonset=GetSecs;
previousflip=stimonset; 
count=1;
thisresptime=NaN;
thisresp=NaN;

while this_probe_count<=number_probes && flag_escp==0

    resprec=0;
    resprec=0;
    while GetSecs<stimonset+dur_face_presentation_rand && resprec==0
        [keyIsDown,keySecs, keyCode,deltaSecs] = KbCheck(-1);
        if keyIsDown && resprec==0
            thisresp=find(keyCode); thisresp=thisresp(1);
                thisresptime=keySecs;
                resprec=1;
                lastpress=1;
                if flag_PPort
                    SendTrigger(trig_response);
                end
        end
    end
    while GetSecs<stimonset+dur_face_presentation_rand
    end

    if GetSecs>this_probetime
        probe_SART_SLHIP_v1;
        this_probe_count=this_probe_count+1;
        this_probe=this_probe+1;
        this_probetime=GetSecs+probe_intervals(this_probe_count);
    end
    if GetSecs>stimonset+dur_face_presentation_rand % update face identity
        this_nogo=NaN;
        this_go=NaN;
        if this_seq_trial==TargetID && isnan(thisresp)
            this_nogo=1;
        elseif  this_seq_trial==TargetID && ~isnan(thisresp) %&& strcmp(KbName(thisresp(1)),'space')
            this_nogo=0;
        end
        if this_seq_trial~=TargetID && isnan(thisresp)
            this_go=0;
        elseif  this_seq_trial~=TargetID && ~isnan(thisresp) %&& strcmp(KbName(thisresp(1)),'space')
            this_go=1;
        end
        if thisresp==KbName(AbortKey)
            flag_escp=1;
        end
        fprintf('NOGO: %g | GO: %g\n',this_nogo,this_go)
        test_res=[test_res; [nblock this_blockcond thiset ntrial this_seq_trial TargetID thisresp stimonset dur_face_presentation_rand thisresptime  this_nogo this_go]];
        dur_face_presentation_rand=dur_face_presentation + dur_face_presentation_jitter*rand;

        ntrial=ntrial+1;
        this_seq_trial=all_seq(ntrial);
        stimonset=GetSecs;
        thisresptime=NaN;
        thisresp=NaN;
        if flag_PPort
            SendTrigger(trig_startTrial+this_seq_trial);
        end
        if flag_EyeLink
            Eyelink('Message', sprintf('B%g_T%g_S%g',nblock,ntrial,this_seq_trial));
        end
        if flag_1diodes % at the begining of each probe, turn the din1 to white
            Screen('FillPoly', w ,[1 1 1]*255, din2_pos);
        end
    end
    Screen('TextSize',w, letterSize);
    DrawFormattedText(w,num2str(this_seq_trial),'center','center',cross_colour,[],[],[],[],[]);
    Screen('Flip',w);
end


%% end block
Screen('Flip',w);
if flag_PPort
    SendTrigger(trig_endBlock);
end
if flag_EyeLink
    Eyelink('Message', sprintf('EB%g',nblock));
end

WaitSecs(3);
