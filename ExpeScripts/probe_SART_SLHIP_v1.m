%% Supervised questions
[wx, wy] = Screen('WindowSize', w);
Screen('FillRect', w, [0, 0, 0]);
Screen('Flip',w);
Screen('TextSize',w, InstrFont);
Screen('FillRect', w, [0, 0, 0]);
DrawFormattedText(w, 'STOP', 'center', 'center', [255, 255, 255]);
if flag_1diodes % at the begining of each probe, turn the din1 to white
    Screen('FillPoly', w ,[1 1 1]*255, din2_pos);
end

% Prepare a bip
if flag_bip
    InitializePsychSound;
    freq = 44100;
    pahandle = PsychPortAudio('Open', [], [], 0, freq, 1);
    [beep,samplingRate] = MakeBeep(440,0.5,freq);
    PsychPortAudio('FillBuffer', pahandle, beep);
    PsychPortAudio('Start', pahandle, 1, 0, 1);
end
startProbe=Screen('Flip',w);
if flag_PPort
    SendTrigger(trig_probestart);
end
if flag_EyeLink
    Eyelink('Message', sprintf('P%g',this_probe));
end
WaitSecs(0.5);
if flag_bip
    PsychPortAudio('Stop', pahandle);
    PsychPortAudio('Close', pahandle);
end
Screen('Flip',w);
KbReleaseWait(-1);
WaitSecs(0.5);


for nQ=1:length(supervised_questions)
    if nQ==2 && stateresp~=2
        probe_responses(nQ,:)=nan;
        continue;
    end
    Screen('FillRect', w, [0, 0, 0]);
    %     if flag_EEG
    %     Screen('FillPoly', w ,[1 1 1]*255, din2_pos);
    %     end
    DrawFormattedText(w, supervised_questions_headers{nQ}, 'center', 0.1*wy, [255, 0, 0]);

    DrawFormattedText(w, supervised_questions{nQ}, 'center', 'center', [255, 255, 255]);
    qflip=Screen('Flip',w);
    if flag_EyeLink
        Eyelink('Message', sprintf('P%g_Q%g',this_probe,nQ));

    end
    if flag_PPort
        SendTrigger(trig_startQuestion);
    end
    % Wait for response
    [keyIsDown, secs, keyCode, deltaSecs] = KbCheck(-1);
    while keyIsDown==0 || isempty(find(ismember(find(keyCode),myKeyMap(supervised_questions_acceptedanswers{nQ}))))
        [keyIsDown, secs, keyCode, deltaSecs] = KbCheck(-1);
    end


    KbReleaseWait(-1);
    keyPressed=find(keyCode);
    keyPressed=keyPressed(1);
    probe_responses(nQ,1)=keyPressed;
    probe_responses(nQ,2)=secs;
    probe_responses(nQ,3)=qflip;
    %%%% YOU SHOULD ADD MISSING PART HERE
    thiskeyCode=find(myKeyMap==keyPressed);
    if ~isempty(thiskeyCode)
        probe_responses(nQ,4)=thiskeyCode;
    else
        probe_responses(nQ,4)=NaN;
    end
    if nQ==1
        stateresp=probe_responses(nQ,4);
    end
    if flag_PPort
        SendTrigger(trig_startQuestion+keyPressed);
    end
    WaitSecs(0.5);
end
probe_res=[probe_res ; [this_probe this_probetime startProbe nblock this_blockcond ntrial probe_responses(:,1)' probe_responses(:,2)' probe_responses(:,3)' probe_responses(:,4)']];

if flag_PPort
    SendTrigger(trig_probeend);
end
if flag_1diodes % at the begining of each probe, turn the din1 to white
    Screen('FillPoly', w ,[1 1 1]*255, din2_pos);
    Screen('Flip',w);
    WaitSecs(0.5);
    Screen('Flip',w);
end
if flag_EyeLink
    Eyelink('Message', sprintf('EP%g',this_probe));
end

DrawFormattedText(w, 'Préparez-vous à reprendre la tâche\n\n Placez votre main dominante sur la barre espace', 'center', 'center', [255, 255, 255]);
Screen('Flip',w);
WaitSecs(2);
