function [cherries conditions cherries_mu sr] = ...
        pick_cherries(whichEvent, cherriesname, segstart,segstop, ...
                      saveflag, TrialInfosName, FinalEventsName, verbose)
    
%% function [cherries conditions cherries_mu] = ...
%        pick_cherries(whichEvent, cherriesname, segstart,segstop, ...
%                      saveflag, TrialInfosName, FinalEventsName)

if ~exist('TrialInfosName', 'var') || isempty(TrialInfosName)
    TrialInfosName = 'TrialInfos.mat';
end

if ~exist('FinalEventsName', 'var') || isempty(FinalEventsName)
    FinalEventsName = 'finalevents.mat';
end

if ~exist('verbose', 'var')
    verbose = true;
end

load(TrialInfosName);
load(FinalEventsName);


if exist('cluster_info.mat')
    load cluster_info
end

if ~exist('whichEvent', 'var')
    whichEvent = 1; % stim onset
end

if ~exist('cherriesname', 'var')
    cherriesname = 'cherries.mat';
end

if ~exist('segstart', 'var') || isempty(segstart)
    segstart = -1000; 
end

if ~exist('segstop', 'var') || isempty(segstop)
    segstop = 2000;
end

if ~exist('saveflag', 'var') || isempty(saveflag)
    saveflag = true;
end

%% try to find out the sampling rate of the recording
f = fopen('CSC1.ncs');
header_size = 16*1024;
[header num] = fread(f, header_size, '*char');
header = header';
magic = '######## Neuralynx Data File Header';

if ~strcmp(header(1:length(magic)), magic)
    fclose(f);
    error('could not determine sampling rate' , ...
        'file CSC1.ncs does not contain magic line');
end

header = deblank(header);
content = textscan(header, '%s %s %s %s %s %s %s %s%*[^\n]');
fclose(f);
sridx = find(strcmp('-SamplingFrequency', content{1}));
sr = str2num(content{2}{sridx});

%% set some parameters
ditch_artefacts = true;
origchnnames = textread('ChannelNames.txt', '%s');
channels = 1:numel(origchnnames);
assert(length(origchnnames) == length(channels));
nchans = length(channels);
if isfield(trials, 'verb')
    ntrials = length(trials.verb);
elseif isfield(trials, 'name')
     ntrials = length(trials.name);
elseif isfield(trials, 'word')
    ntrials = length(trials.word); %since the last repetition has not been saved in trials as empty cell
end
Event_Time = Event_Time(ismember(Event_Value, whichEvent));
Event_Value = Event_Value(ismember(Event_Value, whichEvent));

conditions = trials;
conditions.onset_time = Event_Time./1000;
cc = 1;

%% process channels 
for c = 1:nchans
    if verbose
        fprintf('%d-%s ', c,  origchnnames{c}(1:end-4))
    end
    
    filename=sprintf('CSC%d_spikes.mat',c);
    if ~exist(filename,'file')
        if verbose
            fprintf(':spikes not found! ')
            disp( ' ');
        end
        continue; 
    end
    load(filename);
    filename=sprintf('times_CSC%d.mat',c);
    if ~exist(filename,'file')
        if verbose
            fprintf(':times not found! ')
            disp( ' ');
        end
        continue; 
    end

    load(filename)
    
    %% multi unit
    cherries_mu(c).channr = c;
    cherries_mu(c).chnname = origchnnames{c}(1:end-4);
    cherries_mu(c).site=origchnnames{c}(1:end-5);
    cherries_mu(c).allspiketimes = index_ts * 1000;
    if verbose
        fprintf('AllMU ')
    end
   
    for t = 1:ntrials
        idx = conditions.onset_time(t)+segstart < index_ts & ...
              conditions.onset_time(t)+segstop  > index_ts;        
    
        cherries_mu(c).trial{t} = index_ts(idx) - ...
            conditions.onset_time(t);        

    end
    
    %% sorted
    uclas = unique(cluster_class(:,1));
    uclas(uclas == 0) = []; % remove unassigned cluster
    nclasses = length(uclas);
    if size(cluster_class,1) ~= numel(index_ts)
        if verbose
            fprintf(':cluster_class~=index_ts! ')
            disp(' ');
        end
        
        continue
    end
    
    for class = 1:nclasses
        if exist('cluster_info', 'var') && ...
                ditch_artefacts && ...
                ~isempty(cluster_info{1,c}) && ...
                cluster_info{1,c}(class) < 0
            
            continue
        end
        
        clsidx = cluster_class(:,1) == uclas(class);
        ts = index_ts(clsidx);
        for t = 1:ntrials
            idx = conditions.onset_time(t)+segstart < ts & ...
                  conditions.onset_time(t)+segstop  > ts;        

            
            cherries(cc).trial{t} = ts(idx) - ...
                conditions.onset_time(t);        
            
        end
        
        cherries(cc).classno = uclas(class);
        cherries(cc).allspiketimes = ts;

        if exist('cluster_info', 'var')
            if uclas(class) == 0
                cherries(cc).kind = 'Unassigned';
            else
                ci = cluster_info{1,c}(class);
                switch ci
                  case 1
                    cherries(cc).kind = 'MU';
                  case 2
                    cherries(cc).kind = 'SU';
                  case -1 
                    cherries(cc).kind = 'Artifact';
                end
            end
        else
            cherries(cc).kind = 'Unknown';
        end
        if verbose
            fprintf('%s ', cherries(cc).kind)
        end
        cherries(cc).channr = c;
        cherries(cc).chnname = origchnnames{c}(1:end-4);
        cherries(cc).site=origchnnames{c}(1:end-5);
        cherries(cc).meanspike = mean(spikes(clsidx,:));
        
        cc = cc + 1;
        clear ts; 
        
    end
    if verbose
        disp(' ');
    end
    
end

if saveflag
    save(cherriesname, 'cherries', 'cherries_mu', 'conditions', 'sr');
end
disp(' ');
