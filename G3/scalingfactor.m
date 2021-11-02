clear
close all
clc

calib=[];
freqg=[]

Files=dir('*.txt');
for k=1:length(Files)
   FileNames=Files(k).name
   
   data=[];
   temp=importdata(FileNames);
   data=temp.data;

  
    L = 30;   % ?©ì¶”êµ? 60m, ?œì‹¤êµ? 45m
    fr=[4 5];

    strain_loc=[8 15 22];% Location of strain gage!ca
    acc_loc=[15];

    

    acc = data(:,4);
    str = [-data(:,5), -data(:,6), data(:,7)];

    acc = detrend(acc);
    str = detrend(str);

    % Data filtering
    loc=find(abs(acc)>80)
    for i=1:length(loc)

        acc(loc(i)-8:loc(i)+8,:)=0;
        str(loc(i)-8:loc(i)+8,:)=0;
    end

    fs = 100;
    N = length(acc);
    time = (0:N-1)/fs;

   
    acc = acc;
    strain = str;
    
    dynamic=1:length(acc);
    % cpsd
    acc=acc(dynamic,:)*9.81;
    % acc=acc(dynamic,:);
    strain=strain(dynamic,:);


    strain(:,1)=strain(:,1)-mean(strain(:,1));
    strain(:,2)=strain(:,2)-mean(strain(:,2));
    strain(:,3)=strain(:,3)-mean(strain(:,3));


    %% NEUTRAL AXIS
    % acc, strain 1:5 channel, 100 Hz
    if size(strain,2)>size(strain,1)
        strain=strain';
    end

    nfft=2^12;
    acc = acc;
    strain=strain(:,[1 2 3]);

    %% Calibration
    tim=(0:length(acc)-1)/100;
    % figure;
    % plot(tim,acc(:,1)/10);
    % xlabel('Time (sec)');ylabel('Acceleration (mg)')
    % ylim([-20 20])
    % subplot(1,2,2);
    % plot(tim,strain(:,2));
    % xlabel('Time (sec)');ylabel('Strain (us)')


    if 1
        dt = 1/fs;
        Nw = 2.68;
        y = 1;
        nmodes = 1;
        us= strain2disp(strain, L, strain_loc, acc_loc, y, nmodes);    

    end



    % Calibration factor
    d1 = us;
    a0 = acc;
    [b,a]=butter(5,9/50,'low');
    d1=filtfilt(b,a,d1);
    a0=filtfilt(b,a,a0);
    d1=resample(d1,1,4);
    a0=resample(a0,1,4);
    fs=25;
    nfft=2^12;
    [Sd_strn,freq] = cpsd((d1(:,1)),(d1(:,1)),boxcar(nfft),nfft/2,nfft,fs);
    [Sa_meas,freq] = cpsd(detrend(a0(:,1)),detrend(a0(:,1)),boxcar(nfft),nfft/2,nfft,fs);
    % figure;plot(freq,Sa_meas);hold on;plot(freq,Sd_dtrn)
    idxFreqRange = find(fr(1)<freq & freq<fr(2));
    psdMax = max(Sa_meas(idxFreqRange));
    idxFreqPeak = find(Sa_meas==psdMax);
    freqPeak = freq(idxFreqPeak)
    %figure;semilogy(freq,Sa_meas);legend('ACC')
    %figure;semilogy(freq,(1./(2*pi*freq).^4).*Sa_meas);hold on;semilogy(freq,Sd_strn);title('Before calibration')
    caly=sqrt(1/(2*pi*freqPeak)^4*max(Sa_meas(idxFreqRange))/max(Sd_strn(idxFreqRange)));
    % caly=sqrt((1/(2*pi*freqPeak))^4*max(Sa_meas(idxFreqRange))/max(Sd_strn(idxFreqRange)));
    calib=[calib caly];
    freqg=[freqg freqPeak];
end
%%
figure;plot(calib)
figure;plot(freqg)

