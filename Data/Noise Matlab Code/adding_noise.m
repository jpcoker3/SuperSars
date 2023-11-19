clc
clear

%set noise mean
m=0.2;

for i=1:8

    if i==1
% %2S1
inputFolder='C:\GitHub\SuperSars\Data\Padded_imgs\2S1\';
outputFolder='C:\Noise\Gaussian0.2\2S1\';

    elseif i==2
% %BRDM_2
inputFolder='C:\GitHub\SuperSars\Data\Padded_imgs\BRDM_2\';
outputFolder='C:\Noise\Gaussian0.2\BRDM_2\';
    elseif i==3
% %BTR_60
inputFolder='C:\GitHub\SuperSars\Data\Padded_imgs\BTR_60\';
outputFolder='C:\Noise\Gaussian0.2\BTR_60\';
    elseif i==4
% %D7
inputFolder='C:\GitHub\SuperSars\Data\Padded_imgs\D7\';
outputFolder='C:\Noise\Gaussian0.2\D7\';
    elseif i==5
% %SLICY
inputFolder='C:\GitHub\SuperSars\Data\Padded_imgs\SLICY\';
outputFolder='C:\Noise\Gaussian0.2\SLICY\';
    elseif i==6
% %T62
inputFolder='C:\GitHub\SuperSars\Data\Padded_imgs\T62\';
outputFolder='C:\Noise\Gaussian0.2\T62\';
    elseif i==7
%ZIL131
inputFolder='C:\GitHub\SuperSars\Data\Padded_imgs\ZIL131\';
outputFolder='C:\Noise\Gaussian0.2\ZIL131\';
    else
%ZSU_23_4
inputFolder='C:\GitHub\SuperSars\Data\Padded_imgs\ZSU_23_4\';
outputFolder='C:\Noise\Gaussian0.2\ZSU_23_4\';
    end


filePattern=fullfile(inputFolder,'*.JPG');
fileList=dir(filePattern);

%iterate through files in input folder and set ouput file names to be the
%same as the original file names, but output as PNG instead of JPG
%reference for code: https://matlab.fandom.com/wiki/FAQ#How_can_I_process_a_sequence_of_files.3F
for k = 1 : length(fileList)
    baseFileName = fileList(k).name;
    fullFileName = fullfile(fileList(k).folder, baseFileName);
    outputFileName = fullfile(outputFolder,baseFileName);
    pngoutputFileName= strrep(outputFileName,'JPG','PNG');
    I = imread(fullFileName);

    %convert rgb image to true B&W
    %required to ensure added noise is B&W 
    IG = im2gray(I);
    
    
    % Add Gaussian noise to image
    N=imnoise(IG,'gaussian',m);


    %convert grayscale back to rgb depth
    OrigFormat(:,:,1)=N;
    OrigFormat(:,:,2)=N;
    OrigFormat(:,:,3)=N;
  
    %save file with added noise added
    imwrite(OrigFormat,pngoutputFileName);
  
end

end

return
