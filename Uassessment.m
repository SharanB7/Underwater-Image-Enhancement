%setDir = fullfile('I:\outPut_restoration');
setDir = fullfile('C:\Users\PRIYAM RAJ BHARTI\OneDrive\Documents\image processing code\code\Results');
imds = imageDatastore(setDir,'FileExtensions',{'.jpeg'});
FileNames =  imds.Files;
numFiles =  size(FileNames,1);
Fianl = [];
for fileIndex = 1:numFiles
     filename = cell2mat(FileNames(fileIndex));
     img = imread(filename);
     uiqm = UIQM(img);
     Fianl(fileIndex) = uiqm;
end
MeanUiqmInitial = mean(Fianl);
