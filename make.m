% Okba BEKHELIFI <okba.bekhelifi@univ-usto.dz>
% 11-20-2017
% This make.m is for MATLAB under Windows
function make()
try
    mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims libsvmread.c
    mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims libsvmwrite.c

    mex CFLAGS="\$CFLAGS -std=c99" -I.. -largeArrayDims svm_train_plus.c svm.cpp svm_model_matlab.c
    mex CFLAGS="\$CFLAGS -std=c99" -I.. -largeArrayDims svm_predict_plus.c svm.cpp svm_model_matlab.c
catch err
    fprintf('Error: %s failed (line %d)\n', err.stack(1).file, err.stack(1).line);
    disp(err.message);
    fprintf('=> Please check README for detailed instructions.\n');
end
