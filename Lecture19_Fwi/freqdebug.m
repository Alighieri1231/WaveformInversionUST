clear
clc

% Load Functions
addpath(genpath('Functions'));

% Load Problem Data
load('RecordedData.mat', 'x', 'y', 'C', ...
    'x_circ', 'y_circ', 'f', 'REC_DATA');
numElements = numel(x_circ); % Number of Transducer Elements
assert(numElements == numel(y_circ));

% Which Subset of Transmits to Use
dwnsmp = 1; % can be 1, 2, or 4 (faster with more downsampling)
            % NOTE: dwnsmp = 1 to get the results in the paper
tx_include = 1:dwnsmp:numElements;
REC_DATA = REC_DATA(tx_include,:); 

% Extract Subset of Signals within Acceptance Angle
numElemLeftRightExcl = 31;
elemLeftRightExcl = -numElemLeftRightExcl:numElemLeftRightExcl;
elemInclude = true(numElements, numElements);
for tx_element = 1:numElements 
    elemLeftRightExclCurrent = elemLeftRightExcl + tx_element;
    elemLeftRightExclCurrent(elemLeftRightExclCurrent<1) = numElements + ...
         elemLeftRightExclCurrent(elemLeftRightExclCurrent<1);
    elemLeftRightExclCurrent(elemLeftRightExclCurrent>numElements) = ...
        elemLeftRightExclCurrent(elemLeftRightExclCurrent>numElements) - numElements;
    elemInclude(tx_element,elemLeftRightExclCurrent) = false;
end