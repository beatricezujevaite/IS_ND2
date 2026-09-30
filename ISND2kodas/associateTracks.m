function [matchedPairs, unmatchedTracks, unmatchedDetections] = ...
    associateTracks(trackBboxes, detectionBboxes, e, maxCost)
% associateTracks
%
% ByteTrack asociacijos etapas.
%
% Įvestis:
%   trackBboxes       - Kalman prognozuoti boxes
%   detectionBboxes  - YOLO detections
%   e                 - EIoU expansion
%   maxCost           - maksimalus leidžiamas cost
%
% Išvestis:
%   matchedPairs
%   unmatchedTracks
%   unmatchedDetections

nTracks = size(trackBboxes,1);
nDetections = size(detectionBboxes,1);

%% Tušti atvejai

if nTracks == 0

    matchedPairs = zeros(0,2);

    unmatchedTracks = zeros(0,1);

    unmatchedDetections = ...
        (1:nDetections)';

    return;

end

if nDetections == 0

    matchedPairs = zeros(0,2);

    unmatchedTracks = ...
        (1:nTracks)';

    unmatchedDetections = zeros(0,1);

    return;

end

%% --------------------------------------------------------
% EIoU cost matrix
% ---------------------------------------------------------

costMatrix = calculateEIoUCost( ...
    trackBboxes, ...
    detectionBboxes, ...
    e);

%% --------------------------------------------------------
% Hungarian assignment per MATLAB
% ---------------------------------------------------------

[matchedPairs, unmatchedTracks, unmatchedDetections] = ...
    assignDetectionsToTracks( ...
    costMatrix, ...
    maxCost);

end