function objFig = ShowFeatureMatchingsPredictions(ui8Image1, ui8Image2, dMatchedPoints1, dMatchedPoints2, kwargs)
arguments
    ui8Image1 {mustBeNumeric, mustBeNonempty}
    ui8Image2 {mustBeNumeric, mustBeNonempty}
    dMatchedPoints1 (:,2) double {mustBeNonempty}
    dMatchedPoints2 (:,2) double {mustBeNonempty}
end
arguments
    kwargs.bUseBlackBackground      (1,1) logical = false;
    kwargs.objFig                   {mustBeA(kwargs.objFig, ["double", "matlab.ui.Figure"])} = 0;
    kwargs.charFigTitle             string = "Feature Matches";
    kwargs.plotMatchColor           (1,:) char = 'g';
    kwargs.plotKpsColor1            (1,:) char = 'r';
    kwargs.plotKpsColor2            (1,:) char = 'b';
    kwargs.matchLineWidth           (1,1) double = 0.75;
    kwargs.markerSize               (1,1) double = 3;
    kwargs.markerLineWidth          (1,1) double = 3;
    kwargs.bEnableLegend            (1,1) logical = true;
    kwargs.legendLocation           (1,:) string = "north";
end

%% Create or use provided figure and set background/text color
if kwargs.objFig == 0
    objFig = figure('Renderer', 'painters', 'Position', [100, 100, 1000, 500]);
    if kwargs.bUseBlackBackground
        set(objFig, 'Color', 'k');
        charTextColor = 'w';
    else
        charTextColor = 'k';
    end
else
    objFig = kwargs.objFig;
    charTextColor = objFig.Color;
end

figure(objFig)
hold on

%% Display the combined image (side-by-side)
dCombinedImage = [ui8Image1, ui8Image2];
imshow(dCombinedImage);
axis image;  % Maintain aspect ratio

%% Calculate offset for the second image and plot matches
offset = size(ui8Image1, 2);

% Prepare data for vectorized line drawing (each column is a pair of x/y coordinates)
dXcoord = [dMatchedPoints1(:,1), dMatchedPoints2(:,1) + offset]';
dYcoord = [dMatchedPoints1(:,2), dMatchedPoints2(:,2)]';
line(dXcoord, dYcoord, 'Color', kwargs.plotMatchColor, 'LineWidth', kwargs.matchLineWidth);
hold on

%% Plot keypoints using scatter for clarity
hKey1 = scatter(dMatchedPoints1(:,1), dMatchedPoints1(:, 2), kwargs.markerSize^2, ...
    kwargs.plotKpsColor1, "x", ...
    'LineWidth', kwargs.markerLineWidth);

hKey2 = scatter(dMatchedPoints2(:,1) + offset, dMatchedPoints2(:,2), kwargs.markerSize^2, ...
    kwargs.plotKpsColor2, "x", ...
    'LineWidth', kwargs.markerLineWidth);

%% Create dummy handle for match lines to build legend
hLine = plot(nan, nan, 'Color', kwargs.plotMatchColor, 'LineWidth', kwargs.matchLineWidth);

%% Add title and legend if enabled
title(kwargs.charFigTitle, 'Color', charTextColor);
if kwargs.bEnableLegend
    legend([hLine, hKey1, hKey2], {'Matches', 'Keypoints in Image 1', 'Keypoints in Image 2'}, 'Location', kwargs.legendLocation);
end

hold off;
end
