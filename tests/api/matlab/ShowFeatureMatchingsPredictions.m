function ShowFeatureMatchingsPredictions(ui8Image1, ui8Image2, matchedPoints1, matchedPoints2, kwargs)
arguments
    ui8Image1 {mustBeNumeric, mustBeNonempty}
    ui8Image2 {mustBeNumeric, mustBeNonempty}
    matchedPoints1 (:,2) double {mustBeNonempty}
    matchedPoints2 (:,2) double {mustBeNonempty}
end
arguments
    kwargs.bUseBlackBackground      (1,1) logical = false;
    kwargs.objFig                   {mustBeA(kwargs.objFig, ["double", "matlab.ui.Figure"])} = 0;
    kwargs.charFigTitle             string = "Feature Matches";
    kwargs.plotMatchColor           (1,:) char = 'g';
    kwargs.plotKpsColor1            (1,:) char = 'r';
    kwargs.plotKpsColor2            (1,:) char = 'b';
    kwargs.matchLineWidth           (1,1) double = 1;
    kwargs.markerSize               (1,1) double = 5;
    kwargs.markerLineWidth          (1,1) double = 1.5;
    kwargs.bEnableLegend            (1,1) logical = true;
    kwargs.legendLocation           (1,:) string = "best";
end

%% Create or use provided figure and set background/text color
if kwargs.objFig == 0
    objFig = figure('Renderer', 'painters', 'Position', [100, 100, 1200, 400]);
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
gold on
%% Display the combined image (side-by-side)
combinedImage = [ui8Image1, ui8Image2];
imshow(combinedImage);
axis image;  % Maintain aspect ratio

%% Calculate offset for the second image and plot matches
offset = size(ui8Image1, 2);

% Prepare data for vectorized line drawing (each column is a pair of x/y coordinates)
x = [matchedPoints1(:,1), matchedPoints2(:,1) + offset]';
y = [matchedPoints1(:,2), matchedPoints2(:,2)]';
line(x, y, 'Color', kwargs.plotMatchColor, 'LineWidth', kwargs.matchLineWidth);

%% Plot keypoints using scatter for clarity
hKey1 = scatter(matchedPoints1(:,1), matchedPoints1(:,2), kwargs.markerSize^2, kwargs.plotKpsColor1, 'filled', 'LineWidth', kwargs.markerLineWidth);
hKey2 = scatter(matchedPoints2(:,1) + offset, matchedPoints2(:,2), kwargs.markerSize^2, kwargs.plotKpsColor2, 'filled', 'LineWidth', kwargs.markerLineWidth);

%% Create dummy handle for match lines to build legend
hLine = plot(nan, nan, 'Color', kwargs.plotMatchColor, 'LineWidth', kwargs.matchLineWidth);

%% Add title and legend if enabled
title(kwargs.charFigTitle, 'Color', charTextColor);
if kwargs.bEnableLegend
    legend([hLine, hKey1, hKey2], {'Matches', 'Keypoints in Image 1', 'Keypoints in Image 2'}, 'Location', kwargs.legendLocation);
end

hold off;
end
