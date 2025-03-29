function ShowCentroidRangePredictions(ui8Image, strCentroidRangePredictions, strImageLabels)

dCentroidPrediction         = double(strCentroidRangePredictions.Predictions(1:2));
dRangePrediction            = double(strCentroidRangePredictions.Predictions(3));
dApparentRadiusPrediction   = double(strCentroidRangePredictions.Predictions(4));

% Show output
figure();
imshow(ui8Image);
hold on;
plot(dCentroidPrediction(2), dCentroidPrediction(1), 'rx', 'DisplayName', 'Centre of figure Prediction')

% Draw circle with apparent radius (gently offered by GPT4.0)
dTheta = linspace(0, 2*pi, 100); % Angles for the circle
dxCircle = dApparentRadiusPrediction * cos(dTheta) + dCentroidPrediction(2);
dyCircle = dApparentRadiusPrediction * sin(dTheta) + dCentroidPrediction(1);
plot(dxCircle, dyCircle, 'g-', 'DisplayName', 'Predicted conic');

dxCircle_GT = strImageLabels.dRadiusInPix * cos(dTheta) + strImageLabels.dCentroid(2);
dyCircle_GT = strImageLabels.dRadiusInPix * sin(dTheta) + strImageLabels.dCentroid(1);
plot(strImageLabels.dCentroid(2), strImageLabels.dCentroid(1), 'bo', 'DisplayName', 'Centre of figure GroundTruth')
plot(dxCircle_GT, dyCircle_GT, 'b-', 'DisplayName', 'GroundTruth conic');

% Add text to display range prediction
textPosition1 = [10, 10]; % Position for text on the image
textString1 = sprintf(['Predictions:\n' ...
    '  Centroid: [%.2f, %.2f]\n' ...
    '  Apparent Radius: %.2f\n' ...
    '  Range: %.2f'], ...
    dCentroidPrediction(1), dCentroidPrediction(2), ...
    dApparentRadiusPrediction, dRangePrediction);
text(textPosition1(1), textPosition1(2), textString1, 'Color', 'yellow', 'FontSize', 10, 'FontWeight', 'bold', 'VerticalAlignment', 'top');

% Add text to display labels from strImageLabels
textPosition2 = [10, 95]; % Position for the labels text below the range prediction
textString2 = sprintf(['Labels:\n' ...
    '  Centroid: [%.2f, %.2f]\n' ...
    '  Radius in Pixels: %.2f\n' ...
    '  Range in Radii: %.2f\n' ...
    '  Reference Radius: %.2f'], ...
    strImageLabels.dCentroid(1), strImageLabels.dCentroid(2), ...
    strImageLabels.dRadiusInPix, strImageLabels.dRangeInRadii, ...
    strImageLabels.dRefRadius);
text(textPosition2(1), textPosition2(2), textString2, 'Color', 'cyan', 'FontSize', 10, 'FontWeight', 'bold', 'VerticalAlignment', 'top');


legend('show'); % Show legend
hold off;

end
